// Copyright 2026 Jakub Hlavnicka
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//! Recording from an input device — the other end of the app from playback.
//!
//! One take at a time, straight into memory: a recording is something a person
//! is standing in front of, seconds to minutes long, and the file it becomes is
//! written once at the end rather than streamed. What it becomes is the point —
//! the take is handed to Resynthesis, so a sound can go from a microphone to a
//! Fourier grid without the user ever naming a file.
//!
//! **Which is why the name is a timestamp.** Nobody stops a take to type a name,
//! and a "recording.wav" that the next take overwrites is worse than useless. So
//! the name is the moment the take *started*, down to the millisecond
//! ([`Recording::file_name`]): two takes a second apart cannot collide, they
//! sort into the order they were made, and the file says when it was made
//! without anything else having to remember.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// What the source select box shows for "whatever the system is set to".
pub const DEFAULT_INPUT: &str = "System default input";

/// How much a take reserves up front, in seconds. Beyond it the buffer grows in
/// the audio callback, which can cost a dropout; a minute covers the takes this
/// is for without holding a device's worth of memory for the ones it is not.
const RESERVE_SECS: usize = 60;

/// Longest take kept, in seconds. A recorder left running overnight is a
/// mistake, not a take, and this bounds what that mistake costs.
const MAX_SECS: usize = 30 * 60;

/// Every input device the host can offer, by name, the system default first.
///
/// Names are what the select box shows and what [`Recorder::start`] takes back,
/// so a device that cannot say its name is left out rather than offered under a
/// placeholder that would not resolve.
pub fn input_device_names() -> Vec<String> {
    let mut out = vec![DEFAULT_INPUT.to_string()];
    let host = cpal::default_host();
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            if let Ok(name) = d.name() {
                if !out.contains(&name) {
                    out.push(name);
                }
            }
        }
    }
    out
}

/// A finished take, still in memory.
pub struct Recording {
    /// Interleaved, `channels` per frame.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// When the take *started* — the name is made from this, not from when it
    /// stopped, so a file sorts by when its first sample was captured.
    pub started: chrono::DateTime<chrono::Local>,
    /// The device it came from, for the status line.
    pub device: String,
    /// Whether [`MAX_SECS`] cut the take short.
    pub truncated: bool,
}

impl Recording {
    pub fn duration_secs(&self) -> f64 {
        let frames = self.samples.len() / self.channels.max(1) as usize;
        frames as f64 / self.sample_rate.max(1) as f64
    }

    /// The file this take is saved as: `recording_<when it started>.wav`, to the
    /// millisecond. Far finer than the half-second two takes could plausibly be
    /// started within, and every character of it is safe in a file name on every
    /// platform we build for.
    pub fn file_name(&self) -> String {
        format!(
            "recording_{}.wav",
            self.started.format("%Y-%m-%d_%H-%M-%S.%3f")
        )
    }
}

/// A take in progress. Dropping it stops the stream and discards what was
/// captured; [`Recorder::finish`] is what keeps it.
pub struct Recorder {
    /// Held for its lifetime — dropping the stream is what stops the device.
    _stream: cpal::Stream,
    buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    started: chrono::DateTime<chrono::Local>,
    device: String,
}

impl Recorder {
    /// Open `device` (by the name [`input_device_names`] gave, or `None` for the
    /// system default) and start capturing.
    pub fn start(device: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device.filter(|n| *n != DEFAULT_INPUT) {
            Some(name) => host
                .input_devices()
                .context("cannot list input devices")?
                .find(|d| d.name().is_ok_and(|n| n == name))
                .with_context(|| format!("input device '{name}' is not there any more"))?,
            None => host
                .default_input_device()
                .context("no input device — nothing is set as the system's recording source")?,
        };
        let name = device.name().unwrap_or_else(|_| DEFAULT_INPUT.to_string());
        let supported = device
            .default_input_config()
            .with_context(|| format!("'{name}' does not say what format it records in"))?;
        let sample_rate = supported.sample_rate().0;
        let channels = supported.channels();
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();

        let cap = RESERVE_SECS * sample_rate as usize * channels as usize;
        let limit = MAX_SECS * sample_rate as usize * channels as usize;
        let buffer = Arc::new(Mutex::new(Vec::with_capacity(cap)));
        let err = |e| log::error!("Recording error: {e}");

        // One closure body, three sample formats: a device hands over whatever
        // it natively records in, and everything downstream of here is f32.
        macro_rules! stream {
            ($t:ty, $to_f32:expr) => {{
                let sink = buffer.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[$t], _: &cpal::InputCallbackInfo| {
                        let Ok(mut buf) = sink.lock() else { return };
                        if buf.len() >= limit {
                            return;
                        }
                        let room = limit - buf.len();
                        buf.extend(data.iter().take(room).copied().map($to_f32));
                    },
                    err,
                    None,
                )
            }};
        }
        let stream = match format {
            cpal::SampleFormat::F32 => stream!(f32, |s: f32| s),
            cpal::SampleFormat::I16 => stream!(i16, |s: i16| s as f32 / 32768.0),
            cpal::SampleFormat::U16 => stream!(u16, |s: u16| (s as f32 - 32768.0) / 32768.0),
            other => anyhow::bail!("'{name}' records in {other}, which is not supported"),
        }
        .with_context(|| format!("cannot open '{name}' for recording"))?;
        stream.play().context("cannot start the recording stream")?;

        log::info!("Recording from '{name}': {sample_rate} Hz, {channels} ch, {format}");
        Ok(Self {
            _stream: stream,
            buffer,
            sample_rate,
            channels,
            // Taken *after* the stream is running, so the name names a moment
            // the device was actually capturing.
            started: chrono::Local::now(),
            device: name,
        })
    }

    /// How long the take is so far, for the counter on the button.
    pub fn secs(&self) -> f64 {
        let frames =
            self.buffer.lock().map(|b| b.len()).unwrap_or(0) / self.channels.max(1) as usize;
        frames as f64 / self.sample_rate.max(1) as f64
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    /// Stop and take what was captured.
    pub fn finish(self) -> Recording {
        let samples = self
            .buffer
            .lock()
            .map(|mut b| std::mem::take(&mut *b))
            .unwrap_or_default();
        let truncated =
            samples.len() >= MAX_SECS * self.sample_rate as usize * self.channels.max(1) as usize;
        Recording {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            started: self.started,
            device: self.device.clone(),
            truncated,
        }
    }
}

/// Where takes are written: `~/ReSynthRecordings`, created on demand.
///
/// A fixed folder rather than a dialog per take — a recording is stopped with
/// one click and must not open a file picker over whatever the user is doing —
/// and a folder of their own rather than the working directory, which for this
/// app is wherever it happened to be launched from. The panel prints the full
/// path, so the answer to "where did it go" is on screen.
pub fn recordings_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("no home directory to put recordings in")?;
    let dir = PathBuf::from(home).join("ReSynthRecordings");
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir)
}

/// `~/GemstoneRecordings`, where takes went before the app was renamed from
/// Gemstone DAW — if it is there. Never created or written: it is only
/// searched, so a project that names a take by a path that no longer exists
/// still finds it by name.
pub fn legacy_recordings_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    let dir = PathBuf::from(home).join("GemstoneRecordings");
    dir.is_dir().then_some(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(ms: u32) -> Recording {
        Recording {
            samples: Vec::new(),
            sample_rate: 48_000,
            channels: 1,
            started: chrono::Local
                .with_ymd_and_hms(2026, 9, 6, 1, 42, 33)
                .unwrap()
                + chrono::Duration::milliseconds(ms as i64),
            device: "mic".to_string(),
            truncated: false,
        }
    }

    /// The name carries the start of the take finely enough that two takes
    /// cannot land on one file — half a second apart is the coarsest anyone
    /// could start two, and this resolves a thousandth.
    #[test]
    fn two_takes_half_a_second_apart_are_two_files() {
        assert_eq!(at(0).file_name(), "recording_2026-09-06_01-42-33.000.wav");
        assert_ne!(at(0).file_name(), at(500).file_name());
        assert_ne!(at(500).file_name(), at(501).file_name());
        // Sorting the folder is sorting by when each take was started.
        let mut names = [
            at(700).file_name(),
            at(200).file_name(),
            at(500).file_name(),
        ];
        names.sort();
        assert_eq!(names[0], at(200).file_name());
        assert_eq!(names[2], at(700).file_name());
    }

    /// Nothing in a take's name needs quoting or escaping on any platform we
    /// build for — a colon alone would make the name unwritable on Windows.
    #[test]
    fn the_name_is_safe_on_every_platform() {
        let name = at(123).file_name();
        assert!(
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || "_-.".contains(c)),
            "{name} has a character a file name cannot carry"
        );
    }

    /// A take really does come back as a `.wav` the app can read again — the
    /// whole point of the feature, and the one part of it no amount of unit
    /// testing of the name would catch. Skipped where there is no input device, or
    /// one that gives nothing.
    #[test]
    fn a_take_is_a_wav_the_app_can_decode() {
        let rec = match Recorder::start(None) {
            Ok(r) => r,
            Err(e) => {
                println!("no input device here ({e:#}) — nothing to record");
                return;
            }
        };
        std::thread::sleep(std::time::Duration::from_millis(400));
        let take = rec.finish();
        // A device that opens but stays silent (muted, idle, held by another
        // app) has nothing to write either — the same as having none.
        if take.samples.is_empty() {
            println!("the input device gave nothing in 400 ms — nothing to record");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(take.file_name());
        crate::audio::write_wav_i16(&path, &take.samples, take.channels, take.sample_rate).unwrap();
        let back = crate::audio::decode_audio_file(&path).unwrap();
        assert!(
            (back.duration_secs() as f64 - take.duration_secs()).abs() < 0.05,
            "wrote {:.3}s, read back {:.3}s",
            take.duration_secs(),
            back.duration_secs()
        );
        assert!(
            take.duration_secs() > 0.2,
            "only {:.3}s in 400 ms",
            take.duration_secs()
        );
    }

    /// The system default is always offered, whatever the machine has attached.
    #[test]
    fn the_device_list_always_offers_the_default() {
        let names = input_device_names();
        assert_eq!(names.first().map(String::as_str), Some(DEFAULT_INPUT));
        assert_eq!(
            names.iter().filter(|n| *n == DEFAULT_INPUT).count(),
            1,
            "the default is listed twice: {names:?}"
        );
    }
}
