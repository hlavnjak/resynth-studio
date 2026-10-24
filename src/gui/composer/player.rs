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
//! Real-time playback of a composition: one output stream, one plugin instance
//! per row, mixed with per-row gain.
//!
//! **One stream, many plugins.** An editor drives its own instance from its own
//! `cpal` stream; a composition needs several instruments summed inside one
//! callback on one sample clock, hence this second audio path.
//!
//! **The Composer loads its own instances** from each track's recipe
//! ([`PlaybackSource`]) rather than borrowing the editor's: a VST3 processor is
//! not re-entrant, so one already pulled by an editor's stream cannot also be
//! pulled by this one. The grid is snapshotted from the live editor when there
//! is one, so what plays is what the user is editing.
//!
//! **One instance serves as many rows as it can** ([`share_groups`]). An
//! instrument is polyphonic — the rows of a chord are one instrument playing
//! three notes, not three instruments — and an instance is not cheap: a LeSynth
//! grid and its rendered key buffers are tens of megabytes, and a sampler's
//! whole kit is more. A row therefore only gets an instance of its own when it
//! *needs* one: a different sound, a different level, or notes that would
//! collide with another row's on the same instance.
//!
//! The schedule is resolved to sample times up front, so the callback only walks
//! a sorted per-row cursor — no allocation or locking of its own.
//!
//! **A repeat picks up edits.** While it loops, the panel hands the transport
//! the composition again whenever it changes ([`CompositionPlayer::update_live`]);
//! the callback swaps it in at the loop point, so a pass is never rearranged
//! under itself. Only the schedules and gains change that way — which plugin
//! each row plays cannot, because loading one is not something an audio callback
//! can do.
//!
//! **Repeat** loops on the composition's *written* length — the longest row,
//! trailing silence included — not on the last note plus its release. The wrap is
//! cut at the exact sample, inside the device block if need be, so a loop does
//! not drift by up to a buffer every pass.
//!
//! **Export renders through the same voices** ([`render_offline`]): the loading,
//! the schedule and the per-block mix are shared with playback, so a `.wav` is
//! what the transport plays rather than a second implementation of it that can
//! drift. It runs as fast as the plugins allow — a LeSynth key renders
//! synchronously when its buffer is not ready, so no note can come out silent
//! for being asked for early.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crate::audio::{decode_audio_file, DecodedAudio, Limiter};
use crate::gui::registry::PlaybackSource;
use crate::audio::engine::declared_block_size;
use crate::plugin::{detect_format, BlockProcessor, PluginFormat, PluginInstance};
use crate::vst::{next_instance_token, Vst3Instance, Vst3Module};

/// Velocity every composed note is played at. The Composer has no velocity
/// control, and a mid-scale value keeps VSTs that map velocity to level audible
/// without slamming them.
const NOTE_VELOCITY: u8 = 100;

/// Extra time played after the last note-off so releases and (for LeSynth in
/// Analysis mode) the tail of a long note are not cut off by the transport
/// stopping exactly on the final event.
const TAIL_SECS: f64 = 1.5;

/// Fade at each end of a wav note. A file starts and ends wherever it happens
/// to, and a note shorter than the file cuts it mid-waveform; either way a step
/// straight from silence to the signal is a click, and five milliseconds is
/// short enough not to be heard as an attack.
const CLICK_FADE_SECS: f64 = 0.005;

/// One note on the timeline, already resolved to seconds.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PlannedNote {
    pub at_secs: f64,
    pub dur_secs: f64,
    pub pitch: u8,
    /// Where in the file the note starts, for a row that plays one — see
    /// [`crate::gui::composer::Item::start`]. Zero, and ignored, for a note on a
    /// plugin: an instrument has nothing to start into.
    pub start_secs: f64,
}

impl PlannedNote {
    /// Whether this note is sounding at the same time as `other` **on the same
    /// pitch** — the one thing two rows cannot do on one instance, because a
    /// note-off names only a pitch and would cut the other row's note.
    fn collides_with(&self, other: &Self) -> bool {
        self.pitch == other.pitch
            && self.at_secs < other.at_secs + other.dur_secs
            && other.at_secs < self.at_secs + self.dur_secs
    }
}

/// One note of a sample voice, resolved to the transport's sample clock: when it
/// starts, when it stops, and where in the file it plays from.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Hit {
    on: u64,
    off: u64,
    /// Seconds into the file — the frame's start slider. Kept in seconds rather
    /// than in source samples because a schedule is built without knowing which
    /// file it is for, and the rate is the file's, not the device's.
    start_secs: f64,
}

/// One composer row, ready to play.
pub struct RowPlan {
    /// The panel's row id, which is how a live edit finds the voice again.
    pub row_id: u64,
    pub source: PlaybackSource,
    pub gain: f32,
    pub notes: Vec<PlannedNote>,
}

/// One row of an edit made while the transport is running: everything about a
/// row that can be changed without loading a different plugin.
#[derive(Clone, PartialEq, Debug)]
pub struct RowEdit {
    pub row_id: u64,
    pub gain: f32,
    pub notes: Vec<PlannedNote>,
}

/// The stretch of the composition a repeat plays over, in seconds from its
/// start: `[start_secs, end_secs)`.
///
/// The panel names both ends as *note lengths* from the beginning of the
/// composition, because that is the vocabulary the rest of it is written in —
/// a bar is a length rather than a timestamp. By the time it reaches here they
/// are seconds, like every other time in this module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopSpan {
    pub start_secs: f64,
    pub end_secs: f64,
}

impl LoopSpan {
    /// The whole composition — what a repeat looped over before it could be
    /// given a window, and what a pair of untouched boxes still means.
    pub fn whole(end_secs: f64) -> Self {
        Self { start_secs: 0.0, end_secs }
    }
}

/// A composition edited mid-flight, waiting for the next loop point.
struct LiveUpdate {
    rows: Vec<VoiceEdit>,
    loop_start: u64,
    loop_sample: u64,
    /// Set by the audio callback once it has taken this. What is left in `rows`
    /// afterwards is the *old* schedules, swapped out rather than dropped —
    /// freeing them is the GUI thread's job, not the audio thread's.
    applied: bool,
}

struct VoiceEdit {
    /// Which voice this is for, by position. The voice list is fixed for the
    /// life of a player — loading an instance is not something an audio callback
    /// can do — so an index is a stable name for one, and a voice serving
    /// several rows has no single row id to be found by.
    voice: usize,
    gain: f32,
    schedule: Vec<(u64, u8, bool)>,
    /// The same notes as file ranges, for a voice that plays one — see
    /// [`Voice::hits`]. Built for every voice because it costs a vector of
    /// triples and saves the edit having to know which kind it is addressing.
    hits: Vec<Hit>,
}

/// A row's sound plus its schedule, as the callback sees it.
struct Voice {
    /// The panel's row ids: which rows of the composition this one voice is
    /// playing. Usually one; several when they were found to be shareable (see
    /// [`share_groups`]).
    row_ids: Vec<u64>,
    gain: f32,
    /// `(sample time, pitch, note-on)`, sorted by time — what a plugin voice
    /// sends.
    schedule: Vec<(u64, u8, bool)>,
    /// The same notes as [`Hit`]s — what a sample voice plays over. Built from
    /// the same notes as the schedule by [`schedule_from`], so the two can never
    /// say different things about when a note sounds.
    hits: Vec<Hit>,
    /// Next event in `schedule` for a plugin voice; next hit in `hits` for a
    /// sample one. A voice is one or the other, so one cursor is all there is to
    /// keep.
    cursor: usize,
    sound: Sound,
}

/// What a voice makes its sound with. Two kinds, because a wav track has no
/// plugin at all: there is nothing to instantiate, nothing to send MIDI to, and
/// the file plays at the pitch it was recorded at.
enum Sound {
    Plugin(PluginSound),
    Sample(SampleSound),
}

/// A plugin instance and everything `process()` needs pointed at it. The
/// processor holds the instance, so the plugin cannot be terminated or its
/// library unloaded while the callback that calls `process()` on it still
/// exists.
struct PluginSound {
    processor: BlockProcessor,
    /// How many notes of each pitch this instance has been sent a note-on for
    /// and not yet the note-off. A repeat's wrap releases exactly these — see
    /// [`render_voice`].
    sounding: [u16; 128],
}

/// A decoded audio file, played from its start for as long as the note lasts.
///
/// Every note of a wav row plays the same thing, so nothing here is per-note:
/// where a note is in the file is `(now − its start) × step`, which makes the
/// playback position a function of the transport's own sample clock rather than
/// a cursor to keep in step with it. A rewind, a loop and an offline render then
/// all give the identical samples, with nothing to reset.
struct SampleSound {
    /// The decoded file, shared by every voice playing it — a minute of audio is
    /// ten megabytes, and each of them plays the identical samples.
    audio: Arc<DecodedAudio>,
    /// Source samples per output frame — the file's rate against the device's.
    /// The file plays at its own pitch, so this is a rate conversion and nothing
    /// more.
    step: f64,
    /// Frames of fade at each end of a note, so a file cut short by the note's
    /// length — or one that does not start and end at silence — does not click.
    fade: u64,
}

impl SampleSound {
    /// The file at `at` source samples in, linearly interpolated: the step is
    /// almost never a whole number of samples (44.1 kHz material on a 48 kHz
    /// device), and taking the nearest sample instead puts a rasp of jitter
    /// noise over everything.
    fn at(&self, at: f64) -> f32 {
        let i = at.floor();
        if i < 0.0 {
            return 0.0;
        }
        let (i, frac) = (i as usize, (at - i) as f32);
        let a = *self.audio.samples.get(i).unwrap_or(&0.0);
        let b = *self.audio.samples.get(i + 1).unwrap_or(&0.0);
        a + (b - a) * frac
    }

    /// The fade envelope `n` frames into a note that lasts `len` of them: in at
    /// the start, out at the end, and flat everywhere between. A note shorter
    /// than two fades gets a shorter fade rather than one that never reaches
    /// full level.
    fn envelope(&self, n: u64, len: u64) -> f32 {
        let fade = self.fade.min(len / 2).max(1);
        let up = (n + 1).min(fade) as f32 / fade as f32;
        let down = (len.saturating_sub(n)).min(fade) as f32 / fade as f32;
        up.min(down)
    }
}

/// A running composition. Dropping it stops playback: the stream field is
/// declared first, so it is torn down before the plugin instances (and the
/// libraries they live in) it points at.
pub struct CompositionPlayer {
    stream: Option<cpal::Stream>,
    /// Keeps every loaded instance (and its library) alive for the stream's life.
    _plugins: Vec<Arc<PluginInstance>>,
    position: Arc<AtomicU64>,
    finished: Arc<AtomicBool>,
    /// Where a repeat goes back to, in samples — the start of the window the
    /// panel's two length boxes name. Zero is the top of the composition.
    loop_start: Arc<AtomicU64>,
    /// Where a repeat wraps, in samples. Owned by the callback (it is the one
    /// that adopts a live edit's new length) and read here for the transport
    /// readout.
    loop_sample: Arc<AtomicU64>,
    /// An edit waiting for the next loop point. The callback only ever
    /// `try_lock`s it, so a busy GUI thread cannot stall the audio thread.
    live: Arc<Mutex<Option<LiveUpdate>>>,
    /// When the first block was produced. Set by the callback rather than by
    /// `start_prepared`, because `stream.play()` returning is not the same
    /// moment as the device asking for audio — and a recording is lined up
    /// against this.
    started: Arc<OnceLock<Instant>>,
    /// How far the callback runs ahead of the ear, in microseconds: the device's
    /// own output latency, as it reports it.
    latency_us: Arc<AtomicU64>,
    sample_rate: f64,
    /// Which rows each voice is playing, by position — the grouping
    /// [`share_groups`] settled on, which a live edit has to merge along.
    voice_rows: Vec<Vec<u64>>,
    /// Set when a live edit gave two rows sharing an instance different gains.
    /// One instance has one output and so one level, so the panel has to say
    /// that the transport cannot do what the sliders now ask for.
    gains_diverged: Arc<AtomicBool>,
    /// Rows that actually loaded, and rows asked for.
    pub loaded_rows: usize,
    /// Plugin instances those rows are sharing.
    pub instances: usize,
    pub total_rows: usize,
    /// End of the composition, tail included.
    pub total_secs: f64,
}

/// A composition with its plugins loaded and its schedules resolved, waiting for
/// a stream. Built by [`CompositionPlayer::prepare`], off the GUI thread.
pub struct PreparedComposition {
    voices: Vec<Voice>,
    plugins: Vec<Arc<PluginInstance>>,
    end_sample: u64,
    sample_rate: f64,
    channels: usize,
    stream_cfg: cpal::StreamConfig,
    total_rows: usize,
    /// Rows that ended up with an instance behind them — not the same as the
    /// number of instances, since one can serve several rows.
    rows_playing: usize,
}

impl PreparedComposition {
    /// Rows that loaded, and rows asked for.
    pub fn loaded_rows(&self) -> (usize, usize) {
        (self.rows_playing, self.total_rows)
    }

    /// How many plugin instances those rows are sharing.
    pub fn instances(&self) -> usize {
        self.plugins.len()
    }
}

impl CompositionPlayer {
    /// Load an instance per row and start the output stream.
    ///
    /// A row whose plugin fails to load is logged and skipped rather than
    /// aborting the transport — the rest of the composition still plays, and the
    /// caller reports the shortfall through [`Self::loaded_rows`].
    /// `span` is the stretch a repeat loops over — by default the composition's
    /// whole written length, which is not the same as where playback ends when
    /// it does *not* repeat (that is the last note-off plus [`TAIL_SECS`]).
    /// `repeat` is read every block, so the checkbox takes effect on a running
    /// transport.
    pub fn start(plans: Vec<RowPlan>, span: LoopSpan, repeat: Arc<AtomicBool>) -> Result<Self> {
        Self::start_prepared(Self::prepare(plans)?, span, repeat)
    }

    /// Load every row's plugin and resolve its schedule — the slow half of
    /// starting, and the reason it is a step of its own.
    ///
    /// A plugin instance costs real time to create (jdrummer ~80 ms, a big
    /// synth a second) and the Composer needs one *per row*, so a six-row
    /// composition is half a second before a sound. Doing that on the GUI thread
    /// freezes the window, which is what makes a Play button feel broken; the
    /// result is `Send`, so the caller can do this on a thread of its own and
    /// hand it to [`Self::start_prepared`] when it lands.
    pub fn prepare(plans: Vec<RowPlan>) -> Result<PreparedComposition> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No audio output device found")?;
        let cfg = device.default_output_config()?;
        let sample_rate = cfg.sample_rate().0 as f64;
        // Not the device's raw maximum: see `declared_block_size`. Setting a
        // plugin up for millions of frames makes it — and the scratch below —
        // reserve memory by the hundreds of megabytes, per row.
        let max_block = declared_block_size(cfg.buffer_size());
        let channels = cfg.channels() as usize;
        let stream_cfg: cpal::StreamConfig = cfg.into();

        let total_rows = plans.len();
        let PreparedVoices {
            voices,
            plugins,
            end_sample,
            rows_playing,
        } = prepare_voices(plans, sample_rate, max_block as i32)?;
        Ok(PreparedComposition {
            voices,
            plugins,
            end_sample,
            sample_rate,
            channels,
            stream_cfg,
            total_rows,
            rows_playing,
        })
    }

    /// Open the output stream for an already-prepared composition. Cheap — a
    /// couple of milliseconds — so this half belongs on the GUI thread, where
    /// `cpal::Stream` has to live anyway.
    pub fn start_prepared(
        prepared: PreparedComposition,
        span: LoopSpan,
        repeat: Arc<AtomicBool>,
    ) -> Result<Self> {
        let PreparedComposition {
            voices,
            plugins,
            end_sample,
            sample_rate,
            channels,
            stream_cfg,
            total_rows,
            rows_playing,
        } = prepared;
        // Which rows each voice plays, kept on this side too: a live edit has to
        // merge the rows of a voice back into one schedule, and the voices
        // themselves belong to the audio callback from here on.
        let voice_rows: Vec<Vec<u64>> = voices.iter().map(|v| v.row_ids.clone()).collect();
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No audio output device found")?;

        let mut voices = voices;
        // One bus for the life of the stream: its limiters have to see the
        // whole thing, not a block at a time.
        let mut bus = MixBus::new(voices.len(), channels, sample_rate);
        let bus_latency_us = (bus.latency_frames() as f64 / sample_rate.max(1.0) * 1e6) as u64;
        let position = Arc::new(AtomicU64::new(0));
        let finished = Arc::new(AtomicBool::new(false));

        // Where a repeat runs between — see `loop_window`.
        let last_event = end_sample.saturating_sub((TAIL_SECS * sample_rate).round() as u64);
        let (start, end) = loop_window(span, sample_rate, last_event);
        // Started with Repeat already ticked, the very first pass begins where
        // every later one does: the window is what was asked to be heard, and
        // an intro played in front of it only the first time round is a wait.
        if repeat.load(Ordering::Relaxed) {
            rewind(&mut voices, &position, start);
        }
        let loop_start = Arc::new(AtomicU64::new(start));
        let loop_sample = Arc::new(AtomicU64::new(end));
        let live: Arc<Mutex<Option<LiveUpdate>>> = Arc::new(Mutex::new(None));

        let cb_position = position.clone();
        let cb_finished = finished.clone();
        let cb_loop = loop_sample.clone();
        let cb_loop_start = loop_start.clone();
        let cb_live = live.clone();
        let cb_repeat = repeat;
        let started = Arc::new(OnceLock::new());
        let latency_us = Arc::new(AtomicU64::new(0));
        let cb_started = started.clone();
        let cb_latency = latency_us.clone();

        let stream = device.build_output_stream(
            &stream_cfg,
            move |out: &mut [f32], info: &cpal::OutputCallbackInfo| {
                let frames = out.len() / channels;
                if frames == 0 {
                    return;
                }
                // Where this pass sits on the wall clock, for anything lining
                // itself up against the transport. The device says when this
                // block will actually be *heard*, which is a device buffer
                // ahead of when it is asked for — 100 ms on this machine, most
                // of a 1/16 note, and a recording placed without it lands a
                // grid step late all the way through.
                if cb_started.get().is_none() {
                    let _ = cb_started.set(Instant::now());
                    let stamp = info.timestamp();
                    let latency = stamp
                        .playback
                        .duration_since(&stamp.callback)
                        .map_or(0, |d| d.as_micros() as u64);
                    // Plus the mix's own look-ahead: the limiters hand the
                    // device audio that is that much older than the clock this
                    // block was rendered against.
                    cb_latency.store(latency + bus_latency_us, Ordering::Relaxed);
                }
                // One pass per chunk of the device block. Without a repeat there
                // is exactly one; with one, the block is split at the loop point
                // so the wrap lands on the right sample.
                let mut done = 0usize;
                while done < frames {
                    let pos = cb_position.load(Ordering::Relaxed);
                    let repeat = cb_repeat.load(Ordering::Relaxed);
                    let loop_sample = cb_loop.load(Ordering::Relaxed);
                    let loop_start = cb_loop_start.load(Ordering::Relaxed);

                    // Ticking Repeat on during the release tail: the loop point
                    // is already behind us and will not come round again, so go
                    // back now rather than play on to the end.
                    //
                    // Before the window rather than past it, the pass simply
                    // plays on into it: Repeat ticked on mid-intro must not cut
                    // to the loop under a sound the user is already hearing.
                    // (Started with Repeat on, there is no intro — the stream
                    // opens at the window's start.)
                    if repeat && pos >= loop_sample {
                        rewind(&mut voices, &cb_position, loop_start);
                        continue;
                    }

                    let remaining = frames - done;
                    let n = if repeat {
                        remaining.min((loop_sample - pos) as usize)
                    } else {
                        remaining
                    };
                    // The last chunk of a pass releases every note still
                    // sounding. A note ending exactly on the loop point, or
                    // running on past it, would otherwise have its note-off
                    // skipped and sound forever.
                    let wraps = repeat && pos + n as u64 >= loop_sample;
                    mix_block(
                        &mut voices,
                        &mut out[done * channels..(done + n) * channels],
                        channels,
                        pos,
                        n,
                        wraps,
                        &mut bus,
                    );

                    let new_pos = pos + n as u64;
                    cb_position.store(new_pos, Ordering::Relaxed);
                    done += n;

                    if wraps {
                        // Straight into the next pass: nothing is reset on the
                        // plugins, so releases ring on over the loop.
                        rewind(&mut voices, &cb_position, loop_start);
                        // The loop point is also where an edit made while this
                        // was playing comes in, so a pass is never rearranged
                        // underneath itself.
                        take_live_update(
                            &cb_live,
                            &mut voices,
                            &cb_loop_start,
                            &cb_loop,
                            &cb_position,
                        );
                    } else if !repeat && new_pos >= end_sample {
                        cb_finished.store(true, Ordering::Relaxed);
                    }
                }
            },
            |e| log::error!("Composer audio error: {}", e),
            None,
        )?;
        stream.play()?;

        let instances = plugins.len();
        log::info!(
            "Composer playback started: {rows_playing}/{total_rows} row(s) on \
             {instances} instance(s), {:.1}s",
            end_sample as f64 / sample_rate
        );

        Ok(Self {
            stream: Some(stream),
            _plugins: plugins,
            position,
            finished,
            loop_start,
            loop_sample,
            live,
            started,
            latency_us,
            sample_rate,
            voice_rows,
            gains_diverged: Arc::new(AtomicBool::new(false)),
            loaded_rows: rows_playing,
            instances,
            total_rows,
            total_secs: end_sample as f64 / sample_rate,
        })
    }

    /// The length the transport is looping on, in seconds — it follows a live
    /// edit, so a tempo change or an added note moves it.
    pub fn loop_secs(&self) -> f64 {
        self.loop_sample.load(Ordering::Relaxed) as f64 / self.sample_rate.max(1.0)
    }

    /// Where it goes back to, in seconds. Zero unless the panel's window says
    /// otherwise.
    pub fn loop_start_secs(&self) -> f64 {
        self.loop_start.load(Ordering::Relaxed) as f64 / self.sample_rate.max(1.0)
    }

    /// Hand the transport a composition edited while it plays. It is taken up
    /// whole at the next loop point; until then the pass in flight is untouched.
    ///
    /// Rows are matched by [`RowEdit::row_id`]: a row the transport is not
    /// playing is ignored (it has no plugin loaded, and an audio callback cannot
    /// load one), and a row it *is* playing that no longer appears simply falls
    /// silent. Which track a row plays therefore cannot be changed this way.
    ///
    /// Rows sharing an instance are merged back along the grouping the transport
    /// started with — that cannot be revisited either, for the same reason. If an
    /// edit gives two of them different gains, the instance keeps the first row's
    /// and [`Self::gains_diverged`] says so: one output cannot be at two levels.
    pub fn update_live(&self, rows: &[RowEdit], span: LoopSpan) {
        let mut last_event = 0u64;
        let mut diverged = false;
        let rows: Vec<VoiceEdit> = self
            .voice_rows
            .iter()
            .enumerate()
            .map(|(voice, ids)| {
                let mut notes: Vec<PlannedNote> = Vec::new();
                let mut gain = None;
                for edit in ids.iter().filter_map(|id| rows.iter().find(|r| r.row_id == *id)) {
                    notes.extend_from_slice(&edit.notes);
                    match gain {
                        None => gain = Some(edit.gain),
                        Some(first) => diverged |= first != edit.gain,
                    }
                }
                let (schedule, hits, last) = schedule_from(&notes, self.sample_rate);
                last_event = last_event.max(last);
                VoiceEdit {
                    voice,
                    gain: gain.unwrap_or(0.0),
                    schedule,
                    hits,
                }
            })
            .collect();
        self.gains_diverged.store(diverged, Ordering::Relaxed);
        let (loop_start, loop_sample) = loop_window(span, self.sample_rate, last_event);
        let update = LiveUpdate {
            rows,
            loop_start,
            loop_sample,
            applied: false,
        };
        // Blocking is fine here: the audio callback only ever tries the lock, and
        // holds it for the length of a swap. Replacing what is in the slot drops
        // the previous update — old schedule buffers included — on this thread.
        if let Ok(mut slot) = self.live.lock() {
            *slot = Some(update);
        }
    }

    /// Where the composition was *heard* to be at wall-clock instant `t`, in
    /// seconds from its start — what a note played then lines up with.
    ///
    /// Not simply `t` minus the moment Play was pressed: the callback produces
    /// audio a device buffer before it is heard, so the ear is that far behind
    /// the transport and a player answering what they hear is that far "late".
    /// Subtracting the device's own reported latency puts the take where the
    /// user meant it.
    ///
    /// `None` until the device has asked for its first block (there is no
    /// clock to place anything on yet), and negative for an instant before it.
    pub fn heard_secs_at(&self, t: Instant) -> Option<f64> {
        let started = *self.started.get()?;
        // Signed: an instant from before the first block is a negative time,
        // which is how the caller tells a keystroke that beat the transport
        // from one that followed it.
        let since = match t.checked_duration_since(started) {
            Some(d) => d.as_secs_f64(),
            None => -started.saturating_duration_since(t).as_secs_f64(),
        };
        Some(since - self.latency_us.load(Ordering::Relaxed) as f64 / 1e6)
    }

    /// Whether the last live edit asked for two levels on one instance — rows
    /// that share one were given different gains. The transport plays them at
    /// the first row's; hearing the rest takes a fresh Play, which is free to
    /// group them differently.
    pub fn gains_diverged(&self) -> bool {
        self.gains_diverged.load(Ordering::Relaxed)
    }

    /// Seconds played so far.
    pub fn position_secs(&self) -> f64 {
        self.position.load(Ordering::Relaxed) as f64 / self.sample_rate.max(1.0)
    }

    /// True once the last note plus its tail has been played.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Relaxed)
    }
}

/// How far past the last event a wrap may still be pulled back to it, in
/// samples. The composition's written length and its last note-off are worked
/// out from the same units by two different roundings, so they can disagree by
/// a sample or so; a stop the user *asked* for is never this close.
const ROUNDING_SLACK: u64 = 4;

/// Where a repeat runs between, in samples — start included, end excluded.
///
/// The end is the written length, except that a wrap landing a hair before the
/// last event is pulled back onto it: rounding must not cut the note the
/// composition ends on. A [`LoopSpan`] that stops early on purpose is a whole
/// note value short of that, so it keeps the end it asked for.
///
/// The start is then held at least one sample in front of the end, whatever the
/// boxes say. A window of no length is not a short loop — it is an audio
/// callback with no samples to fill its block with, going round forever.
fn loop_window(span: LoopSpan, sample_rate: f64, last_event: u64) -> (u64, u64) {
    let end = loop_sample_for(span.end_secs, sample_rate, last_event);
    let start = (span.start_secs * sample_rate).round().max(0.0) as u64;
    (start.min(end - 1), end)
}

/// Where a repeat wraps, in samples: the written length, but never a hair
/// before the last event — see [`ROUNDING_SLACK`].
fn loop_sample_for(loop_secs: f64, sample_rate: f64, last_event: u64) -> u64 {
    let asked = (loop_secs * sample_rate).round().max(1.0) as u64;
    if asked < last_event && last_event - asked <= ROUNDING_SLACK {
        last_event
    } else {
        asked
    }
}

/// A row's notes, resolved to sample times two ways: the sorted
/// `(sample, pitch, note-on)` schedule a plugin voice sends, and the
/// `(start, end)` ranges a sample voice plays over. Also the sample the last
/// event falls on.
///
/// Both come out of the same loop deliberately: a note is a note whatever plays
/// it, and two resolutions of "when" would be two things to keep in step.
fn schedule_from(
    notes: &[PlannedNote],
    sample_rate: f64,
) -> (Vec<(u64, u8, bool)>, Vec<Hit>, u64) {
    let mut schedule: Vec<(u64, u8, bool)> = Vec::with_capacity(notes.len() * 2);
    let mut hits: Vec<Hit> = Vec::with_capacity(notes.len());
    let mut last_event = 0u64;
    for n in notes {
        let on = (n.at_secs * sample_rate).round().max(0.0) as u64;
        // At least one sample of note, so a 1/128 at a low tempo cannot
        // collapse into a note-off that precedes its own note-on.
        let off = on + ((n.dur_secs * sample_rate).round().max(1.0) as u64);
        schedule.push((on, n.pitch, true));
        schedule.push((off, n.pitch, false));
        hits.push(Hit { on, off, start_secs: n.start_secs.max(0.0) });
        last_event = last_event.max(off);
    }
    schedule.sort_by_key(|&(t, _, on)| (t, on));
    hits.sort_by_key(|h| (h.on, h.off));
    (schedule, hits, last_event)
}

/// Adopt an edit, if one is waiting. Called at the loop point, from the audio
/// thread, so it must not allocate or free: the new schedules are *swapped* with
/// the old, which are left in the slot for the GUI thread to drop.
fn take_live_update(
    live: &Mutex<Option<LiveUpdate>>,
    voices: &mut [Voice],
    loop_start: &AtomicU64,
    loop_sample: &AtomicU64,
    position: &AtomicU64,
) {
    let Ok(mut slot) = live.try_lock() else {
        // The GUI is mid-write; next pass, then.
        return;
    };
    let Some(update) = slot.as_mut() else { return };
    if update.applied {
        return;
    }
    for (i, voice) in voices.iter_mut().enumerate() {
        // By position: an edit is built from the same voice list this walks, so
        // the two cannot drift, and a voice serving several rows has no single
        // row id to be found by.
        match update.rows.iter_mut().find(|r| r.voice == i) {
            Some(edit) => {
                std::mem::swap(&mut voice.schedule, &mut edit.schedule);
                std::mem::swap(&mut voice.hits, &mut edit.hits);
                voice.gain = edit.gain;
            }
            // Every row this voice played was deleted. Its plugin stays loaded —
            // unloading is not something this thread can do — but it has nothing
            // left to play.
            None => {
                voice.schedule.clear();
                voice.hits.clear();
            }
        }
        // The rewind just before this seeked to the *old* window's start; the
        // edit may have moved it, and a cursor left pointing into the schedule
        // it no longer belongs to is the one thing a swap must not leave behind.
        seek(voice, update.loop_start);
    }
    loop_start.store(update.loop_start, Ordering::Relaxed);
    loop_sample.store(update.loop_sample, Ordering::Relaxed);
    position.store(update.loop_start, Ordering::Relaxed);
    update.applied = true;
}

/// Back to `from`, where the next pass begins: every row plays its schedule
/// again from the first event at or after it. The plugins are left alone — a
/// note still releasing carries over into the next pass, which is what makes a
/// loop sound like a loop.
fn rewind(voices: &mut [Voice], position: &AtomicU64, from: u64) {
    for voice in voices.iter_mut() {
        seek(voice, from);
    }
    position.store(from, Ordering::Relaxed);
}

/// Put one voice's cursor where playback is about to resume.
///
/// The cursor is what stops an event being sent twice, so a pass that starts
/// part-way in has to *skip* what is behind it rather than let the next block
/// find it due. Left at zero, every note-on and note-off in front of the
/// window would be handed to the plugin at once, on the loop's first sample:
/// not the composition played from there, but a click and whatever notes the
/// pairing happened to leave hanging.
///
/// A note straddling the start is simply not begun again — a repeat resumes the
/// written composition, it does not retrigger what was already sounding — and a
/// sample voice needs no help at all: [`render_voice`] walks its hits past the
/// block it is filling on its own, which also leaves one straddling the start
/// playing from the middle, where the transport's clock says it is.
fn seek(voice: &mut Voice, from: u64) {
    voice.cursor = match &voice.sound {
        Sound::Plugin(_) => first_event_at(&voice.schedule, from),
        Sound::Sample(_) => 0,
    };
}

/// Where a pass beginning at `from` picks a schedule up: the first event at or
/// after it. A schedule is sorted by time, so this is a search rather than a
/// walk — and an event landing exactly on the start belongs to the pass.
fn first_event_at(schedule: &[(u64, u8, bool)], from: u64) -> usize {
    schedule.partition_point(|&(at, _, _)| at < from)
}

/// Which rows can be played by one plugin instance, as groups of indices into
/// `plans`.
///
/// **Why share.** An instance is expensive in exactly the way a composition
/// multiplies: a LeSynth track carries its grid and a rendered buffer per key,
/// a sampler its whole kit, and the Composer used to load one per row. Six rows
/// of one drum track were six copies of that kit in memory — and six times the
/// loading. A chord recorded from the keyboard is three rows of one instrument,
/// which is the same instrument playing three notes.
///
/// **When it is safe.** Three things have to hold, and each of them is a way the
/// sharing would otherwise be audible:
///
/// * **The same recipe.** A different library, class, grid or saved state is a
///   different sound; nothing to share.
/// * **The same gain.** One instance has one output, so rows sharing it are
///   mixed at one level. Rows at different levels get an instance each.
/// * **No collision.** A note-off names a pitch, not a note: if two rows have
///   the same pitch sounding at once, one row's release would cut the other's
///   note short. Rows that would do that are kept apart.
///
/// A **wav row** is outside all of it: a sample voice has a single playback
/// position, and the file it decoded is shared between voices anyway, so a row
/// playing one always gets a voice to itself.
///
/// First-fit, in row order, so the first row of a group is the one whose recipe
/// and gain the group is named by — and a composition of unrelated rows comes
/// out exactly as it went in, one group each.
fn share_groups(plans: &[RowPlan]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    'plan: for (i, plan) in plans.iter().enumerate() {
        // A wav row never shares. A sample voice has one playback position, so
        // two rows on one would have to be mixed inside it — and the expense
        // sharing exists to avoid is not there anyway: what a wav track costs is
        // the decoded file, and that is shared by every voice playing it (see
        // `decode_samples`) whether the rows share a voice or not.
        if plan.source.wav.is_none() {
            for group in groups.iter_mut() {
                let host = &plans[group[0]];
                if !same_recipe(&host.source, &plan.source) || host.gain != plan.gain {
                    continue;
                }
                if group
                    .iter()
                    .any(|&j| notes_collide(&plans[j].notes, &plan.notes))
                {
                    continue;
                }
                group.push(i);
                continue 'plan;
            }
        }
        groups.push(vec![i]);
    }
    groups
}

/// Whether two rows would load the identical instance. The track's *name* is not
/// part of it: two registry entries pointing at the same library with the same
/// state are the same sound whatever they are called.
fn same_recipe(a: &PlaybackSource, b: &PlaybackSource) -> bool {
    a.plugin_path == b.plugin_path
        && a.class_id == b.class_id
        && a.plugin_id == b.plugin_id
        && a.is_lesynth == b.is_lesynth
        && a.state == b.state
        && a.vst_state == b.vst_state
        && a.wav == b.wav
}

/// Whether any note of `a` sounds at the same time as one of `b` on the same
/// pitch.
///
/// Quadratic in principle, and deliberately so: it runs once per candidate pair
/// when Play is pressed, over the notes a person typed into a row. The pitch
/// test comes first, so the arithmetic only happens for the few notes that could
/// possibly collide.
fn notes_collide(a: &[PlannedNote], b: &[PlannedNote]) -> bool {
    a.iter()
        .any(|x| b.iter().any(|y| x.collides_with(y)))
}

/// Load an instance per row, import its grid and resolve its notes to sample
/// times. Returns the voices, the instances keeping their libraries alive, and
/// the last sample of the composition (the final note-off plus [`TAIL_SECS`]).
///
/// A row whose plugin fails to load is logged and skipped: one broken track must
/// not take the rest of the composition with it. `plugins.len()` is how many
/// actually loaded.
/// One ready instance per plan, `None` where the row could not be loaded — a
/// broken row is logged and skipped rather than taking the composition with it.
///
/// **One at a time, deliberately.** Loading six rows takes six times as long as
/// one, and doing them on six threads would take about one — but a plugin's
/// instances are not independent enough for that. jdrummer livelocks on it: two
/// threads inside its construction spin forever, about one run in three. Making
/// the first instance alone (a plugin's one-time global setup) and the rest
/// together only lowered that to one in ten. A DAW that freezes for good one
/// time in ten is worse than one that takes three seconds, so this stays
/// sequential; the loading is off the GUI thread instead, where waiting for it
/// costs the user a message rather than a dead window.
fn load_instances(
    plans: &[RowPlan],
    groups: &[Vec<usize>],
    sample_rate: f64,
    max_block: i32,
) -> Vec<Option<Arc<PluginInstance>>> {
    // One module per distinct VST3 library rather than one per group: that is
    // what a VST3 module is for, and it saves a `dlopen` and a `ModuleEntry`
    // each time. Every other format loads its own library per instance.
    let mut modules: HashMap<PathBuf, Option<Arc<Vst3Module>>> = HashMap::new();
    // A wav row has no library to open — its `plugin_path` is the audio file.
    for plan in plans.iter().filter(|p| p.source.wav.is_none()) {
        let is_vst3 = matches!(
            detect_format(&plan.source.plugin_path, plan.source.plugin_id.as_deref()),
            Ok(PluginFormat::Vst3)
        );
        if !is_vst3 {
            continue;
        }
        modules
            .entry(plan.source.plugin_path.clone())
            .or_insert_with(|| match Vst3Module::open(&plan.source.plugin_path) {
                Ok(m) => Some(Arc::new(m)),
                Err(e) => {
                    log::warn!("Composer: '{}' failed to load: {e:#}", plan.source.name);
                    None
                }
            });
    }

    // One instance per **group**, built from the row that opened it — every row
    // in a group has the same recipe, so any of them describes it.
    groups
        .iter()
        .map(|group| {
            let plan = &plans[group[0]];
            if plan.source.wav.is_some() {
                return None;
            }
            let module = modules.get(&plan.source.plugin_path).cloned().flatten();
            load_one(module, plan, sample_rate, max_block)
        })
        .collect()
}

/// Decode the audio files the wav rows play, one per distinct path however many
/// rows name it: a file is tens of megabytes of `f32`, and every voice playing
/// it plays the identical samples.
///
/// A file that will not decode is logged and left out, and its rows fall silent
/// the same way a row whose plugin will not load does — one broken track must
/// not take the composition with it.
fn decode_samples(plans: &[RowPlan]) -> HashMap<PathBuf, Arc<DecodedAudio>> {
    let mut decoded: HashMap<PathBuf, Arc<DecodedAudio>> = HashMap::new();
    for plan in plans {
        let Some(path) = &plan.source.wav else { continue };
        if decoded.contains_key(path) {
            continue;
        }
        match decode_audio_file(path) {
            Ok(audio) => {
                decoded.insert(path.clone(), Arc::new(audio));
            }
            Err(e) => log::warn!(
                "Composer: '{}' could not decode {}: {e:#}",
                plan.source.name,
                path.display()
            ),
        }
    }
    decoded
}

/// One instance, ready to play: created, given the state the project saved, and
/// set up for audio.
fn load_one(
    module: Option<Arc<Vst3Module>>,
    plan: &RowPlan,
    sample_rate: f64,
    max_block: i32,
) -> Option<Arc<PluginInstance>> {
    // Only LeSynth exposes the state ABI, and only a tagged instance can be
    // addressed by it.
    let token = plan.source.is_lesynth.then(next_instance_token);
    // A VST3 is made from the module opened once above; any other format is
    // loaded whole.
    let loaded = match module {
        Some(module) => {
            Vst3Instance::from_module(module, plan.source.class_id.as_ref(), token)
                .map(PluginInstance::from_vst3)
        }
        None => PluginInstance::load(
            &plan.source.plugin_path,
            plan.source.class_id.as_ref(),
            plan.source.plugin_id.as_deref(),
            token,
        ),
    };
    let inst = match loaded {
        Ok(i) => Arc::new(i),
        Err(e) => {
            log::warn!("Composer: '{}' failed to load: {e:#}", plan.source.name);
            return None;
        }
    };

    // Before activation, as the spec has it: a plugin sizes its buffers for the
    // state it is given. This is where a third-party VST3's knobs come back —
    // what the user set in its editor, or what the project saved.
    if let Some(bytes) = &plan.source.vst_state {
        // Only if it would change something. A fresh instance is often already
        // in the saved state (the project was saved from one), and restoring
        // anyway is not free: jdrummer reloads its whole SoundFont on
        // `setState` whether the kit changed or not — 430 ms a row, against a
        // fraction of a millisecond to ask what state it is in.
        let unchanged = inst
            .save_state()
            .is_ok_and(|current| current == *bytes);
        if !unchanged {
            if let Err(e) = inst.restore_state(bytes) {
                log::warn!("Composer: '{}' state restore failed: {e:#}", plan.source.name);
            }
        }
    }
    let _ = inst.initialize_audio(sample_rate, max_block);
    if let Some(state) = &plan.source.state {
        if let Err(e) = inst.import_state(state) {
            log::warn!("Composer: '{}' grid import failed: {e:#}", plan.source.name);
        }
    }
    Some(inst)
}

/// Everything [`prepare_voices`] settles: the voices themselves, the instances
/// keeping their libraries alive, where the composition ends, and how many rows
/// actually got a voice — which is not the number of voices, since one serves
/// however many rows it can.
struct PreparedVoices {
    voices: Vec<Voice>,
    plugins: Vec<Arc<PluginInstance>>,
    end_sample: u64,
    rows_playing: usize,
}

fn prepare_voices(
    plans: Vec<RowPlan>,
    sample_rate: f64,
    max_block: i32,
) -> Result<PreparedVoices> {
    // Which rows can be played by one instance. Everything below is per *group*
    // from here on: the loading, the schedule and the mix.
    let groups = share_groups(&plans);
    // The module is opened once per library rather than once per group: that is
    // what a VST3 module is for, and `ModuleEntry` is not a thing to race.
    let instances = load_instances(&plans, &groups, sample_rate, max_block);
    // And the wav rows' files, decoded once each.
    let decoded = decode_samples(&plans);

    let mut voices: Vec<Voice> = Vec::new();
    let mut plugins: Vec<Arc<PluginInstance>> = Vec::new();
    let mut last_event = 0u64;
    let mut rows_playing = 0usize;

    for (group, inst) in groups.iter().zip(instances) {
        let host = &plans[group[0]];

        // Every note of every row in the group, on one instance. Sorted into one
        // schedule by `schedule_from`, so the callback still only walks a cursor.
        let notes: Vec<PlannedNote> = group
            .iter()
            .flat_map(|&i| plans[i].notes.iter().copied())
            .collect();
        let (schedule, hits, group_last_event) = schedule_from(&notes, sample_rate);

        // A wav row: no instance, no events — the file, played from its start
        // for as long as each note lasts.
        if let Some(path) = &host.source.wav {
            let Some(audio) = decoded.get(path) else { continue };
            last_event = last_event.max(group_last_event);
            rows_playing += group.len();
            voices.push(Voice {
                row_ids: group.iter().map(|&i| plans[i].row_id).collect(),
                gain: host.gain,
                schedule,
                hits,
                cursor: 0,
                sound: Sound::Sample(SampleSound {
                    audio: audio.clone(),
                    // The file plays at its own pitch, so the only conversion is
                    // of rate: a 44.1 kHz file on a 48 kHz device advances by
                    // less than a sample a frame.
                    step: audio.sample_rate.max(1.0) as f64 / sample_rate.max(1.0),
                    fade: (CLICK_FADE_SECS * sample_rate).round().max(1.0) as u64,
                }),
            });
            continue;
        }

        let Some(inst) = inst else { continue };
        last_event = last_event.max(group_last_event);
        rows_playing += group.len();

        // The buffers for the bus layout the plugin settled on in
        // `initialize_audio`, and whatever its format's `process()` needs
        // pointed at them.
        let processor = match BlockProcessor::new(inst.clone(), max_block.max(0) as usize) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("Composer: '{}' cannot be processed: {e:#}", host.source.name);
                continue;
            }
        };

        voices.push(Voice {
            row_ids: group.iter().map(|&i| plans[i].row_id).collect(),
            gain: host.gain,
            schedule,
            hits,
            cursor: 0,
            sound: Sound::Plugin(PluginSound { processor, sounding: [0; 128] }),
        });
        plugins.push(inst);
    }

    anyhow::ensure!(!voices.is_empty(), "no playable rows");
    let end_sample = last_event + (TAIL_SECS * sample_rate).round() as u64;
    Ok(PreparedVoices {
        voices,
        plugins,
        end_sample,
        rows_playing,
    })
}

/// Full scale, the level nothing may leave the mix above. Kept at 1.0 rather
/// than a hair below: a limiter that holds exactly this ceiling gives away no
/// headroom, and the device's own converter takes it from there.
const CEILING: f32 = 1.0;

/// The state a mix carries from one block to the next: a scratch buffer for the
/// voice being rendered, and the limiters. Each voice has its own, so a row
/// boosted past full scale is turned down by itself instead of ducking the
/// whole composition every time it peaks; the master's catches what is left
/// when several loud rows land together — the bass first, so a sub played at
/// full scale makes room for the drums instead of ducking them with everything
/// else (see [`Limiter::low_band`]).
struct MixBus {
    /// One voice's block, summed into the mix once its limiter has seen it.
    /// Reused across voices and blocks: an audio callback allocates nothing.
    scratch: Vec<f32>,
    voices: Vec<Limiter>,
    master_low: Limiter,
    master: Limiter,
}

impl MixBus {
    fn new(voices: usize, channels: usize, sample_rate: f64) -> Self {
        Self {
            scratch: Vec::new(),
            voices: (0..voices)
                .map(|_| Limiter::new(channels, sample_rate, CEILING))
                .collect(),
            master_low: Limiter::low_band(channels, sample_rate, CEILING),
            master: Limiter::new(channels, sample_rate, CEILING),
        }
    }

    /// How far behind its own clock the mix runs: a voice's limiter looks ahead,
    /// and then the master's two do it again — the bass one by its split's delay
    /// on top. Every voice's limiter is built at the one rate, so they cost the
    /// same.
    fn latency_frames(&self) -> usize {
        self.master.latency_frames() * 2 + self.master_low.latency_frames()
    }
}

/// Render one voice's block into `out` (interleaved, `frames * channels`),
/// which arrives zeroed and holds nothing but this voice on the way out — its
/// own limiter sees it before the mix does.
fn render_voice(
    voice: &mut Voice,
    out: &mut [f32],
    channels: usize,
    block_start: u64,
    frames: usize,
    flush_events: bool,
) {
    let block_end = block_start + frames as u64;
    let sound = match &mut voice.sound {
        Sound::Plugin(p) => p,
        // A file, played from its start for as long as each note lasts.
        // Nothing to process and no events to send: where a note is in the
        // file follows from the transport's own clock.
        Sound::Sample(s) => {
            // Notes already finished are behind us for good — until a rewind,
            // which puts the cursor back to the top.
            while voice
                .hits
                .get(voice.cursor)
                .is_some_and(|h| h.off <= block_start)
            {
                voice.cursor += 1;
            }
            for hit in &voice.hits[voice.cursor.min(voice.hits.len())..] {
                if hit.on >= block_end {
                    break;
                }
                let len = hit.off - hit.on;
                // Where the note starts *in the file* — the frame's start
                // slider, in the file's own samples.
                let skip = hit.start_secs * s.audio.sample_rate as f64;
                for pos in hit.on.max(block_start)..hit.off.min(block_end) {
                    let n = pos - hit.on;
                    let v =
                        s.at(skip + n as f64 * s.step) * s.envelope(n, len) * voice.gain;
                    let frame = (pos - block_start) as usize;
                    for ch in 0..channels {
                        out[frame * channels + ch] += v;
                    }
                }
            }
            return;
        }
    };

    // Events due in this block, offset to their sample in it.
    let processor = &mut sound.processor;
    while let Some(&(at, pitch, on)) = voice.schedule.get(voice.cursor) {
        if at >= block_end {
            break;
        }
        let status = if on { 0x90 } else { 0x80 };
        let velocity = if on { NOTE_VELOCITY } else { 0 };
        // Events already due (a late start, or two in the same block) land on
        // the block's first sample.
        let offset = at.saturating_sub(block_start).min(u32::MAX as u64) as u32;
        if !processor.push_event(offset, [status, pitch, velocity]) {
            // The block is full; the rest wait for the next one.
            break;
        }
        let count = &mut sound.sounding[pitch as usize & 0x7f];
        *count = if on { count.saturating_add(1) } else { count.saturating_sub(1) };
        voice.cursor += 1;
    }

    // `flush_events`: the caller is about to rewind, so a note still sounding
    // would never get its note-off. Release exactly those, on the block's last
    // sample — and *only* those. Emptying the rest of the schedule instead
    // handed the plugin every note-on and note-off after the loop's end in one
    // burst: a drum sampler fired them all, and with the song running on past
    // the window the burst overflowed the block's event list, leaving whatever
    // note-on had lost its note-off ringing out of time with the loop.
    if flush_events {
        let last = frames.saturating_sub(1).min(u32::MAX as usize) as u32;
        for pitch in 0..128u8 {
            let count = &mut sound.sounding[pitch as usize];
            while *count > 0 && processor.push_event(last, [0x80, pitch, 0]) {
                *count -= 1;
            }
        }
    }

    // Never more than this instance was set up for (see `PluginIo::max_block`).
    let frames = processor.process(frames);

    // Main output bus into the mix. Its channel count is the plugin's, not
    // the device's: a mono plugin repeats, a wider one has the extra dropped.
    if processor.main_out() > 0 {
        for ch in 0..channels {
            let src = processor.output(ch);
            for frame in 0..frames {
                out[frame * channels + ch] += src[frame] * voice.gain;
            }
        }
    }
}

/// Process one block of every voice and sum it into `out` (interleaved,
/// `frames * channels`), which is overwritten.
///
/// Every voice passes its own limiter on the way in and the sum passes the
/// master's on the way out, so a gain set past full scale is a level the mix
/// comes down to rather than a waveform with its top sheared off. `bus` carries
/// that state — and the delay it costs, [`MixBus::latency_frames`], which the
/// caller has to allow for.
///
/// Shared by the transport's callback and the offline export, so what a `.wav`
/// contains is what the transport plays — down to the summing and the
/// limiting.
fn mix_block(
    voices: &mut [Voice],
    out: &mut [f32],
    channels: usize,
    block_start: u64,
    frames: usize,
    flush_events: bool,
    bus: &mut MixBus,
) {
    out.fill(0.0);
    bus.scratch.clear();
    bus.scratch.resize(frames * channels, 0.0);

    for (voice, limiter) in voices.iter_mut().zip(bus.voices.iter_mut()) {
        bus.scratch.fill(0.0);
        render_voice(voice, &mut bus.scratch, channels, block_start, frames, flush_events);
        limiter.process(&mut bus.scratch);
        for (s, v) in out.iter_mut().zip(bus.scratch.iter()) {
            *s += v;
        }
    }

    bus.master_low.process(out);
    bus.master.process(out);
    // The limiter holds the ceiling on its own; this is the belt to its braces,
    // and costs a compare on a buffer the device is about to read anyway.
    for s in out.iter_mut() {
        *s = s.clamp(-CEILING, CEILING);
    }
}

/// Rate and channel count to export at: the default output device's, so a
/// rendered file matches what the transport plays through it. Falls back to CD
/// stereo when there is no device to ask — an export is not playback and has no
/// reason to fail for want of one.
pub fn default_export_format() -> (f64, usize) {
    cpal::default_host()
        .default_output_device()
        .and_then(|d| d.default_output_config().ok())
        .map(|cfg| (cfg.sample_rate().0 as f64, cfg.channels() as usize))
        .unwrap_or((44_100.0, 2))
}

/// The composition rendered to interleaved samples, off the audio device — what
/// "Export WAV" writes.
///
/// Faster than real time: nothing waits on a clock, and a LeSynth key with no
/// pre-rendered buffer renders synchronously on its note-on, so a note cannot
/// come out silent for being asked for sooner than the transport would.
///
/// Returns the samples and how many of `plans` actually loaded, so the caller
/// can report a composition that exported short of what was asked for.
pub fn render_offline(
    plans: Vec<RowPlan>,
    sample_rate: f64,
    channels: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    anyhow::ensure!(channels > 0, "an export needs at least one channel");
    let total_rows = plans.len();
    // The same block size the transport asks for by default. Block size changes
    // nothing about the result — events are placed by sample — so a fixed one
    // keeps the export independent of whatever device happens to be attached.
    const BLOCK: usize = 512;
    let PreparedVoices {
        mut voices,
        plugins,
        end_sample,
        rows_playing,
    } = prepare_voices(plans, sample_rate, BLOCK as i32)?;

    let mut bus = MixBus::new(voices.len(), channels, sample_rate);
    // The mix's limiters hand back audio a moment older than what they were
    // given, so the render runs on past the end by exactly that and drops the
    // same amount off the front: the file starts where the composition does,
    // and ends with the last of it rather than a silence of look-ahead.
    let latency = bus.latency_frames() as u64;
    let render_end = end_sample + latency;

    let mut block = vec![0f32; BLOCK * channels];
    let mut out: Vec<f32> = Vec::with_capacity(render_end as usize * channels);
    let mut pos = 0u64;
    while pos < render_end {
        let frames = BLOCK.min((render_end - pos) as usize);
        let buf = &mut block[..frames * channels];
        mix_block(&mut voices, buf, channels, pos, frames, false, &mut bus);
        out.extend_from_slice(buf);
        pos += frames as u64;
    }
    out.drain(..(latency as usize * channels).min(out.len()));

    // Voices hold `ComPtr`s into these libraries: drop them first, in the order
    // the transport tears its stream down in.
    drop(voices);
    drop(plugins);
    Ok((out, rows_playing, total_rows))
}

impl Drop for CompositionPlayer {
    fn drop(&mut self) {
        // Explicit, so the ordering that matters is stated rather than inferred
        // from field order: the stream stops before `_plugins` unloads the
        // libraries its callback calls into.
        self.stream = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(at_secs: f64, dur_secs: f64, pitch: u8) -> PlannedNote {
        PlannedNote {
            at_secs,
            dur_secs,
            pitch,
            start_secs: 0.0,
        }
    }

    fn source(path: &str) -> PlaybackSource {
        PlaybackSource {
            name: path.to_string(),
            plugin_path: PathBuf::from(path),
            class_id: None,
            plugin_id: None,
            is_lesynth: false,
            state: None,
            vst_state: None,
            wav: None,
        }
    }

    fn plan(row_id: u64, source: PlaybackSource, gain: f32, notes: Vec<PlannedNote>) -> RowPlan {
        RowPlan {
            row_id,
            source,
            gain,
            notes,
        }
    }

    /// The saving this exists for: a chord recorded from the keyboard is several
    /// rows of one instrument, and one instance can play all of it.
    #[test]
    fn rows_of_one_instrument_share_one_instance() {
        let plans = vec![
            plan(0, source("/x.so"), 1.0, vec![note(0.0, 1.0, 60)]),
            plan(1, source("/x.so"), 1.0, vec![note(0.0, 1.0, 64)]),
            plan(2, source("/x.so"), 1.0, vec![note(0.0, 1.0, 67)]),
        ];
        assert_eq!(share_groups(&plans), vec![vec![0, 1, 2]]);
    }

    /// The window a repeat runs between, in samples.
    ///
    /// Two things have to hold at once, and they pull opposite ways: a wrap must
    /// not land a rounding hair before the composition's last note-off and cut
    /// it, and a stop the user *asked* for — which is a whole note value early,
    /// never a hair — must be left exactly where they put it. The clamp is
    /// therefore a tolerance rather than an unbounded pull; before this it was
    /// `max(last_event)`, which would have quietly undone every early stop the
    /// panel can name.
    #[test]
    fn a_loop_window_keeps_a_deliberate_stop_and_still_covers_a_rounding_hair() {
        let rate = 48_000.0;
        let last_event = 96_000; // two seconds of composition

        // A hair short of the last event: rounding, so it is pulled back onto it.
        let (_, end) = loop_window(LoopSpan::whole(1.99999), rate, last_event);
        assert_eq!(end, last_event, "a wrap a sample early must not cut the last note");

        // Half a second short: a stop that was asked for, and is kept.
        let asked = LoopSpan { start_secs: 0.5, end_secs: 1.5 };
        assert_eq!(loop_window(asked, rate, last_event), (24_000, 72_000));

        // A window with nothing in it is not a short loop — it is a callback
        // with no samples to fill its block with, going round forever. The
        // start is held in front of the end whatever it asks for.
        let (start, end) = loop_window(LoopSpan { start_secs: 9.0, end_secs: 1.0 }, rate, 0);
        assert!(start < end, "start {start} must stay in front of end {end}");
        let (start, end) = loop_window(LoopSpan { start_secs: 0.0, end_secs: 0.0 }, rate, 0);
        assert!(start < end, "an empty composition still needs a sample to play");
    }

    /// A pass that starts part-way in has to *skip* the events behind it. Left
    /// at zero, the cursor would hand the plugin every note-on and note-off in
    /// front of the window at once, on the loop's first sample: not the
    /// composition played from there, but a click and whatever notes the
    /// pairing happened to leave hanging.
    #[test]
    fn seeking_a_voice_skips_the_events_in_front_of_the_window() {
        let (schedule, hits, _) = schedule_from(
            &[note(0.0, 0.5, 60), note(1.0, 0.5, 64), note(2.0, 0.5, 67)],
            48_000.0,
        );
        // At 1.0 s the first note's on *and* off are behind the window; the
        // second note's on falls exactly on it and belongs to the pass.
        assert_eq!(first_event_at(&schedule, 48_000), 2, "the first note's pair is behind us");
        assert_eq!(schedule[2].0, 48_000, "…and the second note's on is due");
        assert_eq!(first_event_at(&schedule, 0), 0, "a window at the top skips nothing");
        assert_eq!(
            first_event_at(&schedule, u64::MAX),
            schedule.len(),
            "a window past everything skips everything"
        );

        // A sample voice needs no seek at all: `render_voice` walks its hits
        // past the block it is filling on its own, which also leaves one
        // straddling the start playing from where the transport's clock says it
        // is rather than beginning it again.
        let mut voice = Voice {
            row_ids: vec![0],
            gain: 1.0,
            schedule,
            hits,
            cursor: 7,
            sound: Sound::Sample(SampleSound {
                audio: Arc::new(DecodedAudio { samples: vec![0.0; 16], sample_rate: 48_000.0 }),
                step: 1.0,
                fade: 1,
            }),
        };
        seek(&mut voice, 96_000);
        assert_eq!(voice.cursor, 0);
    }

    /// A different sound is a different instance, however alike the rows look.
    #[test]
    fn a_different_recipe_is_never_shared() {
        let mut other = source("/x.so");
        other.class_id = Some([9; 16]);
        let mut stateful = source("/x.so");
        stateful.vst_state = Some(vec![1, 2, 3]);
        let plans = vec![
            plan(0, source("/x.so"), 1.0, vec![note(0.0, 1.0, 60)]),
            plan(1, source("/y.so"), 1.0, vec![note(0.0, 1.0, 62)]),
            plan(2, other, 1.0, vec![note(0.0, 1.0, 64)]),
            plan(3, stateful, 1.0, vec![note(0.0, 1.0, 65)]),
            // Same library, same state, a different name: the same sound.
            plan(4, source("/x.so"), 1.0, vec![note(0.0, 1.0, 67)]),
        ];
        assert_eq!(
            share_groups(&plans),
            vec![vec![0, 4], vec![1], vec![2], vec![3]]
        );
    }

    /// One instance has one output and so one level. Rows at different gains
    /// have to be kept apart or one of them plays at the other's volume.
    #[test]
    fn rows_at_different_levels_are_never_shared() {
        let plans = vec![
            plan(0, source("/x.so"), 1.0, vec![note(0.0, 1.0, 60)]),
            plan(1, source("/x.so"), 0.5, vec![note(0.0, 1.0, 64)]),
            plan(2, source("/x.so"), 1.0, vec![note(0.0, 1.0, 67)]),
        ];
        assert_eq!(share_groups(&plans), vec![vec![0, 2], vec![1]]);
    }

    /// A note-off names a pitch, not a note: two rows sounding the same pitch at
    /// once on one instance would have the first release cut the second note.
    #[test]
    fn rows_whose_notes_would_collide_are_never_shared() {
        let plans = vec![
            plan(0, source("/x.so"), 1.0, vec![note(0.0, 1.0, 60)]),
            // Same pitch, overlapping — must not share.
            plan(1, source("/x.so"), 1.0, vec![note(0.5, 1.0, 60)]),
            // Same pitch, but after the first has finished — safe.
            plan(2, source("/x.so"), 1.0, vec![note(2.0, 1.0, 60)]),
        ];
        assert_eq!(share_groups(&plans), vec![vec![0, 2], vec![1]]);

        // Touching end to end is not a collision: the note-off is sorted ahead
        // of the note-on at the same sample.
        let touching = vec![
            plan(0, source("/x.so"), 1.0, vec![note(0.0, 1.0, 60)]),
            plan(1, source("/x.so"), 1.0, vec![note(1.0, 1.0, 60)]),
        ];
        assert_eq!(share_groups(&touching), vec![vec![0, 1]]);
    }

    /// Sharing must not change what is played: the merged schedule is every
    /// note of every row in the group, in time order, note-offs ahead of the
    /// note-ons they meet.
    #[test]
    fn a_shared_schedule_holds_every_row_s_notes() {
        let rate = 1000.0;
        let notes = vec![note(0.0, 1.0, 60), note(0.0, 1.0, 64), note(1.0, 1.0, 60)];
        let (schedule, hits, last) = schedule_from(&notes, rate);
        assert_eq!(
            schedule,
            vec![
                (0, 60, true),
                (0, 64, true),
                (1000, 60, false),
                (1000, 64, false),
                (1000, 60, true),
                (2000, 60, false),
            ]
        );
        assert_eq!(last, 2000);
        // And the same notes as the ranges a sample voice plays over, in time
        // order: what the wav row of a composition is played from.
        assert_eq!(
            hits,
            vec![
                Hit { on: 0, off: 1000, start_secs: 0.0 },
                Hit { on: 0, off: 1000, start_secs: 0.0 },
                Hit { on: 1000, off: 2000, start_secs: 0.0 },
            ]
        );
    }

    fn wav_source(path: &str) -> PlaybackSource {
        let mut src = source(path);
        src.wav = Some(PathBuf::from(path));
        src
    }

    /// A wav row is a voice of its own, always: a sample voice has one playback
    /// position, so two rows on one would step on each other — and the decoded
    /// file, the only thing about it worth sharing, is shared regardless.
    #[test]
    fn a_wav_row_never_shares_a_voice() {
        let plans = vec![
            plan(0, wav_source("/loop.wav"), 1.0, vec![note(0.0, 1.0, 60)]),
            // Same file, same gain, no collision — a plugin row would join the
            // group above on all three counts.
            plan(1, wav_source("/loop.wav"), 1.0, vec![note(2.0, 1.0, 64)]),
            plan(2, source("/x.so"), 1.0, vec![note(0.0, 1.0, 60)]),
            plan(3, source("/x.so"), 1.0, vec![note(0.0, 1.0, 64)]),
        ];
        assert_eq!(share_groups(&plans), vec![vec![0], vec![1], vec![2, 3]]);
    }

    /// A plugin row and a wav row are never the same recipe, however alike the
    /// rest of them looks — `plugin_path` on a wav track is the audio file, and
    /// a plugin one is a library that would be loaded.
    #[test]
    fn a_wav_track_is_not_the_plugin_of_the_same_name() {
        assert!(!same_recipe(&wav_source("/loop.wav"), &source("/loop.wav")));
        assert!(same_recipe(&wav_source("/loop.wav"), &wav_source("/loop.wav")));
    }

    /// The file is played from its start, at the rate it was recorded at, and
    /// faded at both ends so a note that cuts it does not click. Rendered from
    /// the transport's clock alone, so any block size gives the same samples.
    #[test]
    fn a_sample_voice_plays_the_file_from_the_start_of_every_note() {
        // A ramp, so every output sample says which input sample it came from.
        let audio = DecodedAudio {
            samples: (0..100).map(|i| i as f32 / 100.0).collect(),
            sample_rate: 1000.0,
        };
        let sound = SampleSound {
            audio: Arc::new(audio),
            // Device rate == file rate: one source sample per frame.
            step: 1.0,
            fade: 2,
        };
        // Two notes, the second shorter than the file.
        let mut voice = Voice {
            row_ids: vec![0],
            gain: 1.0,
            schedule: Vec::new(),
            hits: vec![
                Hit { on: 10, off: 110, start_secs: 0.0 },
                Hit { on: 200, off: 240, start_secs: 0.0 },
            ],
            cursor: 0,
            sound: Sound::Sample(sound),
        };

        // Rendered in blocks of 7, which divides neither note: a voice that kept
        // a position of its own rather than reading the clock would drift here.
        // Everything comes out `d` frames late — the mix's look-ahead — and,
        // since a ramp to 0.99 never reaches the ceiling, otherwise untouched.
        let mut bus = MixBus::new(1, 1, 1000.0);
        let d = bus.latency_frames();
        let mut out = vec![0f32; 300];
        for start in (0..300).step_by(7) {
            let frames = 7.min(300 - start);
            let mut block = vec![0f32; frames];
            mix_block(
                std::slice::from_mut(&mut voice),
                &mut block,
                1,
                start as u64,
                frames,
                false,
                &mut bus,
            );
            out[start..start + frames].copy_from_slice(&block);
        }

        // Silence before the first note and between the two.
        assert!(out[..10 + d].iter().all(|&s| s == 0.0), "{:?}", &out[..10 + d]);
        assert!(out[110 + d..200 + d].iter().all(|&s| s == 0.0));
        assert!(out[240 + d..].iter().all(|&s| s == 0.0));
        // The body of the first note is the file, sample for sample.
        for n in 3..97 {
            assert!(
                (out[10 + d + n] - n as f32 / 100.0).abs() < 1e-6,
                "frame {n} of the note is {}, not the file's {}",
                out[10 + d + n],
                n as f32 / 100.0
            );
        }
        // The second note starts the file again rather than carrying on.
        assert!((out[200 + d + 10] - 0.10).abs() < 1e-6, "{}", out[200 + d + 10]);
        // Faded in at the start and out at the end, both times: the fade is what
        // keeps a cut file from clicking.
        assert!(out[200 + d] < 0.5 * 0.001 + 0.001);
        assert!(out[239 + d].abs() < out[220 + d].abs());
    }
}
