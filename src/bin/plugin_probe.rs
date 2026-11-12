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
//! `plugin_probe <path> [--id <plugin id>] [mode]` — load a plugin of any
//! format the way the DAW does and report what the host sees: the format, the
//! bus layout, whether it has an editor, whether its state survives a round
//! trip, and what it sounds like playing a chord.
//!
//! This is the tool to reach for when a third-party plugin "does not open": it
//! prints the same error the Instruments panel would show, with nothing else in the
//! way. `plugin_probe --list` prints every plugin the picker would offer.
//!
//! Modes, after the path:
//!
//!   * `--render <out.wav>` — also write the chord it rendered to a file.
//!   * `--keys <60,64,67>` — play these keys instead of the C major chord.
//!   * `--editor [seconds]` — open the plugin's editor window, with an audio
//!     stream behind it, the way the Instruments panel does, and close it again.
//!   * `--window [seconds]` / `--audio [seconds]` — only one half of that,
//!     which is how a crash in teardown is narrowed down to one of them.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use resynth_studio::audio::{write_wav_f32, AudioEngine};
use resynth_studio::gui::editor_window::open_editor_in_thread;
use resynth_studio::gui::instrument::EditorInstance;
use resynth_studio::midi::new_midi_queue;
use resynth_studio::plugin::{detect_format, scan_installed, BlockProcessor, PluginInstance};

/// The chord the render plays unless told otherwise: C major, held for a
/// second.
const CHORD: [u8; 3] = [60, 64, 67];
const HOLD_SECS: f64 = 1.0;
const TAIL_SECS: f64 = 0.5;

fn main() {
    env_logger_fallback();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--list") {
        list();
        return;
    }
    let Some(path) = args.first().map(PathBuf::from) else {
        eprintln!("usage: plugin_probe <plugin> [--id <plugin id>] [--render out.wav | --editor [s] | --window [s] | --audio [s]]");
        eprintln!("       plugin_probe --list");
        std::process::exit(2);
    };
    let mut plugin_id: Option<String> = None;
    let mut keys: Vec<u8> = CHORD.to_vec();
    let mut mode = String::new();
    let mut mode_arg: Option<String> = None;
    let mut rest = args[1..].iter();
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--id" => plugin_id = rest.next().cloned(),
            "--keys" => {
                keys = rest
                    .next()
                    .map(|k| k.split(',').filter_map(|n| n.trim().parse().ok()).collect())
                    .unwrap_or_default()
            }
            m if m.starts_with("--") => {
                mode = m.to_string();
                mode_arg = rest.next().cloned();
            }
            _ => {}
        }
    }

    match detect_format(&path, plugin_id.as_deref()) {
        Ok(f) => println!("format:   {}", f.label()),
        Err(e) => {
            println!("format:   unknown — {e:#}");
            std::process::exit(1);
        }
    }
    let plugin = match PluginInstance::load(&path, None, plugin_id.as_deref(), None) {
        Ok(p) => p,
        Err(e) => {
            println!("load:     FAILED — {e:#}");
            std::process::exit(1);
        }
    };
    println!("loaded:   '{}'", plugin.name());

    // The device's own format, exactly as the Instruments panel does it: a plugin set
    // up for a smaller block than the stream later hands it writes past the
    // buffers it sized, which looks like a crash in the host. With no device
    // (a container, a test box) a common format stands in.
    let (rate, block) = AudioEngine::query_device_config()
        .map(|c| (c.sample_rate, c.max_buffer_size as i32))
        .unwrap_or((48_000.0, 512));
    if let Err(e) = plugin.initialize_audio(rate, block) {
        println!("audio:    FAILED — {e:#}");
        std::process::exit(1);
    }
    let io = plugin.io();
    println!(
        "audio:    {rate} Hz, up to {block} frames, inputs {:?}, outputs {:?}",
        io.inputs, io.outputs
    );
    println!(
        "editor:   {}",
        if plugin.has_editor() { "yes" } else { "none this platform can show" }
    );

    match plugin.save_state() {
        Ok(state) => match plugin.restore_state(&state) {
            Ok(()) => {
                let again = plugin.save_state().map(|s| s == state).unwrap_or(false);
                println!(
                    "state:    {} bytes, restores{}",
                    state.len(),
                    if again { " and saves back identically" } else { " (saves back differently)" }
                );
            }
            Err(e) => println!("state:    {} bytes, restore FAILED — {e:#}", state.len()),
        },
        Err(e) => println!("state:    none — {e:#}"),
    }

    let plugin = Arc::new(plugin);
    match render_chord(&plugin, &keys, rate, block.max(1) as usize) {
        Ok((samples, channels)) => {
            let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
            let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt();
            println!("render:   keys {keys:?} for {HOLD_SECS}s → peak {peak:.4}, rms {rms:.4}");
            if mode == "--render" {
                let out = PathBuf::from(mode_arg.clone().unwrap_or_else(|| "probe.wav".into()));
                match write_wav_f32(&out, &samples, channels as u16, rate as u32) {
                    Ok(()) => println!("render:   written to {}", out.display()),
                    Err(e) => println!("render:   write FAILED — {e:#}"),
                }
            }
        }
        Err(e) => println!("render:   FAILED — {e:#}"),
    }

    let secs: u64 = mode_arg.and_then(|s| s.parse().ok()).unwrap_or(10);
    let hold = |what: &str, closed: &dyn Fn() -> bool| {
        println!("{what}: holding for {secs}s");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
        while std::time::Instant::now() < deadline && !closed() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        println!("{what}: tearing down");
    };
    match mode.as_str() {
        "--editor" => match EditorInstance::open(plugin, new_midi_queue()) {
            Ok(editor) => {
                println!("window:   open (audible: {})", editor.is_audible());
                hold("window", &|| editor.is_closed());
            }
            Err(e) => println!("window:   FAILED — {e:#}"),
        },
        // Editor window, no audio stream.
        "--window" => match open_editor_in_thread(&plugin) {
            Ok(handle) => {
                hold("window", &|| handle.closed.load(Ordering::Relaxed));
                resynth_studio::gui::editor_window::request_close(&handle.close_flag);
                let _ = handle.handle.join();
                drop(plugin);
            }
            Err(e) => println!("window:   FAILED — {e:#}"),
        },
        // Audio stream, no editor window.
        "--audio" => match AudioEngine::start(plugin.clone(), new_midi_queue()) {
            Ok(engine) => {
                hold("audio", &|| false);
                drop(engine);
                drop(plugin);
            }
            Err(e) => println!("audio:    FAILED — {e:#}"),
        },
        _ => drop(plugin),
    }
    // Every mode has let go of the instance by now, on the thread the app would.
    // If anything of the plugin outlives it in code the host has unmapped — a
    // thread it left running, a thread-local destructor — the process dies
    // within the next second, rather than one click later in the DAW. The host
    // pins plugin libraries for exactly this (see `plugin::open_pinned`).
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("unload:   survived 1 s after the instance was dropped");
    println!("done");
}

/// Play `keys` through the same [`BlockProcessor`] the DAW's audio paths use,
/// block by block, and return the main bus interleaved.
fn render_chord(
    plugin: &Arc<PluginInstance>,
    keys: &[u8],
    rate: f64,
    block: usize,
) -> anyhow::Result<(Vec<f32>, usize)> {
    let mut processor = BlockProcessor::new(plugin.clone(), block)?;
    let channels = processor.main_out().max(1);
    let block = processor.max_block().min(512);
    let total = ((HOLD_SECS + TAIL_SECS) * rate) as usize;
    let off_at = (HOLD_SECS * rate) as usize;
    let mut out = Vec::with_capacity(total * channels);
    let mut pos = 0usize;
    while pos < total {
        let frames = block.min(total - pos);
        if pos == 0 {
            for &key in keys {
                processor.push_event(0, [0x90, key, 100]);
            }
        }
        if (pos..pos + frames).contains(&off_at) {
            for &key in keys {
                processor.push_event((off_at - pos) as u32, [0x80, key, 0]);
            }
        }
        let done = processor.process(frames);
        for f in 0..done {
            for ch in 0..channels {
                out.push(processor.output(ch).get(f).copied().unwrap_or(0.0));
            }
        }
        pos += done.max(1);
    }
    Ok((out, channels))
}

/// Every plugin the Instruments panel's picker would offer.
fn list() {
    let (found, searched) = scan_installed();
    for dir in &searched {
        println!("searched: {}", dir.display());
    }
    for p in &found {
        println!(
            "{:<4}  {:<40}  {}{}",
            p.format.label(),
            p.name,
            p.path.display(),
            p.plugin_id.as_deref().map(|id| format!("  [{id}]")).unwrap_or_default()
        );
    }
    println!("{} plugins", found.len());
}

/// Route the crate's `log` output to stderr.
fn env_logger_fallback() {
    let _ = fern::Dispatch::new()
        .level(log::LevelFilter::Info)
        .chain(std::io::stderr())
        .apply();
}
