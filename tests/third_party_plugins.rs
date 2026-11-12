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
//! Hosting **other people's** plugins, in every format, against whatever is
//! installed on this machine. Nothing here is about our own plugin: it is about
//! the parts of a host that only a third-party plugin exercises — bundles,
//! entry points, a factory or a bundle with a plugin to choose from, and above
//! all a bus layout that is not "no inputs, one stereo output".
//!
//! That last one is why the render half matters: a plugin with an audio input
//! reads its input buffers whether or not the host meant to send anything, so
//! a host that hard-codes the layout does not fail politely — it reads
//! pointers nobody provided.
//!
//! The scan honours `VST3_PATH`, `CLAP_PATH`, `VST_PATH` and `LV2_PATH`, so a
//! directory of test plugins can be added without installing them:
//!
//! ```sh
//! CLAP_PATH=~/plugins LV2_PATH=~/plugins cargo test --test third_party_plugins -- --nocapture
//! ```
//!
//! With nothing installed there is nothing to test and the test says so and
//! passes; run it with `--nocapture` to see what it found.

use resynth_studio::gui::sequencer::player::{render_offline, PlannedNote, RowPlan};
use resynth_studio::gui::registry::PlaybackSource;
use resynth_studio::plugin::{scan_installed, validate, PluginFormat, PluginInstance};

const RATE: f64 = 44_100.0;
const CHANNELS: usize = 2;
/// The release `player` renders past the last note-off.
const TAIL_SECS: f64 = 1.5;
/// Enough per format to prove the path without walking a large collection.
const MAX_PER_FORMAT: usize = 6;

#[test]
fn installed_plugins_load_and_render_in_every_format() {
    let (found, searched) = scan_installed();
    if found.is_empty() {
        println!("no plugins installed in {searched:?} — nothing to test");
        return;
    }

    for format in PluginFormat::ALL {
        let plugins: Vec<_> = found
            .iter()
            .filter(|p| p.format == format)
            // A plugin's own name for a known-unloadable one: libsitala.so is a
            // VST2 wanting a Debian-only libcurl SONAME on this machine.
            .filter(|p| validate(&p.path, p.plugin_id.as_deref()).is_ok())
            .take(MAX_PER_FORMAT)
            .collect();
        println!("{}: {} to test", format.label(), plugins.len());

        for plugin in plugins {
            let name = &plugin.name;
            let id = plugin.plugin_id.as_deref();

            // 1) A real instance, initialised the way an instrument's editor does.
            let instance = PluginInstance::load(&plugin.path, None, id, None)
                .unwrap_or_else(|e| panic!("'{name}' ({}) failed to load: {e:#}", format.label()));
            assert_eq!(instance.format(), format, "'{name}' loaded as the wrong format");
            instance
                .initialize_audio(RATE, 512)
                .unwrap_or_else(|e| panic!("'{name}' failed to initialize: {e:#}"));
            let io = instance.io();
            assert!(!io.outputs.is_empty(), "'{name}' negotiated no audio output at all");

            // 2) Its state, out and back in — what carries an instrument's knobs into
            //    the Sequencer and into a project.
            let state = instance.save_state();
            if let Ok(bytes) = &state {
                instance
                    .restore_state(bytes)
                    .unwrap_or_else(|e| panic!("'{name}' refused its own state: {e:#}"));
            }
            drop(instance);

            // 3) Two notes through the Sequencer's offline render — the path that
            //    hands `process()` its buffers, for exactly this bus layout, and
            //    with the saved state restored into a fresh instance.
            let plan = RowPlan {
                row_id: 0,
                source: PlaybackSource {
                    name: name.clone(),
                    plugin_path: plugin.path.clone(),
                    class_id: None,
                    plugin_id: plugin.plugin_id.clone(),
                    is_lesynth: false,
                    state: None,
                    vst_state: state.ok().filter(|s| !s.is_empty()),
                    wav: None,
                },
                gain: 1.0,
                notes: vec![
                    PlannedNote { at_secs: 0.0, dur_secs: 0.5, pitch: 60, start_secs: 0.0 },
                    PlannedNote { at_secs: 0.5, dur_secs: 0.5, pitch: 64, start_secs: 0.0 },
                ],
            };
            let (samples, loaded, total) = render_offline(vec![plan], RATE, CHANNELS)
                .unwrap_or_else(|e| panic!("'{name}' failed to render: {e:#}"));
            assert_eq!(loaded, total, "'{name}' did not load for the render");
            let expected = ((1.0 + TAIL_SECS) * RATE).round() as usize * CHANNELS;
            assert_eq!(samples.len(), expected, "'{name}' rendered the wrong length");

            // An instrument should make a sound; an effect with no input has
            // nothing to make one from, so the level is reported, not asserted.
            let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
            println!(
                "  {} {name}: in={:?} out={:?} peak={peak:.4}",
                format.label(),
                io.inputs,
                io.outputs
            );
        }
    }
}
