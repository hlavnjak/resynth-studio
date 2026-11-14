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
//! A `.lsft` saved from the Instruments panel is a whole instrument.
//!
//! The file used to be the harmonic grid alone, so two things set in the
//! editor never reached it: the Modulation tab's envelope (plugin parameters)
//! and the keyboard's "Zero phases on keys" and "Repeat" checkboxes (which were
//! not saved anywhere at all). Version 5 carries the plugin's own state beside the grid,
//! and the plugin keeps the checkbox in that state. This is the whole path: out
//! of one instance, through a file, into another.

use std::path::PathBuf;

use resynth_studio::track_format::TrackState;
use resynth_studio::vst::{class_ids, next_instance_token, Vst3Instance};
use serde_json::Value;

fn load_tagged() -> Vst3Instance {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("internal_plugins")
        .join("liblesynth_fourier.so");
    Vst3Instance::load(
        &path,
        Some(&class_ids::FOURIER_SYNTH),
        Some(next_instance_token()),
    )
    .expect("load plugin")
}

fn state_of(instance: &Vst3Instance) -> Value {
    serde_json::from_slice(&instance.component_state().expect("get state")).expect("state is JSON")
}

#[test]
fn modulation_and_zero_phases_survive_a_saved_lsft() {
    // The instance the user edited: envelope on with its sliders moved, and
    // the keyboard's phases zeroed.
    let source = load_tagged();
    let mut state = state_of(&source);
    assert_eq!(
        state["params"]["env_enabled"]["bool"], false,
        "a fresh instance must not already be in the state under test"
    );
    assert_eq!(
        state["fields"]["zero_key_phases"], "false",
        "the checkbox must be part of the plugin's state, and off"
    );
    state["params"]["env_enabled"] = serde_json::json!({ "bool": true });
    state["params"]["env_attack"] = serde_json::json!({ "f32": 125.0f32 });
    state["params"]["env_sustain"] = serde_json::json!({ "f32": 0.25f32 });
    state["params"]["env_release"] = serde_json::json!({ "f32": 800.0f32 });
    state["params"]["env_release_curve"] = serde_json::json!({ "f32": -0.5f32 });
    state["fields"]["zero_key_phases"] = Value::String("true".to_string());
    assert_eq!(state["fields"]["repeat_playback"], "true", "Repeat is on by default");
    state["fields"]["repeat_playback"] = Value::String("false".to_string());
    source
        .set_component_state(&serde_json::to_vec(&state).unwrap())
        .expect("edit the source instance");
    source.initialize_audio(44_100.0, 512).expect("initialise");

    // What "Export…" and "Save" write: the live grid and the live state.
    let grid = source.export_state().expect("export grid");
    let plugin_state = source.component_state().expect("get state");
    let path = std::env::temp_dir().join(format!("lsft_plugin_state_{}.lsft", std::process::id()));
    grid.write_with_plugin_state(&path, &plugin_state).expect("write");

    // What "Load LeSynth Fourier Instrument" does with it, in the order
    // `PluginInstrument::open_editor` does: state, activation, then the grid.
    let (loaded_grid, loaded_state) = TrackState::read_with_plugin_state(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded_grid, grid, "the grid is untouched by riding with a state");
    let loaded_state = loaded_state.expect("a version 5 file carries the plugin state");

    let target = load_tagged();
    target.set_component_state(&loaded_state).expect("restore state");
    target.initialize_audio(44_100.0, 512).expect("initialise");
    target.import_state(&loaded_grid).expect("import grid");

    let back = state_of(&target);
    assert_eq!(back["params"]["env_enabled"]["bool"], true, "Modulation must come back on");
    assert_eq!(back["params"]["env_attack"]["f32"], 125.0, "…with its attack");
    assert_eq!(back["params"]["env_sustain"]["f32"], 0.25, "…its sustain");
    assert_eq!(back["params"]["env_release"]["f32"], 800.0, "…its release");
    assert_eq!(back["params"]["env_release_curve"]["f32"], -0.5, "…and its curves");
    assert_eq!(
        back["fields"]["zero_key_phases"], "true",
        "Zero phases on keys must come back ticked"
    );
    assert_eq!(
        back["fields"]["repeat_playback"], "false",
        "Repeat must come back unticked"
    );
}
