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
//! What the `.lsft` grid does **not** carry: the controls that drew it.
//!
//! A LeSynth instrument's curves live in the grid, but every control in the Synth
//! editor — each harmonic's curve type, offset and granularity, and the 32
//! nested-Fourier amplitude/phase sliders and base frequency under them — lives
//! in the plugin's own state instead. A project that saved only the grid
//! reloaded a correct picture over sliders that all read zero, and the first
//! touch of any of them redrew the curve from those zeros.
//!
//! So the project folder now keeps a `.vststate` for a LeSynth row too, exactly
//! as it does for a third-party VST3. This is that path: the plugin's state out
//! of one instance and into another.

use std::path::PathBuf;

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

/// The nested-Fourier series is stored as a JSON *string* inside the state's
/// `fields`, one per harmonic. This is the 7th harmonic's, edited the way the
/// editor's sliders would.
fn edited_series() -> String {
    let mut amps = vec![0.0f32; 32];
    let mut phases = vec![0.0f32; 32];
    amps[0] = 0.4;
    amps[3] = 0.12;
    phases[3] = -1.25;
    let grans: Vec<u8> = (0..32).map(|i| if i < 8 { 5 } else { 3 }).collect();
    let series = serde_json::json!({
        "amps": amps,
        "phases": phases,
        "grans": grans,
        "base_freq_hz": 12.0,
    });
    serde_json::to_string(&serde_json::json!({
        "amp_chart": series,
        "phase_chart": series,
    }))
    .unwrap()
}

#[test]
fn the_synth_editors_controls_survive_the_state_a_project_saves() {
    let source = load_tagged();
    let mut state: Value =
        serde_json::from_slice(&source.component_state().expect("get state")).expect("state is JSON");

    // A fresh instance must not already be in the state under test, or this
    // proves nothing.
    let before = state["fields"]["nested_fourier_7"].as_str().unwrap().to_string();
    assert!(
        before.contains("\"base_freq_hz\":0.0"),
        "a fresh instance starts on auto: {before}"
    );

    // Edit it the way the editor would: the nested-Fourier sliders and base
    // frequency (persisted state) and a curve offset (a plain parameter).
    state["fields"]["nested_fourier_7"] = Value::String(edited_series());
    state["params"]["curve_offset_amp_8"] = serde_json::json!({ "f32": 0.33f32 });
    let edited = serde_json::to_vec(&state).expect("re-encode");

    // …and into the instance a loaded project would build.
    let target = load_tagged();
    target.set_component_state(&edited).expect("restore state");
    let back: Value =
        serde_json::from_slice(&target.component_state().expect("get state")).expect("state is JSON");

    let series: Value =
        serde_json::from_str(back["fields"]["nested_fourier_7"].as_str().expect("field is a string"))
            .expect("field is JSON");
    assert_eq!(
        series["amp_chart"]["base_freq_hz"], 12.0,
        "the nested-Fourier base frequency must come back"
    );
    assert_eq!(series["amp_chart"]["amps"][0], 0.4, "…and the amplitude sliders");
    assert_eq!(series["amp_chart"]["phases"][3], -1.25, "…and the phase sliders");
    assert_eq!(
        series["phase_chart"]["grans"][0], 5,
        "…and each slider's granularity, which is the range it can even show"
    );
    assert_eq!(
        back["params"]["curve_offset_amp_8"]["f32"], 0.33,
        "the plain parameters must come back too"
    );
}
