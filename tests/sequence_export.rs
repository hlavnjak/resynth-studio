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
//! "Export Sequence as WAV", end to end: the offline render must
//! produce the sequence's own length of audio, with the notes actually
//! sounding in it. Rendering faster than real time is where an export can go
//! wrong — a note asked for before its buffer is ready would export silence
//! that plays fine in the transport.

use std::path::PathBuf;

use resynth_studio::audio::write_wav_i16;
use resynth_studio::track_format::TrackState;
use resynth_studio::gui::sequencer::player::{render_offline, PlannedNote, RowPlan};
use resynth_studio::gui::registry::PlaybackSource;
use resynth_studio::vst::class_ids;

/// The release the renderer plays past the last note-off, from `player`.
const TAIL_SECS: f64 = 1.5;
const RATE: f64 = 44_100.0;
const CHANNELS: usize = 2;

/// A grid a key can actually be rendered from: a decaying harmonic series over
/// four buckets. An instrument row carries one of these; an instance without one has
/// an empty grid and every key renders silence, so a test that skipped it would
/// be measuring nothing.
fn grid() -> TrackState {
    let (nh, nb) = (256usize, 4usize);
    let mut amplitude = vec![0.0f32; nh * nb];
    for h in 0..8 {
        for b in 0..nb {
            amplitude[h * nb + b] = 0.4 / (h + 1) as f32;
        }
    }
    TrackState {
        num_harmonics: nh,
        num_buckets: nb,
        base_freq: 220.0,
        duration_secs: 0.5,
        sample_rate: 44_100.0,
        display_gain: 1.0,
        amplitude,
        phase: vec![0.0; nh * nb],
        pitch_ratio: vec![1.0; nb],
        bucket_lengths: vec![200; nb],
        dc: vec![0.0; nb],
        nyquist: vec![0.0; nb],
        amp_enabled: Vec::new(),
        phase_enabled: Vec::new(),
    }
}

fn lesynth_source() -> PlaybackSource {
    PlaybackSource {
        name: "LeSynth".to_string(),
        plugin_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("internal_plugins")
            .join("liblesynth_fourier.so"),
        class_id: Some(class_ids::FOURIER_SYNTH),
        plugin_id: None,
        is_lesynth: true,
        state: Some(grid()),
        vst_state: None,
        wav: None,
    }
}

#[test]
fn a_sequence_exports_the_audio_it_plays() {
    let notes = vec![
        PlannedNote { at_secs: 0.0, dur_secs: 0.4, pitch: 60, start_secs: 0.0 },
        PlannedNote { at_secs: 0.5, dur_secs: 0.4, pitch: 64, start_secs: 0.0 },
    ];
    let last_off = 0.9;
    let plans = vec![RowPlan {
        row_id: 0,
        source: lesynth_source(),
        gain: 1.0,
        notes,
    }];

    let (samples, loaded, total) = render_offline(plans, RATE, CHANNELS).expect("render");
    assert_eq!((loaded, total), (1, 1), "the row must have loaded");

    // The render covers the last note-off plus the tail, to the sample.
    let expected = ((last_off + TAIL_SECS) * RATE).round() as usize * CHANNELS;
    assert_eq!(samples.len(), expected, "exported length");

    // The notes are in it: something sounds while the first one is held.
    let frame = |t: f64| (t * RATE) as usize * CHANNELS;
    let peak = |from: f64, to: f64| {
        samples[frame(from)..frame(to).min(samples.len())]
            .iter()
            .fold(0.0f32, |m, &v| m.max(v.abs()))
    };
    assert!(
        peak(0.05, 0.35) > 1e-3,
        "the first note exported silent (peak {})",
        peak(0.05, 0.35)
    );
    assert!(
        peak(0.55, 0.85) > 1e-3,
        "the second note exported silent (peak {})",
        peak(0.55, 0.85)
    );
    assert!(
        samples.iter().all(|s| s.abs() <= 1.0),
        "the mix must be clamped, not wrapped"
    );

    // …and the file that comes out of it is a well-formed stereo WAV.
    let path = std::env::temp_dir().join(format!("gmst_export_{}.wav", std::process::id()));
    write_wav_i16(&path, &samples, CHANNELS as u16, RATE as u32).expect("write wav");
    let bytes = std::fs::read(&path).expect("read back");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(bytes.len(), 44 + samples.len() * 2);
}

