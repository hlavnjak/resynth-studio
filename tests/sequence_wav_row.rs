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
//! A **wav row**, end to end through the sequence renderer: an Instrument that is
//! an audio file, played from its start for the length of the note it is given.
//!
//! Rendered offline through the same voices and the same mix the transport uses,
//! so what this measures is what Play sounds like. No plugin is involved — that
//! is the point of the feature, and it is why this test runs anywhere.

use std::path::PathBuf;

use resynth_studio::audio::write_wav_f32;
use resynth_studio::gui::sequencer::player::{render_offline, PlannedNote, RowPlan};
use resynth_studio::gui::registry::PlaybackSource;

/// The release the renderer plays past the last note-off, from `player`.
const TAIL_SECS: f64 = 1.5;
/// Written and rendered at the same rate, so one output frame is one source
/// sample and the samples can be compared by index. (A rate conversion is
/// covered by the unit tests; what this file is about is the whole path.)
const RATE: f64 = 8_000.0;
/// A second of file.
const FILE_LEN: usize = 8_000;

/// A ramp, so every sample says where in the file it came from.
fn ramp() -> Vec<f32> {
    (0..FILE_LEN).map(|i| i as f32 / FILE_LEN as f32).collect()
}

fn wav_source(path: &PathBuf) -> PlaybackSource {
    PlaybackSource {
        name: "take.wav".to_string(),
        // A wav instrument has no library; its path is the file it plays.
        plugin_path: path.clone(),
        class_id: None,
        plugin_id: None,
        is_lesynth: false,
        state: None,
        vst_state: None,
        wav: Some(path.clone()),
    }
}

/// The whole of it: a row of two notes on one file renders the file twice, each
/// from its start, each cut to its own note's length, with silence where the row
/// has silence.
#[test]
fn a_wav_row_renders_the_file_under_every_note() {
    let dir = std::env::temp_dir().join("resynth-wav-row");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("take.wav");
    let source = ramp();
    write_wav_f32(&path, &source, 1, RATE as u32).expect("write the source file");

    // A half-second note at 0.25 s, and a quarter-second one at 1.5 s — the
    // second shorter than the file, so it proves the note's length is what cuts
    // it rather than the file running out.
    let plans = vec![RowPlan {
        row_id: 0,
        source: wav_source(&path),
        gain: 1.0,
        notes: vec![
            PlannedNote { at_secs: 0.25, dur_secs: 0.5, pitch: 60, start_secs: 0.0 },
            PlannedNote { at_secs: 1.5, dur_secs: 0.25, pitch: 72, start_secs: 0.0 },
        ],
    }];
    let (out, loaded, total) = render_offline(plans, RATE, 1).expect("renders");
    assert_eq!((loaded, total), (1, 1), "the row plays a file, so nothing can fail to load");

    // The written length: the last note-off plus the tail the renderer plays on
    // for. A wav row has no release, but it is mixed with rows that do.
    let expected = ((1.5 + 0.25 + TAIL_SECS) * RATE).round() as usize;
    assert_eq!(out.len(), expected, "the render is not the sequence's length");

    let at = |secs: f64| (secs * RATE).round() as usize;
    // Silence where the row is silent — before the first note, between the two,
    // and through the tail.
    for (from, to) in [(0, at(0.25)), (at(0.75), at(1.5)), (at(1.75), out.len())] {
        let loudest = out[from..to].iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(loudest == 0.0, "sound in the silence at {from}..{to}: {loudest}");
    }

    // Under each note, the file from its own start — sample for sample, once the
    // click fade at either end is past (5 ms, and a hundred samples is well
    // clear of it).
    for (start, len) in [(at(0.25), at(0.5)), (at(1.5), at(0.25))] {
        for n in 100..len - 100 {
            let want = source[n];
            let got = out[start + n];
            assert!(
                (got - want).abs() < 1e-4,
                "sample {n} of the note at {start} is {got}, not the file's {want}"
            );
        }
        // And it *is* faded at both ends: a note that cut the file mid-waveform
        // and started it from a step would click on every repeat.
        assert!(out[start] < source[0] + 0.01);
        assert!(out[start + len - 1].abs() < out[start + len / 2].abs());
    }

    let _ = std::fs::remove_file(&path);
}

/// The point of the start: a note plays from where it is pointed *into* the
/// file, not from the file's first sample — which is what puts the sound on the
/// beat rather than the room tone in front of it. Two notes on one file, started
/// at different places, must play different parts of it.
#[test]
fn a_note_plays_the_file_from_where_its_start_points() {
    let dir = std::env::temp_dir().join("resynth-wav-start");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("take.wav");
    let source = ramp();
    write_wav_f32(&path, &source, 1, RATE as u32).expect("write the source file");

    // The same note twice: one from the top of the file, one from 0.5 s in.
    let plans = vec![RowPlan {
        row_id: 0,
        source: wav_source(&path),
        gain: 1.0,
        notes: vec![
            PlannedNote { at_secs: 0.0, dur_secs: 0.25, pitch: 60, start_secs: 0.0 },
            PlannedNote { at_secs: 1.0, dur_secs: 0.25, pitch: 60, start_secs: 0.5 },
        ],
    }];
    let (out, _, _) = render_offline(plans, RATE, 1).expect("renders");

    let at = |secs: f64| (secs * RATE).round() as usize;
    let skip = at(0.5);
    for n in 100..at(0.25) - 100 {
        assert!(
            (out[n] - source[n]).abs() < 1e-4,
            "the first note left the top of the file: sample {n} is {}, not {}",
            out[n],
            source[n]
        );
        let got = out[at(1.0) + n];
        assert!(
            (got - source[skip + n]).abs() < 1e-4,
            "the second note is at sample {n} playing {got}, not the file's {} \
             half a second in",
            source[skip + n]
        );
    }

    // A start past the end of the file has nothing to play. Not an error — the
    // note simply falls silent, which is what a length running off the end does
    // too.
    let plans = vec![RowPlan {
        row_id: 0,
        source: wav_source(&path),
        gain: 1.0,
        notes: vec![PlannedNote { at_secs: 0.0, dur_secs: 0.25, pitch: 60, start_secs: 9.0 }],
    }];
    let (out, _, _) = render_offline(plans, RATE, 1).expect("renders");
    assert!(out.iter().all(|s| *s == 0.0), "a start past the end of the file must be silent");

    let _ = std::fs::remove_file(&path);
}

/// A file that is not there must not take the sequence with it: the row is
/// dropped, the rest plays, and the caller is told how many rows made it.
#[test]
fn a_missing_file_leaves_the_row_out_rather_than_failing_the_render() {
    let present = std::env::temp_dir().join("resynth-wav-row").join("present.wav");
    std::fs::create_dir_all(present.parent().unwrap()).expect("temp dir");
    write_wav_f32(&present, &ramp(), 1, RATE as u32).expect("write the source file");

    let note = |at_secs: f64| PlannedNote { at_secs, dur_secs: 0.5, pitch: 60, start_secs: 0.0 };
    let plans = vec![
        RowPlan {
            row_id: 0,
            source: wav_source(&PathBuf::from("/nonexistent/gone.wav")),
            gain: 1.0,
            notes: vec![note(0.0)],
        },
        RowPlan {
            row_id: 1,
            source: wav_source(&present),
            gain: 1.0,
            notes: vec![note(0.0)],
        },
    ];
    let (out, loaded, total) = render_offline(plans, RATE, 1).expect("renders");
    assert_eq!((loaded, total), (1, 2), "the missing row should be the only one dropped");
    assert!(out.iter().any(|s| *s != 0.0), "the row that is there should still sound");

    let _ = std::fs::remove_file(&present);
}
