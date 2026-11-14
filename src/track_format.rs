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
//! `.lsft` — the custom LeSynth Fourier Track format.
//!
//! A saved track is the full harmonic grid from a LeSynth editor:
//! `amplitude[h][b]`, `phase[h][b]`, the per-bucket pitch (`pitch_ratio[b]` =
//! `f_local / base_freq`) and the reference `base_freq`. The host reads/writes
//! the file; the grid crosses the C ABI (`PluginInstance::export_state` /
//! `import_state`).
//!
//! Layout (little-endian):
//! ```text
//! "LSFT" (4) | version u32 | num_harmonics u32 | num_buckets u32 |
//! base_freq f32 | duration_secs f32 | sample_rate f32 |
//! display_gain f32 (version >= 2 only) |
//! amplitude[nh*nb] f32 (row-major, h*nb + b) | phase[nh*nb] f32 |
//! pitch_ratio[nb] f32 |
//! bucket_lengths[nb] u32 | dc[nb] f32 | nyquist[nb] f32   (version >= 3 only) |
//! amp_enabled[nh] u8 | phase_enabled[nh] u8                (version >= 4 only) |
//! plugin_state_len u32 | plugin_state[len] u8              (version >= 5 only)
//! ```
//!
//! **Version 5 added the plugin's own state**, opaque to the host: the
//! Modulation envelope, "Zero phases on keys" and the Synth editor's curve
//! controls are plugin parameters, which nothing but that state can restore.
//! [`TrackState`] stays the grid; the state travels beside it
//! ([`TrackState::write_with_plugin_state`] / `read_with_plugin_state`).
//!
//! Version 2 added `display_gain`, without which a reloaded track cannot be
//! auditioned at its source's level. **Version 3 added the three fields the
//! exact inverse needs**, none of them derivable from the grid: the per-bucket
//! length (the last bucket absorbs the remainder, so pitch and length differ),
//! and DC and Nyquist, which are not harmonics and have no row — dropping DC
//! alone costs ~120 dB. Older files load with `display_gain` unknown (`0.0`) and
//! empty [`TrackState::bucket_lengths`], which switches the exact path off
//! rather than feeding it numbers it cannot trust.
//!
//! **Version 4 added the per-harmonic enable flags** — the editor's `amp` and
//! `phase` checkboxes. They are not part of the grid (a disabled harmonic keeps
//! its analysed row and is skipped when rendering), so a file without them
//! reloads with every harmonic switched back on, quietly discarding the
//! selection the track was saved with. Older files load with both vectors empty,
//! which means "all enabled" — the state they were saved in.

use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

const MAGIC: [u8; 4] = *b"LSFT";
const VERSION: u32 = 5;
/// magic..sample_rate — the fields every version has.
const HEADER_V1_LEN: usize = 4 + 4 + 4 + 4 + 4 + 4 + 4;
/// …plus `display_gain`, from version 2 on. The header is unchanged in v3 and
/// v4, which only append payload.
const HEADER_V2_LEN: usize = HEADER_V1_LEN + 4;

/// The full state of one LeSynth Fourier track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackState {
    pub num_harmonics: usize,
    pub num_buckets: usize,
    /// Reference fundamental (Hz); per-bucket freq = `base_freq * pitch_ratio[b]`.
    pub base_freq: f32,
    /// Source wall-clock duration (s) a note renders for in Analysis mode.
    pub duration_secs: f32,
    /// Sample rate the audio was **analysed** at — the source file's, not the
    /// playback device's. No longer informational: [`Self::bucket_lengths`] are
    /// in these samples, so the audition scales by `device_rate / this` to render
    /// the note at the pitch and duration it was recorded with. A wrong value
    /// here transposes the whole track.
    pub sample_rate: f32,
    /// Gain the plugin's display normalisation applied when this grid was
    /// analysed (`grid_amplitude = source_amplitude × display_gain`). Saved so
    /// the Original Pitch And Gain audition can divide it back out and reproduce
    /// the analysed audio at its own level. `0.0` = unknown — a version 1 file, or a
    /// grid that never came from an analysis.
    pub display_gain: f32,
    /// `amplitude`/`phase` are row-major `[h * num_buckets + b]`, `nh*nb` long.
    pub amplitude: Vec<f32>,
    pub phase: Vec<f32>,
    /// Per-bucket pitch ratio (`f_local / base_freq`), `num_buckets` long.
    pub pitch_ratio: Vec<f32>,
    /// Per-bucket inverse-FFT length in whole samples (`num_buckets` long), or
    /// **empty** when this grid cannot be inverted exactly — a v1/v2 file, or a
    /// hand-drawn Synth grid that never came from an analysis. Not the same
    /// quantity as [`Self::pitch_ratio`]: the last bucket absorbs the subtrack's
    /// remainder, so it is shorter than a period without the pitch changing.
    pub bucket_lengths: Vec<u32>,
    /// Per-bucket DC (bin 0) and Nyquist (bin `N/2`) terms — the parts of the
    /// transform that are not harmonics and so have no row in the grid. Empty
    /// exactly when [`Self::bucket_lengths`] is.
    pub dc: Vec<f32>,
    pub nyquist: Vec<f32>,
    /// The editor's per-harmonic `amp` / `phase` checkboxes (`num_harmonics`
    /// long), or **empty** for "every harmonic enabled" — a pre-v4 file, or a
    /// plugin too old to report them. They are not in the grid: a disabled
    /// harmonic keeps its analysed row and is dropped when the note is rendered,
    /// so without these a saved track comes back with the whole spectrum on.
    pub amp_enabled: Vec<bool>,
    pub phase_enabled: Vec<bool>,
}

impl TrackState {
    /// Whether this track carries everything needed to reproduce its source
    /// exactly. False for pre-v3 files and hand-drawn grids, which must be
    /// auditioned through the transposing renderer instead.
    pub fn supports_exact_inverse(&self) -> bool {
        self.bucket_lengths.len() == self.num_buckets
            && self.dc.len() == self.num_buckets
            && self.nyquist.len() == self.num_buckets
            && self.bucket_lengths.iter().all(|&n| n >= 2)
    }
}

impl TrackState {
    /// Serialize to the `.lsft` byte layout, with no plugin state.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes_with_plugin_state(&[])
    }

    /// Serialize to the `.lsft` byte layout, carrying `plugin_state` — the
    /// plugin's own saved state (`PluginInstance::save_state`), opaque here.
    ///
    /// That is where everything *beside* the grid lives: the Modulation tab's
    /// envelope, "Zero phases on keys", and the Synth editor's curve controls.
    /// They are plugin parameters, which only the plugin's own state can
    /// restore, so the file carries that state whole rather than a copy of some
    /// of its fields. Empty means "none": the instrument loads on the plugin's
    /// defaults for all of them, as every file before version 5 does.
    pub fn to_bytes_with_plugin_state(&self, plugin_state: &[u8]) -> Vec<u8> {
        let grid = self.num_harmonics * self.num_buckets;
        let mut out = Vec::with_capacity(
            HEADER_V2_LEN + (grid * 2 + self.num_buckets) * 4 + plugin_state.len(),
        );
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.num_harmonics as u32).to_le_bytes());
        out.extend_from_slice(&(self.num_buckets as u32).to_le_bytes());
        out.extend_from_slice(&self.base_freq.to_le_bytes());
        out.extend_from_slice(&self.duration_secs.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.display_gain.to_le_bytes());
        for &v in self.amplitude.iter().chain(&self.phase).chain(&self.pitch_ratio) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        // v3 tail. Always written (the length is fixed by `num_buckets`, so the
        // size check on read stays exact); zeroed when this grid has no exact
        // inverse, which `supports_exact_inverse` then reports as false.
        let exact = self.supports_exact_inverse();
        for b in 0..self.num_buckets {
            let n = if exact { self.bucket_lengths[b] } else { 0 };
            out.extend_from_slice(&n.to_le_bytes());
        }
        for b in 0..self.num_buckets {
            let v = if exact { self.dc[b] } else { 0.0 };
            out.extend_from_slice(&v.to_le_bytes());
        }
        for b in 0..self.num_buckets {
            let v = if exact { self.nyquist[b] } else { 0.0 };
            out.extend_from_slice(&v.to_le_bytes());
        }
        // v4 tail: the enable checkboxes, one byte each. Also always written —
        // an empty vector is a grid whose flags were never reported, and "all
        // enabled" is exactly what such a grid plays.
        for flags in [&self.amp_enabled, &self.phase_enabled] {
            for h in 0..self.num_harmonics {
                out.push(u8::from(flags.get(h).copied().unwrap_or(true)));
            }
        }
        // v5 tail: the plugin's own state, length-prefixed. Always written, so
        // a file without one still has the (zero) length.
        out.extend_from_slice(&(plugin_state.len() as u32).to_le_bytes());
        out.extend_from_slice(plugin_state);
        out
    }

    /// Parse from the `.lsft` byte layout, validating magic, version and that the
    /// declared grid size matches the actual byte length exactly (which also
    /// bounds allocation to the input length).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with_plugin_state(bytes).map(|(state, _)| state)
    }

    /// [`Self::from_bytes`], plus the plugin state the file carries — empty for
    /// a file older than version 5, or one saved without any.
    pub fn from_bytes_with_plugin_state(bytes: &[u8]) -> Result<(Self, Vec<u8>)> {
        ensure!(bytes.len() >= HEADER_V1_LEN, "file too short to be a LeSynth track");
        ensure!(bytes[0..4] == MAGIC, "not a LeSynth track (bad magic)");
        let mut cur = 4;
        let next_u32 = |cur: &mut usize| -> u32 {
            let v = u32::from_le_bytes(bytes[*cur..*cur + 4].try_into().unwrap());
            *cur += 4;
            v
        };
        let version = next_u32(&mut cur);
        ensure!(
            (1..=VERSION).contains(&version),
            "unsupported .lsft version {version}"
        );
        let num_harmonics = next_u32(&mut cur) as usize;
        let num_buckets = next_u32(&mut cur) as usize;
        let base_freq = f32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
        cur += 4;
        let duration_secs = f32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
        cur += 4;
        let sample_rate = f32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
        cur += 4;
        // Version 1 predates the source level being recorded; 0.0 = unknown.
        let display_gain = if version >= 2 {
            ensure!(bytes.len() >= HEADER_V2_LEN, "truncated .lsft header");
            let g = f32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
            cur += 4;
            g
        } else {
            0.0
        };

        let grid = num_harmonics
            .checked_mul(num_buckets)
            .context("grid size overflow")?;
        // amplitude + phase + pitch_ratio, then v3's three per-bucket tails.
        let per_bucket = if version >= 3 { 4 } else { 1 };
        let floats = grid
            .checked_mul(2)
            .and_then(|g| num_buckets.checked_mul(per_bucket).and_then(|t| g.checked_add(t)))
            .context("payload size overflow")?;
        let header = if version >= 2 { HEADER_V2_LEN } else { HEADER_V1_LEN };
        // v4 appends two bytes per harmonic — the enable checkboxes.
        let flag_bytes = if version >= 4 {
            num_harmonics.checked_mul(2).context("flag size overflow")?
        } else {
            0
        };
        let fixed = header + floats * 4 + flag_bytes;
        // v5 appends the plugin's state behind a length. The length sits at a
        // known offset, so it is read first and the size check stays exact.
        let plugin_state_len = if version >= 5 {
            ensure!(bytes.len() >= fixed + 4, "truncated .lsft: no plugin state length");
            4 + u32::from_le_bytes(bytes[fixed..fixed + 4].try_into().unwrap()) as usize
        } else {
            0
        };
        let expected = fixed + plugin_state_len;
        ensure!(
            bytes.len() == expected,
            "corrupt .lsft: expected {expected} bytes, got {}",
            bytes.len()
        );

        let read_floats = |cur: &mut usize, n: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(f32::from_le_bytes(bytes[*cur..*cur + 4].try_into().unwrap()));
                *cur += 4;
            }
            v
        };
        let amplitude = read_floats(&mut cur, grid);
        let phase = read_floats(&mut cur, grid);
        let pitch_ratio = read_floats(&mut cur, num_buckets);

        // Zeroed lengths mean the writer had no exact inverse to record, so drop
        // all three rather than hand the audition a grid it would mis-invert.
        let (bucket_lengths, dc, nyquist) = if version >= 3 {
            let mut lens = Vec::with_capacity(num_buckets);
            for _ in 0..num_buckets {
                lens.push(u32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap()));
                cur += 4;
            }
            let dc = read_floats(&mut cur, num_buckets);
            let nyq = read_floats(&mut cur, num_buckets);
            if lens.iter().all(|&n| n >= 2) {
                (lens, dc, nyq)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            }
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // A vector that is all-true carries no information the default does not,
        // so it collapses back to empty — "every harmonic enabled" has one
        // representation, whichever version the file was written in.
        let read_flags = |cur: &mut usize| -> Vec<bool> {
            if version < 4 {
                return Vec::new();
            }
            let v: Vec<bool> = bytes[*cur..*cur + num_harmonics]
                .iter()
                .map(|&b| b != 0)
                .collect();
            *cur += num_harmonics;
            if v.iter().all(|&e| e) {
                Vec::new()
            } else {
                v
            }
        };
        let amp_enabled = read_flags(&mut cur);
        let phase_enabled = read_flags(&mut cur);
        let plugin_state = if version >= 5 {
            bytes[cur + 4..].to_vec()
        } else {
            Vec::new()
        };

        let state = Self {
            num_harmonics,
            num_buckets,
            base_freq,
            duration_secs,
            sample_rate,
            display_gain,
            amplitude,
            phase,
            pitch_ratio,
            bucket_lengths,
            dc,
            nyquist,
            amp_enabled,
            phase_enabled,
        };
        Ok((state, plugin_state))
    }

    /// Write the grid alone to `path` (creating/truncating it).
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_with_plugin_state(path, &[])
    }

    /// Write the grid and the plugin's own state to `path` — see
    /// [`Self::to_bytes_with_plugin_state`].
    pub fn write_with_plugin_state(&self, path: &Path, plugin_state: &[u8]) -> Result<()> {
        fs::write(path, self.to_bytes_with_plugin_state(plugin_state))
            .with_context(|| format!("writing track to {}", crate::file_label(path)))
    }

    /// Read a track's grid from `path`.
    pub fn read(path: &Path) -> Result<Self> {
        Self::read_with_plugin_state(path).map(|(state, _)| state)
    }

    /// Read a track's grid and the plugin state saved with it (`None` when the
    /// file carries none).
    pub fn read_with_plugin_state(path: &Path) -> Result<(Self, Option<Vec<u8>>)> {
        let bytes =
            fs::read(path)
                .with_context(|| format!("reading track from {}", crate::file_label(path)))?;
        let (state, plugin_state) = Self::from_bytes_with_plugin_state(&bytes)
            .with_context(|| format!("parsing {}", crate::file_label(path)))?;
        Ok((state, (!plugin_state.is_empty()).then_some(plugin_state)))
    }

    /// Basic shape check: the grids match the declared dimensions. Used before
    /// handing a freshly parsed/exported state to the plugin.
    pub fn validate(&self) -> Result<()> {
        let grid = self.num_harmonics * self.num_buckets;
        if self.amplitude.len() != grid || self.phase.len() != grid {
            bail!("grid length mismatch");
        }
        if self.pitch_ratio.len() != self.num_buckets {
            bail!("pitch_ratio length mismatch");
        }
        // The exact-inverse tail is all-or-nothing: a partial set would be
        // silently mis-inverted rather than rejected.
        let tail = [self.bucket_lengths.len(), self.dc.len(), self.nyquist.len()];
        if !tail.iter().all(|&n| n == 0) && !tail.iter().all(|&n| n == self.num_buckets) {
            bail!("bucket_lengths/dc/nyquist must all be empty or all num_buckets long");
        }
        // Empty means "all enabled"; anything else must cover every harmonic, or
        // the harmonics past its end would take their state from where the
        // vector happens to stop.
        for flags in [&self.amp_enabled, &self.phase_enabled] {
            if !flags.is_empty() && flags.len() != self.num_harmonics {
                bail!("harmonic enable flags must be empty or num_harmonics long");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> TrackState {
        let (nh, nb) = (3usize, 4usize);
        let amplitude: Vec<f32> = (0..nh * nb).map(|i| i as f32 * 0.01).collect();
        let phase: Vec<f32> = (0..nh * nb).map(|i| (i as f32).sin()).collect();
        let pitch_ratio = vec![1.0, 1.01, 0.99, 1.0];
        TrackState {
            num_harmonics: nh,
            num_buckets: nb,
            base_freq: 220.0,
            duration_secs: 0.75,
            sample_rate: 44_100.0,
            display_gain: 15.4,
            amplitude,
            phase,
            pitch_ratio,
            bucket_lengths: vec![200, 198, 202, 200],
            dc: vec![0.01, -0.02, 0.03, 0.0],
            nyquist: vec![0.001, 0.0, -0.002, 0.0],
            // Harmonic 2's amplitude and harmonic 1's phase switched off.
            amp_enabled: vec![true, false, true],
            phase_enabled: vec![false, true, true],
        }
    }

    #[test]
    fn bytes_round_trip() {
        let s = sample_state();
        let parsed = TrackState::from_bytes(&s.to_bytes()).expect("parse");
        assert_eq!(parsed, s);
        s.validate().unwrap();
    }

    #[test]
    fn file_round_trip() {
        let s = sample_state();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lsft_test_{}.lsft", std::process::id()));
        s.write(&path).unwrap();
        let back = TrackState::read(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(back, s);
    }

    /// Build an older-version file from the current writer's output: same header
    /// minus the fields that version predates, and payload minus the v3 tail.
    fn downgrade(s: &TrackState, version: u32) -> Vec<u8> {
        let cur = s.to_bytes();
        // Strip whatever the target version predates: the plugin state's
        // length (v5; `to_bytes` writes no state), the enable flags (v4), then
        // bucket_lengths + dc + nyquist (v3).
        let plugin = 4;
        let flags = s.num_harmonics * 2;
        let exact = s.num_buckets * 3 * 4;
        let drop_tail = match version {
            0..=2 => plugin + flags + exact,
            3 => plugin + flags,
            4 => plugin,
            _ => 0,
        };
        let body = &cur[HEADER_V2_LEN..cur.len() - drop_tail];
        let mut out = Vec::new();
        out.extend_from_slice(&cur[..4]); // magic
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&cur[8..HEADER_V1_LEN]); // dims..sample_rate
        if version >= 2 {
            out.extend_from_slice(&cur[HEADER_V1_LEN..HEADER_V2_LEN]); // display_gain
        }
        out.extend_from_slice(body);
        out
    }

    /// Version 1 files (no recorded source level) must still load, reporting the
    /// gain as unknown rather than guessing one.
    #[test]
    fn reads_version_1_without_a_display_gain() {
        let s = sample_state();
        let parsed = TrackState::from_bytes(&downgrade(&s, 1)).expect("v1 parses");
        assert_eq!(parsed.display_gain, 0.0, "unknown, not invented");
        assert_eq!(
            TrackState {
                display_gain: 0.0,
                bucket_lengths: Vec::new(),
                dc: Vec::new(),
                nyquist: Vec::new(),
                amp_enabled: Vec::new(),
                phase_enabled: Vec::new(),
                ..s
            },
            parsed,
            "everything else must survive"
        );
    }

    /// Version 2 files predate the exact inverse. They must load, and must
    /// report that they cannot be inverted exactly rather than supplying zeros
    /// the audition would treat as real bucket lengths.
    #[test]
    fn reads_version_2_without_the_exact_inverse() {
        let s = sample_state();
        let parsed = TrackState::from_bytes(&downgrade(&s, 2)).expect("v2 parses");
        assert_eq!(parsed.display_gain, s.display_gain, "v2 does carry the gain");
        assert!(parsed.bucket_lengths.is_empty());
        assert!(parsed.dc.is_empty());
        assert!(parsed.nyquist.is_empty());
        assert!(
            !parsed.supports_exact_inverse(),
            "a v2 track must not claim an exact inverse it has no data for"
        );
        assert_eq!(parsed.amplitude, s.amplitude);
        assert_eq!(parsed.pitch_ratio, s.pitch_ratio);
    }

    /// Version 3 predates the enable checkboxes. It must load with them empty —
    /// "all enabled", which is what a v3 file actually plays as.
    #[test]
    fn reads_version_3_without_the_enable_flags() {
        let s = sample_state();
        let parsed = TrackState::from_bytes(&downgrade(&s, 3)).expect("v3 parses");
        assert!(parsed.amp_enabled.is_empty());
        assert!(parsed.phase_enabled.is_empty());
        assert_eq!(parsed.bucket_lengths, s.bucket_lengths, "v3 does carry these");
    }

    /// The point of version 4: a harmonic switched off in the editor is still
    /// switched off when the track is loaded back, instead of the whole spectrum
    /// coming back on.
    #[test]
    fn version_4_carries_the_enable_flags() {
        let s = sample_state();
        let back = TrackState::from_bytes(&s.to_bytes()).expect("parses");
        assert_eq!(back.amp_enabled, s.amp_enabled);
        assert_eq!(back.phase_enabled, s.phase_enabled);
    }

    /// Version 4 predates the plugin state. It must load, with none.
    #[test]
    fn reads_version_4_without_a_plugin_state() {
        let s = sample_state();
        let (parsed, plugin) =
            TrackState::from_bytes_with_plugin_state(&downgrade(&s, 4)).expect("v4 parses");
        assert_eq!(parsed, s);
        assert!(plugin.is_empty());
    }

    /// The point of version 5: the plugin's own state — the Modulation
    /// envelope, "Zero phases on keys", the curve controls — rides in the file
    /// and comes back byte for byte, beside an untouched grid.
    #[test]
    fn version_5_carries_the_plugin_state() {
        let s = sample_state();
        let plugin = br#"{"params":{"env_enabled":{"bool":true}}}"#.to_vec();
        let bytes = s.to_bytes_with_plugin_state(&plugin);
        let (back, back_plugin) =
            TrackState::from_bytes_with_plugin_state(&bytes).expect("parses");
        assert_eq!(back, s);
        assert_eq!(back_plugin, plugin);
        // The grid-only reader takes the same file.
        assert_eq!(TrackState::from_bytes(&bytes).expect("parses"), s);
        // A state cut short is corruption, not a shorter state.
        assert!(TrackState::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }

    /// An all-enabled selection is the default, so it round-trips as the default
    /// (empty) rather than as a vector of `true` that only compares equal by
    /// accident.
    #[test]
    fn an_all_enabled_selection_round_trips_as_the_default() {
        let s = TrackState {
            amp_enabled: Vec::new(),
            phase_enabled: vec![true; 3],
            ..sample_state()
        };
        let back = TrackState::from_bytes(&s.to_bytes()).expect("parses");
        assert!(back.amp_enabled.is_empty());
        assert!(back.phase_enabled.is_empty(), "all-true collapses to the default");
    }

    /// Flags that cover only some harmonics are a bug: the rest would take their
    /// state from wherever the vector stops.
    #[test]
    fn validate_rejects_partial_enable_flags() {
        let s = TrackState { amp_enabled: vec![true, false], ..sample_state() };
        assert!(s.validate().is_err());
    }

    /// A grid with no exact inverse (hand-drawn, or loaded from an old file)
    /// writes a v3 file whose tail is zeroed, and reads back as "not exact"
    /// rather than as buckets of length zero.
    #[test]
    fn a_grid_without_an_exact_inverse_round_trips_as_such() {
        let s = TrackState {
            bucket_lengths: Vec::new(),
            dc: Vec::new(),
            nyquist: Vec::new(),
            ..sample_state()
        };
        assert!(!s.supports_exact_inverse());
        let back = TrackState::from_bytes(&s.to_bytes()).expect("parses");
        assert_eq!(back, s);
        assert!(!back.supports_exact_inverse());
    }

    /// The whole point of version 3: what the exact inverse needs survives a
    /// save/load, so a reloaded track auditions exactly rather than through the
    /// transposing renderer.
    #[test]
    fn version_3_carries_the_exact_inverse() {
        let s = sample_state();
        assert!(s.supports_exact_inverse());
        let back = TrackState::from_bytes(&s.to_bytes()).expect("parses");
        assert_eq!(back.bucket_lengths, s.bucket_lengths);
        assert_eq!(back.dc, s.dc);
        assert_eq!(back.nyquist, s.nyquist);
        assert!(back.supports_exact_inverse());
    }

    /// A partial tail is a bug, not something to silently half-apply.
    #[test]
    fn validate_rejects_a_partial_exact_inverse() {
        let s = TrackState { dc: Vec::new(), ..sample_state() };
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample_state().to_bytes();
        bytes[0] = b'X';
        assert!(TrackState::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_truncated() {
        let bytes = sample_state().to_bytes();
        assert!(TrackState::from_bytes(&bytes[..bytes.len() - 8]).is_err());
        assert!(TrackState::from_bytes(&bytes[..3]).is_err());
    }

    #[test]
    fn rejects_wrong_declared_size() {
        // Tamper num_buckets so the declared payload no longer matches the bytes.
        let mut bytes = sample_state().to_bytes();
        bytes[12..16].copy_from_slice(&999u32.to_le_bytes()); // num_buckets field
        assert!(TrackState::from_bytes(&bytes).is_err());
    }
}
