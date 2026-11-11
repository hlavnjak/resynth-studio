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
//! Split a decoded signal into "subtracks" of roughly constant periodicity.
//!
//! This is the host-side half of the resynthesis pipeline. We run a coarse
//! autocorrelation pitch tracker over short frames, then merge consecutive
//! voiced frames whose detected fundamental stays within ~a semitone into one
//! subtrack. Each resulting subtrack is a region with a single, stable pitch —
//! the natural unit to hand to LeSynth Fourier, which then subdivides it into
//! per-period buckets for harmonic analysis.

/// A contiguous region of the signal with an approximately constant pitch.
#[derive(Debug, Clone)]
pub struct Subtrack {
    /// Sample index range `[start, end)` into the source signal.
    pub start: usize,
    pub end: usize,
    /// Estimated fundamental frequency (Hz): the median of the per-frame track.
    /// This is the transpose reference for resynthesis.
    pub base_freq: f32,
    /// Mean voicedness confidence in [0, 1].
    pub confidence: f32,
    /// Per-frame fundamental track within this subtrack, as
    /// `(frame_center_sample, freq_hz)` ordered by sample position. Retained so
    /// downstream stages (period-synchronous bucketing, per-bucket DFT) can
    /// follow vibrato/drift instead of collapsing it into `base_freq`. Always
    /// non-empty for a flushed subtrack.
    pub pitch_track: Vec<(usize, f32)>,
}

impl Subtrack {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn duration_secs(&self, sample_rate: f32) -> f32 {
        self.len() as f32 / sample_rate.max(1.0)
    }

    /// Local fundamental (Hz) at an absolute `sample_pos`, linearly interpolated
    /// from [`Self::pitch_track`] and clamped to the track's range outside its
    /// span. Falls back to [`Self::base_freq`] if the track is empty. This is the
    /// `f0(t)` that period-synchronous bucketing (Stage B) and the per-bucket DFT
    /// (Stage C) consume.
    pub fn freq_at(&self, sample_pos: usize) -> f32 {
        let track = &self.pitch_track;
        match track.len() {
            0 => self.base_freq,
            1 => track[0].1,
            _ => {
                if sample_pos <= track[0].0 {
                    return track[0].1;
                }
                if sample_pos >= track[track.len() - 1].0 {
                    return track[track.len() - 1].1;
                }
                // Find the bracketing pair (track is sorted by position).
                let hi = track.partition_point(|&(p, _)| p <= sample_pos);
                let (p0, f0) = track[hi - 1];
                let (p1, f1) = track[hi];
                if p1 == p0 {
                    return f0;
                }
                let frac = (sample_pos - p0) as f32 / (p1 - p0) as f32;
                f0 + (f1 - f0) * frac
            }
        }
    }

    /// "Reasonable to analyse": long enough to contain many periods and
    /// *strongly* pitched on average. Weakly-periodic regions (DC clicks,
    /// silence, attack/release noise) are skipped so they don't produce empty
    /// charts. The bar here is higher than the per-frame voiced gate used for
    /// merging, so a sustained note stays in one piece but junk is dropped.
    pub fn is_reasonable(&self, sample_rate: f32) -> bool {
        let periods = self.duration_secs(sample_rate) * self.base_freq;
        self.confidence >= REASONABLE_CONFIDENCE && periods >= 20.0 && self.base_freq > 0.0
    }
}

/// Resolution of the pitch contour handed to LeSynth Fourier per subtrack.
/// Vibrato is slow, so a few hundred points capture it with room to spare.
pub const CONTOUR_POINTS: usize = 256;

/// Uniformly-resampled fundamental (absolute Hz) across a subtrack's span, for
/// the analysis bridge. Lets the plugin follow vibrato/drift instead of a single
/// global pitch. Empty when the subtrack carries no pitch track (→ flat).
///
/// Lives here rather than in the GUI so the resynthesis tests feed the plugin
/// the exact contour the app does.
pub fn build_contour(sub: &Subtrack) -> Vec<f32> {
    if sub.pitch_track.is_empty() {
        return Vec::new();
    }
    let len = sub.len().max(1);
    (0..CONTOUR_POINTS)
        .map(|i| {
            let off = (i as f32 + 0.5) / CONTOUR_POINTS as f32 * len as f32;
            sub.freq_at(sub.start + off as usize)
        })
        .collect()
}

const FRAME: usize = 2048;
const HOP: usize = 1024;
const MIN_FREQ: f32 = 50.0;
const MAX_FREQ: f32 = 1000.0;
/// Per-frame voiced gate used while merging frames into runs (kept low so a
/// note with vibrato stays in one piece).
const MIN_CONFIDENCE: f32 = 0.4;
/// Mean-confidence bar a finished subtrack must clear to be offered for
/// analysis (high enough to reject DC clicks / noise / silence).
const REASONABLE_CONFIDENCE: f32 = 0.75;
/// Pitch tolerance when merging frames: ~1 semitone (2^(1/12) ≈ 1.059).
const MERGE_RATIO: f32 = 1.06;

/// How far past the first qualifying NAC peak the true period's peak can still
/// be: side peaks sit one formant cycle from it, and a first formant at 500 Hz
/// under a 100 Hz voice puts one 20% short. Well under 2, where the octave is.
const SAME_LOBE: f32 = 1.35;

struct FramePitch {
    freq: f32,
    confidence: f32,
}

/// Estimate the fundamental of one frame via normalised autocorrelation.
/// Returns `(freq, confidence)`; `freq == 0` means unvoiced.
fn estimate_frame_pitch(frame: &[f32], sample_rate: f32) -> FramePitch {
    let min_lag = (sample_rate / MAX_FREQ).floor().max(2.0) as usize;
    let max_lag = ((sample_rate / MIN_FREQ).ceil() as usize).min(frame.len() / 2);
    if max_lag <= min_lag {
        return FramePitch { freq: 0.0, confidence: 0.0 };
    }

    let energy: f32 = frame.iter().map(|s| s * s).sum();
    if energy <= 1e-6 {
        return FramePitch { freq: 0.0, confidence: 0.0 };
    }

    // Normalised autocorrelation coefficient per lag. Normalising by the energy
    // of *each overlapping window* (not the whole frame) removes the bias that
    // would otherwise favour short lags / too-high pitches.
    let mut nac = vec![0.0f32; max_lag + 1];
    let mut global_max = 0.0f32;
    for lag in min_lag..=max_lag {
        let n = frame.len() - lag;
        let mut corr = 0.0f32;
        let mut e1 = 0.0f32;
        let mut e2 = 0.0f32;
        for i in 0..n {
            let a = frame[i];
            let c = frame[i + lag];
            corr += a * c;
            e1 += a * a;
            e2 += c * c;
        }
        let denom = (e1 * e2).sqrt();
        if denom > 1e-9 {
            let v = corr / denom;
            nac[lag] = v;
            if v > global_max {
                global_max = v;
            }
        }
    }

    if global_max <= 0.0 {
        return FramePitch { freq: 0.0, confidence: 0.0 };
    }

    // First-peak picking: the fundamental is the *smallest* lag with a local
    // maximum close to the global best. This avoids octave-down errors, where
    // the NAC peaks just as strongly at twice the period.
    let threshold = 0.85 * global_max;
    let mut chosen = min_lag;
    let mut chosen_val = global_max;
    for lag in (min_lag + 1)..max_lag {
        if nac[lag] >= threshold && nac[lag] >= nac[lag - 1] && nac[lag] >= nac[lag + 1] {
            chosen = lag;
            chosen_val = nac[lag];
            break;
        }
    }
    // …and then the *top* of that peak's lobe, which the first local maximum
    // need not be. A formant ripples the NAC, so the true period's peak has side
    // peaks a formant cycle either side of it, and on a voice they clear the
    // threshold too (0.90 against 0.99, at lag 412 against 424). Stopping on the
    // first reads the pitch 3% sharp. Anything higher within reach is the same
    // lobe — the next period's peak is a whole octave away, far past this.
    let reach = ((chosen as f32 * SAME_LOBE).ceil() as usize).min(max_lag - 1);
    for lag in (chosen + 1)..=reach {
        if nac[lag] > chosen_val && nac[lag] >= nac[lag - 1] && nac[lag] >= nac[lag + 1] {
            chosen = lag;
            chosen_val = nac[lag];
        }
    }

    // Sub-sample refinement: the whole-lag grid is coarse at high pitch (a D5 at
    // 22 kHz sits at lag 37, and lag 38 is *46 cents* away), so without it a
    // smooth vibrato is reported as a staircase mostly stuck on one step, which
    // desynchronises the plugin's per-bucket DFT. A parabola through the NAC
    // peak recovers the fraction; `delta` is clamped to ±½ lag.
    let refined = if chosen > min_lag && chosen < max_lag {
        let (y0, y1, y2) = (nac[chosen - 1], nac[chosen], nac[chosen + 1]);
        let denom = y0 - 2.0 * y1 + y2;
        if denom.abs() > 1e-9 {
            chosen as f32 + (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
        } else {
            chosen as f32
        }
    } else {
        chosen as f32
    };

    FramePitch {
        freq: sample_rate / refined.max(1.0),
        confidence: chosen_val.clamp(0.0, 1.0),
    }
}

/// Segment a mono signal into pitch-stable subtracks.
pub fn segment(samples: &[f32], sample_rate: f32) -> Vec<Subtrack> {
    if samples.len() < FRAME || sample_rate <= 0.0 {
        return Vec::new();
    }

    // 1) Per-frame pitch track.
    let mut frames: Vec<(usize, FramePitch)> = Vec::new();
    let mut pos = 0;
    while pos + FRAME <= samples.len() {
        let fp = estimate_frame_pitch(&samples[pos..pos + FRAME], sample_rate);
        frames.push((pos, fp));
        pos += HOP;
    }

    // 2) Merge consecutive voiced frames with a stable fundamental.
    let mut subtracks = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut run_freqs: Vec<f32> = Vec::new();
    let mut run_confs: Vec<f32> = Vec::new();
    let mut run_centers: Vec<usize> = Vec::new();
    let mut last_end = 0usize;

    let flush = |start: usize,
                 end: usize,
                 freqs: &[f32],
                 confs: &[f32],
                 centers: &[usize],
                 out: &mut Vec<Subtrack>| {
        if freqs.is_empty() {
            return;
        }
        let mut sorted = freqs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let base_freq = sorted[sorted.len() / 2]; // median
        let confidence = confs.iter().sum::<f32>() / confs.len() as f32;
        let pitch_track = centers.iter().zip(freqs).map(|(&p, &f)| (p, f)).collect();
        out.push(Subtrack { start, end, base_freq, confidence, pitch_track });
    };

    for (start, fp) in &frames {
        let voiced = fp.freq >= MIN_FREQ && fp.freq <= MAX_FREQ && fp.confidence >= MIN_CONFIDENCE;
        let frame_end = (start + FRAME).min(samples.len());
        // Anchor each frame's pitch at its centre — the best single time for a
        // window-averaged estimate, and what `freq_at` interpolates between.
        let frame_center = start + FRAME / 2;

        if voiced {
            let continues = run_start.is_some()
                && run_freqs
                    .last()
                    .map(|&prev| {
                        let r = (fp.freq / prev).max(prev / fp.freq);
                        r <= MERGE_RATIO
                    })
                    .unwrap_or(false);

            if continues {
                run_freqs.push(fp.freq);
                run_confs.push(fp.confidence);
                run_centers.push(frame_center);
                last_end = frame_end;
            } else {
                // Close the previous run and start a new one.
                if let Some(s) = run_start {
                    flush(s, last_end, &run_freqs, &run_confs, &run_centers, &mut subtracks);
                }
                run_start = Some(*start);
                run_freqs = vec![fp.freq];
                run_confs = vec![fp.confidence];
                run_centers = vec![frame_center];
                last_end = frame_end;
            }
        } else if let Some(s) = run_start.take() {
            flush(s, last_end, &run_freqs, &run_confs, &run_centers, &mut subtracks);
            run_freqs.clear();
            run_confs.clear();
            run_centers.clear();
        }
    }
    if let Some(s) = run_start {
        flush(s, last_end, &run_freqs, &run_confs, &run_centers, &mut subtracks);
    }

    subtracks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn detects_a_steady_tone() {
        let sr = 44_100.0;
        let freq = 220.0;
        let n = sr as usize; // 1 s
        let sig: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect();

        let subs = segment(&sig, sr);
        assert!(!subs.is_empty(), "should find at least one subtrack");
        let s = &subs[0];
        assert!((s.base_freq - freq).abs() < 10.0, "freq off: {}", s.base_freq);
        assert!(s.is_reasonable(sr));
    }

    #[test]
    fn keeps_a_per_frame_pitch_track() {
        let sr = 44_100.0;
        let freq = 220.0;
        let n = sr as usize; // 1 s
        let sig: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect();

        let subs = segment(&sig, sr);
        let s = &subs[0];
        // Multiple frames captured, ordered by sample position, all near 220 Hz.
        assert!(s.pitch_track.len() > 5, "track too short: {}", s.pitch_track.len());
        assert!(
            s.pitch_track.windows(2).all(|w| w[0].0 <= w[1].0),
            "pitch_track not sorted by position"
        );
        assert!(s.pitch_track.iter().all(|&(_, f)| (f - freq).abs() < 10.0));
    }

    #[test]
    fn freq_at_interpolates_and_clamps() {
        let sub = Subtrack {
            start: 0,
            end: 400,
            base_freq: 200.0,
            confidence: 1.0,
            pitch_track: vec![(100, 200.0), (300, 240.0)],
        };
        // Clamped below/above the track span.
        assert_eq!(sub.freq_at(0), 200.0);
        assert_eq!(sub.freq_at(1000), 240.0);
        // Midpoint interpolates linearly.
        assert!((sub.freq_at(200) - 220.0).abs() < 1e-3);
    }

    #[test]
    fn freq_at_empty_track_falls_back_to_base() {
        let sub = Subtrack {
            start: 0,
            end: 10,
            base_freq: 123.0,
            confidence: 0.0,
            pitch_track: Vec::new(),
        };
        assert_eq!(sub.freq_at(5), 123.0);
    }

    /// The whole-lag grid is coarse at high pitch: at 22.05 kHz a ~590 Hz tone
    /// sits between lags 37 (595.9 Hz) and 38 (580.3 Hz), 46 cents apart. Without
    /// sub-sample refinement of the autocorrelation peak the tracker can only
    /// report one of those two, so a pitch that falls between them is off by tens
    /// of cents and a smooth vibrato comes out as a staircase.
    #[test]
    fn pitch_is_refined_between_whole_sample_lags() {
        let sr = 22_050.0;
        for &freq in &[588.0f32, 590.5, 593.0] {
            let n = FRAME * 4;
            let sig: Vec<f32> = (0..n)
                .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
                .collect();
            let subs = segment(&sig, sr);
            let s = subs.first().expect("a steady tone must be detected");
            let cents = 1200.0 * (s.base_freq / freq).log2();
            assert!(
                cents.abs() < 8.0,
                "{} Hz detected as {:.2} Hz ({:+.1} cents) — whole-lag quantisation",
                freq,
                s.base_freq,
                cents
            );
            // And the nearest whole lags really are far away, so this is not
            // passing by accident.
            let lag = sr / freq;
            assert!(
                (1200.0 * (lag / lag.round()).log2()).abs() > 8.0,
                "test frequency {} lands on a whole lag; pick another",
                freq
            );
        }
    }

    /// A voice's formant ripples the autocorrelation: beside the peak at the true
    /// period sit side peaks one formant cycle away, and they clear the
    /// first-peak threshold. Taking the first of them read a 104 Hz voice as
    /// 107 Hz — 3% short, for half the note — so every "one period" bucket the
    /// plugin cut was an incomplete cycle, and every key but the source's own
    /// played the missing piece as a phase jump at each splice.
    #[test]
    fn a_formant_side_peak_is_not_the_period() {
        let sr = 44_100.0f32;
        let period = 424.3f32;
        let w = 2.0 * std::f32::consts::PI / period;
        let samples: Vec<f32> = (0..sr as usize / 2)
            .map(|i| {
                let t = i as f32 * w;
                let low: f32 = (1..=3).map(|k| (k as f32 * t).sin()).sum();
                // A strong formant around the 32nd harmonic.
                let formant = (32.0 * t).sin() + (33.0 * t + 1.0).sin();
                0.2 * low + 0.18 * formant
            })
            .collect();
        let subs = segment(&samples, sr);
        assert!(!subs.is_empty(), "the tone must be found");
        let want = sr / period;
        for &(_, f) in &subs[0].pitch_track {
            assert!(
                (f / want - 1.0).abs() < 0.003,
                "tracked {f:.2} Hz, the period says {want:.2}"
            );
        }
    }

    #[test]
    fn silence_yields_nothing() {
        let subs = segment(&vec![0.0; 44_100], 44_100.0);
        assert!(subs.iter().all(|s| !s.is_reasonable(44_100.0)));
    }
}