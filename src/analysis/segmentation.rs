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

/// Length of one analysis frame at ordinary sample rates.
const FRAME: usize = 2048;
const MIN_FREQ: f32 = 50.0;
/// Top of the range the period is *searched* in. Not simply raised to cover
/// high notes: under a millisecond the autocorrelation of any ordinary sound is
/// high and rippled by its upper formants, and a search let in there reports
/// breath and consonants as confident 3 kHz notes.
const MAX_FREQ: f32 = 1000.0;
/// Top of the range a pitch may be *reported* in — C8, the top of a piano.
/// A tone above [`MAX_FREQ`] is found at two or three of its periods, the
/// shortest lag the search reaches, and then walked back up by
/// [`OCTAVE_UP_RATIO`].
const REPORT_MAX_FREQ: f32 = 4200.0;
/// How nearly the autocorrelation at a fraction of the chosen lag has to match
/// the chosen peak for that fraction to be the real period. Two periods of a tone
/// correlate exactly as one does; a real note an octave lower does not, as its
/// odd harmonics are in antiphase there.
const OCTAVE_UP_RATIO: f32 = 0.95;
/// …and how far the autocorrelation has to fall at half of *that* lag, as a
/// share of its peak. Hiss and room tone correlate at every short lag alike and
/// fall nowhere, which is what tells them from a tone.
const OCTAVE_UP_TROUGH: f32 = 0.5;
/// A frame this far below the loudest one in the file is silence, whatever its
/// autocorrelation says: the coefficient is scale-free, so hum and dither a
/// thousandth the level of the music are as "periodic" as the music.
const SILENCE_RATIO: f32 = 1e-6;

/// Samples in one analysis frame: [`FRAME`], unless that is too short to hold
/// two of the longest period searched for. A fixed count is a fixed *time* only
/// at one sample rate — 2048 samples at 96 kHz are 21 ms, and nothing under
/// 94 Hz repeats inside that — so above about 51 kHz the frame grows with the
/// rate. Below it nothing changes, and neither does anything analysed so far.
fn frame_len(sample_rate: f32) -> usize {
    let two_periods = 2 * (sample_rate / MIN_FREQ).ceil() as usize;
    two_periods.max(FRAME)
}

/// Corner of the high-pass the signal is put through before it is tracked.
/// Nothing under [`MIN_FREQ`] can be reported as a pitch, but all of it is in
/// the autocorrelation — and the worst of it is a *step*: the jump a recording
/// opens with as its input settles is in one frame a block of offset beside a
/// block of none, which no line fits and which correlates with itself at every
/// lag. Through the filter the step is a pulse a few milliseconds long.
const RUMBLE_HZ: f32 = 22.0;

/// The signal with everything well under the pitch range taken out — two
/// passes of a one-pole DC blocker, 12 dB an octave below [`RUMBLE_HZ`]. Only
/// the tracker listens to this; the regions it finds index the original.
fn without_rumble(samples: &[f32], sample_rate: f32) -> Vec<f32> {
    let r = (1.0 - std::f32::consts::TAU * RUMBLE_HZ / sample_rate).clamp(0.0, 0.9999);
    let mut out = samples.to_vec();
    for _ in 0..2 {
        let (mut prev_in, mut prev_out) = (out.first().copied().unwrap_or(0.0), 0.0f32);
        for s in out.iter_mut() {
            let y = *s - prev_in + r * prev_out;
            prev_in = *s;
            prev_out = y;
            *s = y;
        }
    }
    out
}

/// A frame with its own straight-line trend taken out — its mean, and the slope
/// it sits on.
///
/// Neither carries a pitch, and both are in the autocorrelation: a DC offset
/// correlates perfectly with itself at every lag, and so does the slow swing a
/// recording opens with while its offset settles, which across one frame is a
/// ramp. Left in, either reads as a confident note wherever the ripple on top
/// happens to peak. A line is all that is removed, so a low note — a few
/// periods to the frame — is left as it was.
fn detrended(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mid = (n - 1) as f64 / 2.0;
    let mean = frame.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &s) in frame.iter().enumerate() {
        let x = i as f64 - mid;
        num += x * (s as f64 - mean);
        den += x * x;
    }
    let slope = if den > 0.0 { num / den } else { 0.0 };
    frame
        .iter()
        .enumerate()
        .map(|(i, &s)| (s as f64 - mean - slope * (i as f64 - mid)) as f32)
        .collect()
}

/// Energy of a frame once its trend is out — see [`detrended`].
fn frame_energy(frame: &[f32]) -> f32 {
    detrended(frame).iter().map(|s| s * s).sum()
}

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
///
/// `floor` is the energy under which the frame is silence — see
/// [`SILENCE_RATIO`].
fn estimate_frame_pitch(frame: &[f32], sample_rate: f32, floor: f32) -> FramePitch {
    let level = detrended(frame);
    let frame = &level[..];
    let min_lag = (sample_rate / MAX_FREQ).floor().max(2.0) as usize;
    // Lags under the search range, looked at only by the octave check below.
    let short_lag = ((sample_rate / REPORT_MAX_FREQ).floor().max(2.0) as usize).min(min_lag);
    let max_lag = ((sample_rate / MIN_FREQ).ceil() as usize).min(frame.len() / 2);
    if max_lag <= min_lag {
        return FramePitch { freq: 0.0, confidence: 0.0 };
    }

    let energy: f32 = frame.iter().map(|s| s * s).sum();
    if energy <= 1e-6 || energy < floor {
        return FramePitch { freq: 0.0, confidence: 0.0 };
    }

    // Normalised autocorrelation coefficient per lag. Normalising by the energy
    // of *each overlapping window* (not the whole frame) removes the bias that
    // would otherwise favour short lags / too-high pitches.
    let mut nac = vec![0.0f32; max_lag + 1];
    let mut global_max = 0.0f32;
    for lag in 1..=max_lag {
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
            if lag >= min_lag && v > global_max {
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
    let first_peak = ((min_lag + 1)..max_lag).find(|&lag| {
        nac[lag] >= threshold && nac[lag] >= nac[lag - 1] && nac[lag] >= nac[lag + 1]
    });
    // No peak at all: the autocorrelation only falls away across the range, or
    // only climbs. Its largest value is then at an edge of the search, which is
    // a statement about the range and not about the sound.
    let Some(mut chosen) = first_peak else {
        return FramePitch { freq: 0.0, confidence: 0.0 };
    };
    let mut chosen_val = nac[chosen];
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

    // A tone above the search range was found at two or three of its periods,
    // the shortest lag there was to find it at. One period then correlates just
    // as well — which a fraction of the lag never does for a note that really
    // is that low — so the lag is divided by the largest count that still lands
    // on such a peak. Only lags the search could not have reached are tried:
    // anything longer it has already ruled on.
    //
    // And only for a lag found at the very top of the range. Whole periods of a
    // tone too high to search for are less than one shortest lag apart, so the
    // first of them inside the range is under twice that lag; a voice found
    // further down has formants up here, and they must not be taken for it.
    let found_at = peak_height(&nac, chosen);
    let found_lag = chosen;
    let counts = if found_lag < 2 * min_lag { found_lag / short_lag.max(1) } else { 0 };
    for count in (2..=counts).rev() {
        let part = (found_lag as f32 / count as f32).round() as usize;
        if part >= min_lag || part < short_lag.max(2) {
            continue;
        }
        let best = (part - 1..=part + 1)
            .max_by(|&a, &b| nac[a].total_cmp(&nac[b]))
            .unwrap_or(part);
        let is_peak = nac[best] >= nac[best - 1] && nac[best] >= nac[best + 1];
        // …and it has to be a period, not a stretch of the slope every dull
        // sound has near lag zero, where *everything* correlates: half a real
        // period is where a tone is most unlike itself.
        let trough = nac[best / 2] <= OCTAVE_UP_TROUGH * nac[best];
        if is_peak && trough && peak_height(&nac, best) >= OCTAVE_UP_RATIO * found_at {
            chosen = best;
            chosen_val = nac[best];
            break;
        }
    }

    // Sub-sample refinement: the whole-lag grid is coarse at high pitch (a D5 at
    // 22 kHz sits at lag 37, and lag 38 is *46 cents* away), so without it a
    // smooth vibrato is reported as a staircase mostly stuck on one step, which
    // desynchronises the plugin's per-bucket DFT. A parabola through the NAC
    // peak recovers the fraction; `delta` is clamped to ±½ lag.
    let refined = if chosen > short_lag && chosen < max_lag {
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

/// Height of the autocorrelation peak at `lag`, read off a parabola through it
/// and its neighbours. At a period of thirty samples the true peak falls
/// between two lags and the taller of them is still a few percent short of it,
/// which is the whole margin two peaks are being compared by.
fn peak_height(nac: &[f32], lag: usize) -> f32 {
    if lag == 0 || lag + 1 >= nac.len() {
        return nac.get(lag).copied().unwrap_or(0.0);
    }
    let (y0, y1, y2) = (nac[lag - 1], nac[lag], nac[lag + 1]);
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() <= 1e-9 {
        return y1;
    }
    let delta = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
    y1 - 0.25 * (y0 - y2) * delta
}

/// Segment a mono signal into pitch-stable subtracks.
pub fn segment(samples: &[f32], sample_rate: f32) -> Vec<Subtrack> {
    if sample_rate <= 0.0 {
        return Vec::new();
    }
    let frame_len = frame_len(sample_rate);
    let hop = frame_len / 2;
    if samples.len() < frame_len {
        return Vec::new();
    }
    let filtered = without_rumble(samples, sample_rate);
    let samples = &filtered[..];
    let loudest = samples
        .windows(frame_len)
        .step_by(hop)
        .map(frame_energy)
        .fold(0.0f32, f32::max);
    let floor = loudest * SILENCE_RATIO;

    // 1) Per-frame pitch track.
    let mut frames: Vec<(usize, FramePitch)> = Vec::new();
    let mut pos = 0;
    while pos + frame_len <= samples.len() {
        let fp = estimate_frame_pitch(&samples[pos..pos + frame_len], sample_rate, floor);
        frames.push((pos, fp));
        pos += hop;
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
        let voiced =
            fp.freq >= MIN_FREQ && fp.freq <= REPORT_MAX_FREQ && fp.confidence >= MIN_CONFIDENCE;
        let frame_end = (start + frame_len).min(samples.len());
        // Anchor each frame's pitch at its centre — the best single time for a
        // window-averaged estimate, and what `freq_at` interpolates between.
        let frame_center = start + frame_len / 2;

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
                // Close the previous run and start a new one. Where the pitch
                // changed between two frames that overlap, the audio they share
                // would belong to both runs; it is cut down the middle instead,
                // halfway between the two frames' centres.
                let mut begin = *start;
                if let Some(s) = run_start {
                    if last_end > begin {
                        begin = (begin + hop / 2).min(last_end);
                        last_end = begin;
                    }
                    flush(s, last_end, &run_freqs, &run_confs, &run_centers, &mut subtracks);
                }
                run_start = Some(begin);
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

    /// A tone with a few harmonics, the way an instrument has them.
    fn rich_tone(freq: f32, sr: f32, secs: f32) -> Vec<f32> {
        (0..(sr * secs) as usize)
            .map(|i| {
                let t = i as f32 / sr;
                [1.0f32, 0.5, 0.33, 0.25, 0.2]
                    .iter()
                    .enumerate()
                    .map(|(k, a)| a * (2.0 * PI * freq * (k + 1) as f32 * t).sin())
                    .sum::<f32>()
                    * 0.3
            })
            .collect()
    }

    /// White noise in ±`level`, the same every run.
    fn noise(n: usize, level: f32) -> Vec<f32> {
        let mut x = 12345u32;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 8) as f32 / 8_388_608.0 - 1.0) * level
            })
            .collect()
    }

    /// 2048 samples at 96 kHz are 21 ms — less than two periods of anything
    /// under 94 Hz, which used to come back as "1000 Hz" and be thrown away.
    /// The frame grows with the rate, so a low note is a low note at any rate.
    #[test]
    fn a_low_note_is_found_at_a_high_sample_rate() {
        for sr in [44_100.0f32, 48_000.0, 88_200.0, 96_000.0, 192_000.0] {
            let subs = segment(&rich_tone(70.0, sr, 1.0), sr);
            assert_eq!(subs.len(), 1, "{sr} Hz: {subs:?}");
            assert!((subs[0].base_freq - 70.0).abs() < 0.5, "{sr} Hz: {}", subs[0].base_freq);
            assert!(subs[0].is_reasonable(sr));
        }
    }

    /// A tone above the search range was found at twice its period and reported
    /// an octave low, at full confidence. It is walked back up — and a note that
    /// really is in the octave below is left exactly where it is.
    #[test]
    fn a_note_above_the_search_range_is_not_reported_an_octave_low() {
        let sr = 44_100.0;
        for freq in [1200.0f32, 1500.0, 2093.0, 3000.0] {
            let subs = segment(&rich_tone(freq, sr, 1.0), sr);
            assert_eq!(subs.len(), 1, "{freq} Hz: {subs:?}");
            let cents = 1200.0 * (subs[0].base_freq / freq).log2();
            assert!(cents.abs() < 15.0, "{freq} Hz read as {} Hz", subs[0].base_freq);
        }
        for freq in [520.0f32, 600.0, 750.0, 990.0] {
            let subs = segment(&rich_tone(freq, sr, 1.0), sr);
            assert_eq!(subs.len(), 1, "{freq} Hz: {subs:?}");
            let cents = 1200.0 * (subs[0].base_freq / freq).log2();
            assert!(cents.abs() < 15.0, "{freq} Hz read as {} Hz", subs[0].base_freq);
        }
        // Hiss correlates at every short lag alike; it is not a high note.
        assert!(segment(&noise(44_100, 0.3), sr).iter().all(|s| !s.is_reasonable(sr)));
    }

    /// A DC offset correlates perfectly with itself at every lag, so a stretch
    /// of offset with next to nothing on it used to be a row of confident
    /// "notes". The offset is taken out before anything is measured — and a
    /// real tone riding on one is found as before.
    #[test]
    fn a_dc_offset_is_not_a_pitch() {
        let sr = 44_100.0;
        let offset: Vec<f32> = noise(44_100, 0.0005).iter().map(|s| s + 0.3).collect();
        assert!(segment(&offset, sr).is_empty(), "{:?}", segment(&offset, sr));

        let riding: Vec<f32> = rich_tone(220.0, sr, 1.0).iter().map(|s| s * 0.2 + 0.5).collect();
        let subs = segment(&riding, sr);
        assert_eq!(subs.len(), 1);
        assert!((subs[0].base_freq - 220.0).abs() < 0.5);
    }

    /// The autocorrelation coefficient is scale-free: mains hum 70 dB under the
    /// music is as periodic as the music. Against the loudest frame in the file
    /// it is silence.
    #[test]
    fn hum_far_below_the_music_is_silence() {
        let sr = 44_100.0;
        let mut sig = rich_tone(220.0, sr, 1.0);
        let hum_from = sig.len();
        sig.extend((0..44_100).map(|i| 0.0002 * (2.0 * PI * 50.0 * i as f32 / sr).sin()));
        let subs = segment(&sig, sr);
        assert_eq!(subs.len(), 1, "{subs:?}");
        assert!(subs[0].end <= hum_from + FRAME);
    }

    /// Frames overlap by half, so where one note runs straight into the next
    /// the last frame of the one and the first of the other share audio — and
    /// both regions used to claim it. It is cut once, near the change.
    #[test]
    fn two_notes_run_together_share_no_audio() {
        let sr = 44_100.0;
        let mut sig = rich_tone(220.0, sr, 1.0);
        let change = sig.len();
        sig.extend(rich_tone(261.63, sr, 1.0));
        let subs = segment(&sig, sr);
        assert_eq!(subs.len(), 2, "{subs:?}");
        assert_eq!(subs[0].end, subs[1].start, "the regions overlap or leave a gap");
        assert!(subs[0].end.abs_diff(change) <= FRAME / 2, "cut at {}", subs[0].end);
        assert!((subs[0].base_freq - 220.0).abs() < 1.0);
        assert!((subs[1].base_freq - 261.63).abs() < 1.0);
    }

    #[test]
    fn silence_yields_nothing() {
        let subs = segment(&vec![0.0; 44_100], 44_100.0);
        assert!(subs.iter().all(|s| !s.is_reasonable(44_100.0)));
    }
}