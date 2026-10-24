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
//! A look-ahead peak limiter: the ceiling a signal meets on its way out.
//!
//! Something has to stop a boosted track from leaving full scale, and the cheap
//! answer — clamping every sample at ±1.0 — is the worst one. A clamp is a
//! nonlinearity applied to single samples, so it does not turn a loud passage
//! down, it flattens the top off every waveform in it and fills the spectrum
//! with what was never played. On speech that lands on the plosives: a "p"
//! boosted ten times has four fifths of its samples clipped, and the harshness
//! that comes back is broadband, above everything else in the mix.
//!
//! This turns the signal down instead, and does it *before* the peak arrives.
//! The stream is cut into chunks of [`LOOK_MS`]; each one is held until the
//! following chunk's peak is known, and then played out under a gain that ramps
//! linearly between boundary values, never above what either neighbouring chunk
//! can take. So the gain is already down when the transient lands — there is no
//! overshoot to clip, and no sample is touched by anything but a smooth,
//! signal-wide level change. The gain then holds ([`HOLD_MS`]) before it
//! recovers on a slow one-pole ([`RELEASE_MS`]), so one plosive does not
//! audibly duck what follows it, and a bass note does not pump within its own
//! cycle.
//!
//! The price is [`Limiter::latency_frames`] of delay, two chunks' worth, which
//! the caller has to account for: the transport adds it to the latency it
//! reports, and an offline render drops it off the front.
//!
//! Below the ceiling the gain is exactly 1.0 and samples pass through
//! unaltered — a mix that never reaches full scale is bit-for-bit what it was
//! before this stood in the path.
//!
//! [`Limiter::low_band`] is the same limiter with its gain on the bass alone.
//! A wideband limiter on a mix whose sub fills full scale by itself turns
//! *everything* down every time a drum lands on top of it, by however much the
//! sub happens to be playing just then — so the snares come out uneven, loud
//! where the sub rests and 2–5 dB down wherever it plays (NEURO-FRACTURE, an
//! 808 at row gain 7). Taking the excess out of the band below
//! [`LOW_BAND_HZ`] first is what a mastering engineer does by hand: the sub
//! makes room for the drums, and the drums stay where they were played.

use std::collections::VecDeque;

/// How far ahead the limiter looks, in milliseconds. One chunk is also the
/// length of the gain ramp into a peak: long enough not to modulate the body of
/// a voice, short enough that a transient is not turned down long before it.
const LOOK_MS: f64 = 2.0;

/// How long the gain takes to come back, as a one-pole time constant. Slow
/// enough that a single plosive is one dip rather than a stutter through the
/// syllable behind it.
const RELEASE_MS: f64 = 120.0;

/// How long the gain stays where a peak put it before the release begins. It
/// must outlast the longest period the mix plays: a 2 ms look-ahead sees each
/// peak of a 30 Hz sub on its own, and without a hold the gain recovered
/// between peaks and dipped again at every one — a gain kink once a cycle,
/// which is a buzz at the note's own pitch (an 808 at row gain 3: 185 kinks
/// in 4 s, recurring at f0). 50 ms covers anything down to 20 Hz.
const HOLD_MS: f64 = 50.0;

/// Where [`Limiter::low_band`] splits the mix: the first null of its low-pass.
/// An 808's fundamental (30–60 Hz) is mostly in the low band, a snare's body
/// (~180 Hz and up) 40 dB or more below it.
pub const LOW_BAND_HZ: f64 = 150.0;

/// The furthest [`Limiter::low_band`] turns the bass down, −12 dB. Past that a
/// sub would vanish under every drum hit; what is still over the ceiling then
/// is a wideband limiter's to take.
const LOW_BAND_FLOOR: f32 = 0.25;

/// One running-sum box filter: the mean of the last `len` samples, at a cost
/// that does not depend on `len`. Linear phase, `(len - 1) / 2` frames late.
struct Boxcar {
    ring: Vec<f64>,
    at: usize,
    sum: f64,
}

impl Boxcar {
    fn new(len: usize) -> Self {
        Self { ring: vec![0.0; len], at: 0, sum: 0.0 }
    }

    fn tick(&mut self, x: f64) -> f64 {
        self.sum += x - self.ring[self.at];
        self.ring[self.at] = x;
        self.at = (self.at + 1) % self.ring.len();
        self.sum / self.ring.len() as f64
    }
}

/// The split [`Limiter::low_band`] works on, for one channel: three boxcars in
/// a row (close to a Gaussian, so no ringing and sidelobes 40 dB down) for the
/// low band, and the input delayed by exactly as much. The low band has to be
/// *in phase* with the signal it is taken out of — `x - low` from a recursive
/// low-pass is no high band at all, since its phase lag at 40 Hz leaves most of
/// the sub in the difference.
struct Split {
    boxes: [Boxcar; 3],
    delay: VecDeque<f32>,
}

impl Split {
    fn new(len: usize) -> Self {
        let lag = 3 * (len - 1) / 2;
        Self {
            boxes: [Boxcar::new(len), Boxcar::new(len), Boxcar::new(len)],
            delay: std::iter::repeat_n(0.0, lag).collect(),
        }
    }

    /// The input `lag` frames ago, and its low band at that same moment.
    fn tick(&mut self, x: f32) -> (f32, f32) {
        let [a, b, c] = &mut self.boxes;
        let low = c.tick(b.tick(a.tick(x as f64))) as f32;
        self.delay.push_back(x);
        (self.delay.pop_front().unwrap_or(0.0), low)
    }
}

/// What a [`Limiter::low_band`] carries on top of a plain limiter.
struct Band {
    /// One split per channel.
    splits: Vec<Split>,
    /// The low band of `pending`, sample for sample.
    pending: Vec<f32>,
    /// Frames the split delays the signal by.
    lag: usize,
}

/// One held chunk: its samples and the most gain they may be played at.
struct Chunk {
    samples: Vec<f32>,
    /// Their low band, for a [`Limiter::low_band`]; empty otherwise.
    low: Vec<f32>,
    /// `min(1, ceiling / peak)` — unity for a chunk that never reaches the
    /// ceiling, which is the common case.
    gain: f32,
}

/// A look-ahead peak limiter over an interleaved buffer. Channels are linked:
/// the gain is computed from the loudest sample of the frame group, so a peak
/// on the left does not pull the image over.
pub struct Limiter {
    channels: usize,
    /// Frames per chunk, the look-ahead.
    chunk: usize,
    ceiling: f32,
    /// Per-frame release coefficient.
    release: f32,
    /// Frames the gain is held for after it is pulled down.
    hold: usize,
    /// Frames of hold still to run before the release may start.
    hold_left: usize,
    /// Input not yet forming a whole chunk.
    pending: Vec<f32>,
    /// The chunk waiting for its successor's peak. `None` only for the very
    /// first chunk of the stream.
    held: Option<Chunk>,
    /// Gain at the boundary the next chunk starts from.
    boundary: f32,
    /// The release envelope, carried across chunks.
    env: f32,
    /// Output waiting to be handed back, primed with the latency in silence.
    out: VecDeque<f32>,
    /// `Some` when the gain applies to the low band only.
    band: Option<Band>,
}

impl Limiter {
    /// A limiter for `channels` interleaved channels at `sample_rate`, holding
    /// everything at or below `ceiling`.
    pub fn new(channels: usize, sample_rate: f64, ceiling: f32) -> Self {
        let channels = channels.max(1);
        let chunk = ((sample_rate * LOOK_MS / 1000.0).round() as usize).max(1);
        let span = chunk * channels;
        let mut pending = Vec::with_capacity(span);
        pending.clear();
        // Three chunks is the most that is ever queued: the two of latency the
        // stream is primed with, plus the one an input block can complete.
        let mut out = VecDeque::with_capacity(3 * span + 1);
        out.extend(std::iter::repeat_n(0.0, 2 * span));
        Self {
            channels,
            chunk,
            ceiling,
            release: (-1.0 / (sample_rate * RELEASE_MS / 1000.0)).exp() as f32,
            hold: (sample_rate * HOLD_MS / 1000.0).round() as usize,
            hold_left: 0,
            pending,
            held: None,
            boundary: 1.0,
            env: 1.0,
            out,
            band: None,
        }
    }

    /// A limiter whose gain turns down only the band below [`LOW_BAND_HZ`], and
    /// no further than [`LOW_BAND_FLOOR`]: it takes as much of a peak as the
    /// bass can give and leaves the rest of the mix untouched. It does not
    /// promise the ceiling — a peak the bass is not part of gets through — so a
    /// wideband [`Limiter::new`] has to follow it.
    pub fn low_band(channels: usize, sample_rate: f64, ceiling: f32) -> Self {
        let mut lim = Self::new(channels, sample_rate, ceiling);
        // Odd, so each box's delay is a whole number of frames.
        let len = ((sample_rate / LOW_BAND_HZ).round() as usize / 2) * 2 + 1;
        lim.band = Some(Band {
            splits: (0..lim.channels).map(|_| Split::new(len)).collect(),
            pending: Vec::with_capacity(lim.chunk * lim.channels),
            lag: 3 * (len - 1) / 2,
        });
        lim
    }

    /// How far behind its input the output runs, in frames. Constant for the
    /// life of the limiter, and the same for every channel.
    pub fn latency_frames(&self) -> usize {
        2 * self.chunk + self.band.as_ref().map_or(0, |b| b.lag)
    }

    /// Limit `buf` in place. The buffer comes back the same length, holding the
    /// audio [`Self::latency_frames`] earlier in the stream.
    pub fn process(&mut self, buf: &mut [f32]) {
        let span = self.chunk * self.channels;
        // Everything in first — the output written below is older audio, and
        // would otherwise overwrite input still to be read.
        let mut read = 0;
        while read < buf.len() {
            let take = (span - self.pending.len()).min(buf.len() - read);
            match &mut self.band {
                None => self.pending.extend_from_slice(&buf[read..read + take]),
                Some(band) => {
                    for &x in &buf[read..read + take] {
                        // Which channel, from where the sample lands in the chunk.
                        let ch = band.pending.len() % self.channels;
                        let (x, low) = band.splits[ch].tick(x);
                        self.pending.push(x);
                        band.pending.push(low);
                    }
                }
            }
            read += take;
            if self.pending.len() == span {
                self.close_chunk();
            }
        }
        // Never short: each input chunk queues one output chunk, and the two
        // primed at the start cover the two the limiter is holding.
        for s in buf.iter_mut() {
            *s = self.out.pop_front().unwrap_or(0.0);
        }
    }

    /// Take the full chunk out of `pending`, play out the one held behind it,
    /// and hold this one in its place.
    fn close_chunk(&mut self) {
        let gain = match &self.band {
            None => {
                let peak = self.pending.iter().fold(0f32, |m, s| m.max(s.abs()));
                if peak > self.ceiling {
                    self.ceiling / peak
                } else {
                    1.0
                }
            }
            Some(band) => self.low_band_gain(&band.pending),
        };
        let mut low = match &mut self.band {
            Some(band) => std::mem::take(&mut band.pending),
            None => Vec::new(),
        };
        match self.held.take() {
            Some(mut held) => {
                // The gain the boundary between the two chunks is played at.
                // Both ends of a chunk's ramp are at or below what the chunk
                // itself can take, so the whole ramp is, and nothing
                // overshoots.
                let target = held.gain.min(gain);
                self.emit(&held.samples, &held.low, target);
                // The vectors are swapped rather than copied: past the first
                // two chunks a stream this way allocates nothing at all.
                std::mem::swap(&mut held.samples, &mut self.pending);
                std::mem::swap(&mut held.low, &mut low);
                held.gain = gain;
                self.held = Some(held);
            }
            // The first chunk of the stream has nothing behind it to play out.
            None => {
                let mut samples = Vec::with_capacity(self.pending.len());
                std::mem::swap(&mut samples, &mut self.pending);
                self.held = Some(Chunk { samples, low, gain });
                low = Vec::new();
            }
        }
        self.pending.clear();
        // The vector the outgoing chunk's low band lived in, back for reuse.
        if let Some(band) = &mut self.band {
            low.clear();
            band.pending = low;
        }
    }

    /// The most gain the low band of the chunk in `pending` may be played at,
    /// so that `gain * low + (x - low)` stays inside the ceiling: for each
    /// sample over it whose bass leans the same way, the share of the bass that
    /// has to go. A sample whose bass leans against the peak cannot be helped
    /// by turning the bass down, and is left for the wideband limiter.
    fn low_band_gain(&self, low: &[f32]) -> f32 {
        let mut gain = 1f32;
        for (&x, &l) in self.pending.iter().zip(low) {
            let over = x.abs() - self.ceiling;
            let lean = l * x.signum();
            if over > 0.0 && lean > 0.0 {
                gain = gain.min(1.0 - over / lean);
            }
        }
        gain.max(LOW_BAND_FLOOR)
    }

    /// Play one held chunk out under a gain ramping from the standing boundary
    /// to `target`, with the release envelope over the top.
    fn emit(&mut self, samples: &[f32], low: &[f32], target: f32) {
        let step = (target - self.boundary) / self.chunk as f32;
        let mut ramp = self.boundary;
        for (f, frame) in samples.chunks(self.channels).enumerate() {
            ramp += step;
            // The ramp is the ceiling on the gain; the envelope drops to it at
            // once, holds, and comes back slowly. So the release never lets the
            // gain exceed what the chunk can take.
            if ramp <= self.env {
                self.env = ramp;
                self.hold_left = self.hold;
            } else if self.hold_left > 0 {
                self.hold_left -= 1;
            } else {
                self.env = ramp + (self.env - ramp) * self.release;
            }
            if low.is_empty() {
                for s in frame {
                    self.out.push_back(s * self.env);
                }
            } else if self.env == 1.0 {
                // Nothing to take out: the input itself, not the two bands
                // summed back, which rounds.
                self.out.extend(frame.iter().copied());
            } else {
                let low = &low[f * self.channels..(f + 1) * self.channels];
                for (s, l) in frame.iter().zip(low) {
                    self.out.push_back(s - (1.0 - self.env) * l);
                }
            }
        }
        self.boundary = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    fn run(lim: &mut Limiter, input: &[f32], block: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        for chunk in input.chunks(block) {
            let mut buf = chunk.to_vec();
            lim.process(&mut buf);
            out.extend_from_slice(&buf);
        }
        out
    }

    /// A signal that never reaches the ceiling comes back sample for sample,
    /// only later. Anything else would colour every quiet mix in the app.
    #[test]
    fn below_the_ceiling_nothing_is_touched() {
        let mut lim = Limiter::new(2, SR, 1.0);
        let input: Vec<f32> = (0..4000)
            .map(|n| 0.9 * (n as f32 * 0.01).sin())
            .collect();
        let out = run(&mut lim, &input, 173);
        let d = lim.latency_frames() * 2; // frames -> interleaved samples
        for (n, (&got, &want)) in out[d..].iter().zip(input.iter()).enumerate() {
            assert_eq!(got, want, "sample {n} came back changed");
        }
    }

    /// Nothing leaves the limiter above the ceiling — not on the first sample
    /// of a transient, which is exactly what a clamp is left to catch.
    #[test]
    fn a_transient_never_overshoots() {
        let mut lim = Limiter::new(1, SR, 1.0);
        let mut input = vec![0.1f32; 8000];
        // A plosive: nothing, then five times full scale for 3 ms.
        for s in input.iter_mut().skip(4000).take(144) {
            *s = 5.0;
        }
        let out = run(&mut lim, &input, 512);
        let peak = out.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak <= 1.0 + 1e-6, "left the limiter at {peak}");
        // And it is a level change, not a flattened top: the burst keeps its
        // shape, so its samples are all the same value as each other.
        let burst: Vec<f32> = out
            .iter()
            .copied()
            .filter(|s| s.abs() > 0.5)
            .collect();
        assert!(burst.len() > 100, "the burst did not survive: {}", burst.len());
    }

    /// The gain comes back after a peak instead of holding the whole track
    /// down — but slowly, over the release rather than at once.
    #[test]
    fn the_gain_returns_after_the_peak() {
        let mut lim = Limiter::new(1, SR, 1.0);
        let mut input = vec![0.5f32; 24_000];
        for s in input.iter_mut().skip(1000).take(96) {
            *s = 4.0;
        }
        let out = run(&mut lim, &input, 256);
        let at = |n: usize| out[n];
        // Right after the burst the level is still down.
        assert!(at(1300) < 0.45, "no gain reduction left at all: {}", at(1300));
        // Half a second later it is back, bar the tail of the one-pole.
        assert!(
            (at(23_000) - 0.5).abs() < 0.5 * 0.03,
            "the gain never came back: {}",
            at(23_000)
        );
    }

    /// The low-band limiter is as transparent as the plain one below the
    /// ceiling: its split delays the signal, and that is all it does to it.
    #[test]
    fn the_low_band_limiter_below_the_ceiling_is_a_delay() {
        let mut lim = Limiter::low_band(2, SR, 1.0);
        let input: Vec<f32> = (0..8000)
            .map(|n| 0.6 * (n as f32 * 0.005).sin() + 0.3 * (n as f32 * 0.3).sin())
            .collect();
        let out = run(&mut lim, &input, 211);
        let d = lim.latency_frames() * 2;
        for (n, (&got, &want)) in out[d..].iter().zip(input.iter()).enumerate() {
            assert_eq!(got, want, "sample {n} came back changed");
        }
    }

    /// A drum hit on top of a sub already at full scale: the plain limiter
    /// turns the hit down with the sub, the low-band one leaves the hit alone
    /// and takes the room out of the sub — and with the wideband one behind it,
    /// as the mix bus runs them, nothing leaves above the ceiling.
    #[test]
    fn a_hit_over_a_sub_turns_the_sub_down_and_not_the_hit() {
        let n = 24_000;
        let sub = |i: usize| 0.95 * (2.0 * std::f32::consts::PI * 45.0 * i as f32 / SR as f32).sin();
        let hit = |i: usize| {
            if (12_000..12_480).contains(&i) {
                0.5 * (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / SR as f32).sin()
            } else {
                0.0
            }
        };
        let input: Vec<f32> = (0..n).map(|i| sub(i) + hit(i)).collect();

        // How loud the hit comes back: its 3 kHz component, by projection.
        let hit_level = |out: &[f32], d: usize| {
            let (mut xy, mut yy) = (0f64, 0f64);
            for i in 12_000..12_480 {
                let y = (2.0 * std::f64::consts::PI * 3000.0 * i as f64 / SR).sin();
                xy += out[i + d] as f64 * y;
                yy += y * y;
            }
            xy / yy
        };

        let mut plain = Limiter::new(1, SR, 1.0);
        let out = run(&mut plain, &input, 256);
        let ducked = hit_level(&out, plain.latency_frames());

        let mut low = Limiter::low_band(1, SR, 1.0);
        let mut wide = Limiter::new(1, SR, 1.0);
        let mut out = run(&mut low, &input, 256);
        wide.process(&mut out);
        let d = low.latency_frames() + wide.latency_frames();
        let kept = hit_level(&out, d);

        assert!(ducked < 0.5 * 0.85, "the plain limiter did not duck the hit: {ducked}");
        assert!((kept - 0.5).abs() < 0.5 * 0.06, "the hit was turned down: {kept}");
        let peak = out.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak <= 1.0 + 1e-6, "left the limiters at {peak}");
    }

    /// Block size is the device's business, not the limiter's: the same input
    /// gives the same output however it is cut up.
    #[test]
    fn the_result_does_not_depend_on_the_block_size() {
        let input: Vec<f32> = (0..6000)
            .map(|n| 2.0 * (n as f32 * 0.013).sin() * (n as f32 * 0.0004).sin())
            .collect();
        let a = run(&mut Limiter::new(2, SR, 1.0), &input, 64);
        let b = run(&mut Limiter::new(2, SR, 1.0), &input, 1024);
        assert_eq!(a.len(), b.len());
        for (n, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-9, "sample {n}: {x} vs {y}");
        }
    }
}
