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
//! Turning a MIDI keyboard take into Sequencer rows.
//!
//! **What is recorded is a time, what a row holds is a length.** The keyboard
//! gives note-on/note-off instants; a row is a chain of frames where a note's
//! position is nothing but the sum of the lengths before it. Converting between
//! the two is this module, and it happens once, when the take ends — nothing is
//! written into the sequence while the user is still playing.
//!
//! **Auto-rounding.** Nobody plays on the sample. Every start and every length
//! is snapped to the nearest multiple of a chosen note value ([`ROUND_CHOICES`],
//! 1/16 by default), so a take comes out as notes a person would have written
//! rather than as 137 ms and 249 ms. Starts and lengths are snapped
//! *independently of each other* and both against the take's own zero, so a
//! rounding never accumulates: the tenth note lands on the grid position it was
//! played nearest to, not ten roundings away from it.
//!
//! **A gap is exact; a note length is rounded twice.** A take is written in
//! plain lengths — whole notes plus one fraction, no nominator — which cannot
//! express, say, 3/8 in a single frame. Note lengths therefore take the nearest
//! value a plain length *can* express. (The frames themselves do count a
//! fraction, so what is written here can be turned into a 3/8 by hand; teaching
//! the rounding to search nominators is a change to this file, not to what a
//! frame can hold.) Silence
//! does not have to: a space that is not expressible is split across the space
//! frame and one or more silent placeholder frames behind it ([`split_units`]),
//! which the model already allows and which costs nothing but a frame on screen.
//! That is the trade deliberately: **a note's sounding length may be off by up
//! to half a grid step, but no note's position ever is.**
//!
//! **Chords become rows, a late release does not.** A row is one sequence — it
//! cannot hold two notes at once — so notes that genuinely sound together are
//! split across as many rows as the deepest chord in the take
//! ([`split_voices`]), all playing the same instrument. Notes that merely run into
//! each other are *not*: letting go of one key a moment after pressing the next
//! is how anyone plays a line, and a row each would turn a bass line into a pile
//! of one-note rows. The earlier note is shortened to where the later one starts
//! instead.

use std::collections::HashMap;
use std::time::Instant;

use crate::midi::input::TimedMessage;

use super::{Duration, Fraction, Item, MAX_WHOLES, UNITS_PER_WHOLE};

/// The note values a take can be rounded onto, coarsest first, as
/// `(label, units)`. 1/256 is the grid itself, i.e. no rounding worth the name;
/// it is offered so a take can be kept as played.
pub const ROUND_CHOICES: [(&str, i64); 9] = [
    ("1", UNITS_PER_WHOLE),
    ("1/2", UNITS_PER_WHOLE / 2),
    ("1/4", UNITS_PER_WHOLE / 4),
    ("1/8", UNITS_PER_WHOLE / 8),
    ("1/16", UNITS_PER_WHOLE / 16),
    ("1/32", UNITS_PER_WHOLE / 32),
    ("1/64", UNITS_PER_WHOLE / 64),
    ("1/128", UNITS_PER_WHOLE / 128),
    ("1/256", 1),
];

/// What a take is rounded onto unless the user says otherwise. A 1/16 is fine
/// enough to keep a played rhythm recognisable and coarse enough to tidy it.
pub const DEFAULT_ROUND_UNITS: i64 = UNITS_PER_WHOLE / 16;

/// How a rounding note value reads, for the status line and the select box.
pub fn round_label(units: i64) -> &'static str {
    ROUND_CHOICES
        .iter()
        .find(|(_, u)| *u == units)
        .map_or("1/16", |(label, _)| label)
}

/// A length of zero — a placeholder frame's note, or the space after the last
/// note of a take.
const ZERO: Duration = Duration::new(0, Fraction::None);

/// Ceiling on the frames one silence may be split into. A gap needing more than
/// this is minutes long; the cap is only here so a nonsense timestamp cannot
/// make the loop build frames for as long as it is allowed to.
const MAX_SPLIT_TERMS: usize = 64;

/// A note as it was played: seconds from the start of the take, before any
/// rounding.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RecordedNote {
    pub at_secs: f64,
    pub dur_secs: f64,
    pub pitch: u8,
}

/// A note rounded onto the grid, in units from the start of the take.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GridNote {
    pub at: i64,
    pub dur: i64,
    pub pitch: u8,
}

impl GridNote {
    fn end(self) -> i64 {
        self.at + self.dur
    }
}

/// The notes in a captured stream of MIDI messages.
///
/// `clock` places a message on the sequence's timeline; a message it cannot
/// place (it arrived before the transport made a sound) is dropped, and one it
/// places before zero is pulled up to it — a key hit a hair early is a note
/// played at the top of the take, not a note at a negative time.
///
/// Notes still held when the take ended are closed at `end_secs`: a recording
/// that ends with a key down still has that note in it.
pub fn notes_from(
    messages: &[TimedMessage],
    mut clock: impl FnMut(Instant) -> Option<f64>,
    end_secs: f64,
) -> Vec<RecordedNote> {
    let mut open: HashMap<u8, f64> = HashMap::new();
    let mut out: Vec<RecordedNote> = Vec::new();
    let close = |out: &mut Vec<RecordedNote>, pitch: u8, from: f64, to: f64| {
        out.push(RecordedNote {
            at_secs: from,
            dur_secs: (to - from).max(0.0),
            pitch,
        });
    };

    for (stamp, msg) in messages {
        let Some(at) = clock(*stamp) else { continue };
        let at = at.max(0.0);
        let (status, pitch, velocity) = (msg[0] & 0xF0, msg[1], msg[2]);
        match status {
            // A note-on for a pitch already sounding ends the one before it:
            // some keyboards re-strike without releasing, and two starts with
            // one end would otherwise leave a note running to the end of the
            // take.
            0x90 if velocity > 0 => {
                if let Some(from) = open.insert(pitch, at) {
                    close(&mut out, pitch, from, at);
                }
            }
            // Note-off, in both spellings: 0x80, and the 0x90-with-zero-velocity
            // that most keyboards actually send.
            0x80 | 0x90 => {
                if let Some(from) = open.remove(&pitch) {
                    close(&mut out, pitch, from, at);
                }
            }
            _ => {}
        }
    }
    for (pitch, from) in open {
        close(&mut out, pitch, from, end_secs.max(from));
    }
    out.sort_by(|a, b| {
        a.at_secs
            .partial_cmp(&b.at_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.pitch.cmp(&b.pitch))
    });
    out
}

/// Round a take onto the grid: every start and every length to the nearest
/// multiple of `grid` units, at `spu` seconds per unit.
///
/// A note is never rounded away to nothing. A hit shorter than half a grid step
/// is still a note that was played, so it comes out one step long rather than
/// silent.
pub fn quantize(notes: &[RecordedNote], spu: f64, grid: i64) -> Vec<GridNote> {
    let grid = grid.max(1);
    let spu = spu.max(f64::MIN_POSITIVE);
    let mut out: Vec<GridNote> = notes
        .iter()
        .map(|n| GridNote {
            at: snap(n.at_secs / spu, grid).max(0),
            dur: snap(n.dur_secs / spu, grid).max(grid),
            pitch: n.pitch,
        })
        .collect();
    out.sort_by_key(|n| (n.at, n.pitch));
    out
}

/// The nearest multiple of `grid` to a length in (fractional) units.
fn snap(units: f64, grid: i64) -> i64 {
    if !units.is_finite() {
        return 0;
    }
    ((units / grid as f64).round() as i64).saturating_mul(grid)
}

/// Split a take into as many rows as it needs: a note joins the first row that
/// can still take it ([`shares_a_row`]), and starts a new one when none can.
///
/// A row plays one note at a time, so this is what makes a chord playable at
/// all. First-fit rather than round-robin keeps the count down to the deepest
/// chord actually played, and puts the melody of a single-note passage in the
/// first row instead of scattering it.
///
/// `notes` must be sorted by start, as [`quantize`] leaves them; each row comes
/// out sorted too.
pub fn split_voices(notes: &[GridNote]) -> Vec<Vec<GridNote>> {
    let mut voices: Vec<Vec<GridNote>> = Vec::new();
    for note in notes {
        match voices
            .iter_mut()
            .find(|v| v.last().is_none_or(|last| shares_a_row(*last, *note)))
        {
            Some(voice) => voice.push(*note),
            None => voices.push(vec![*note]),
        }
    }
    voices
}

/// Whether `next` can follow `last` in one row — either it starts clear of it,
/// or it only *runs into* it and the earlier note can be shortened to make room.
///
/// Only notes that sound **together** need a row of their own: struck at the
/// same moment, or the earlier one lasting through the whole of the later. Two
/// notes that merely overlap at the edges do not. Releasing a key a moment after
/// pressing the next is how a line is played — more so on a small keyboard, one
/// hand, reaching — and a row for every one of those would leave a bass line as
/// a stack of rows with a note each, which is not something anyone would have
/// written by hand.
///
/// The earlier note is not left overlapping: [`voice_items`] fits every note
/// into the room it has before the next one starts, so keeping the pair in one
/// row *is* the trim.
fn shares_a_row(last: GridNote, next: GridNote) -> bool {
    if last.end() <= next.at {
        return true;
    }
    // Struck together, or swallowed whole — both are sounding for as long as the
    // shorter of them lasts, which one row cannot do.
    next.at != last.at && last.end() < next.end()
}

/// The length a *plain* frame expresses exactly, if it can: whole notes plus one
/// fraction, the nominator left at one. `3/8` has no answer here; `1 + 1/8` does.
///
/// A frame's nominator would give `3/8` one — the take simply does not use it,
/// so that what it writes is split and rounded by the one rule described at the
/// top of this file.
pub fn duration_from_units(units: i64) -> Option<Duration> {
    if units < 0 {
        return None;
    }
    let wholes = units / UNITS_PER_WHOLE;
    if wholes > MAX_WHOLES as i64 {
        return None;
    }
    let rest = units - wholes * UNITS_PER_WHOLE;
    let frac = Fraction::ALL.into_iter().find(|f| f.units() == rest)?;
    Some(Duration::new(wholes as u16, frac))
}

/// The plain length closest to `units` — what a *note* has to settle for,
/// because a note that sounds cannot be split into two frames without being
/// struck twice. Ties go to the shorter, so a note never overruns the one after
/// it for the sake of a rounding.
pub fn nearest_duration(units: i64) -> Duration {
    let mut best = ZERO;
    let mut best_diff = i64::MAX;
    for wholes in 0..=MAX_WHOLES {
        for frac in Fraction::ALL {
            let candidate = Duration::new(wholes, frac);
            let diff = (candidate.units() - units).abs();
            // Strictly closer, or equally close and shorter: a 3/8 note is as
            // near 1/4 as it is 1/2, and the short one is the one that cannot
            // run into the note after it.
            if diff < best_diff || (diff == best_diff && candidate.units() < best.units()) {
                best = candidate;
                best_diff = diff;
            }
        }
    }
    best
}

/// The longest plain length that still fits in `units`.
pub fn longest_within(units: i64) -> Duration {
    split_units(units).first().copied().unwrap_or(ZERO)
}

/// A silence as a chain of plain lengths adding up to it **exactly**,
/// longest first. Empty for zero.
///
/// This is why a recorded position never drifts. A gap of 3/8 cannot be one
/// length, but it can be 1/4 then 1/8 — the first on the note's own space frame,
/// the rest on silent placeholder frames — and the note after it lands exactly
/// where it was played.
pub fn split_units(mut units: i64) -> Vec<Duration> {
    let mut out = Vec::new();
    while units > 0 && out.len() < MAX_SPLIT_TERMS {
        let wholes = (units / UNITS_PER_WHOLE).min(MAX_WHOLES as i64);
        let rest = units - wholes * UNITS_PER_WHOLE;
        let frac = Fraction::ALL
            .into_iter()
            .filter(|f| f.units() > 0 && f.units() <= rest)
            .max_by_key(|f| f.units())
            .unwrap_or(Fraction::None);
        let term = Duration::new(wholes as u16, frac);
        // Every term takes at least a 1/256 (the grid's own step), so this
        // cannot spin: the smallest positive `units` still yields one.
        if term.units() == 0 {
            break;
        }
        units -= term.units();
        out.push(term);
    }
    out
}

/// One row's worth of a take as the Sequencer holds it: the lead space, then the
/// frames.
///
/// Every note is followed by the silence up to the next one, exactly. Where that
/// silence is not one expressible length it continues on placeholder frames —
/// a note frame of length zero, which the model already treats as an ordinary
/// unplayed frame, carrying the rest of the space.
pub fn voice_items(notes: &[GridNote], next_item_id: &mut u64) -> (Duration, Vec<Item>) {
    let mut items: Vec<Item> = Vec::new();
    let Some(first) = notes.first() else {
        return (ZERO, items);
    };

    // The row's lead space takes as much of the run-up as one length can hold;
    // whatever is left leads the first note as placeholder frames.
    let mut lead_terms = split_units(first.at);
    let lead = if lead_terms.is_empty() {
        ZERO
    } else {
        lead_terms.remove(0)
    };
    for term in lead_terms {
        items.push(filler(next_item_id, first.pitch, term));
    }


    for (i, note) in notes.iter().enumerate() {
        // How long until the next note in this row starts — the room this note
        // and the silence behind it have to share.
        let room = notes.get(i + 1).map(|next| next.at - note.at);
        let mut dur = nearest_duration(note.dur);
        if let Some(room) = room {
            if dur.units() > room {
                dur = longest_within(room);
            }
        }
        let gap = room.map_or(0, |room| (room - dur.units()).max(0));
        let mut gap_terms = split_units(gap);
        let space = if gap_terms.is_empty() {
            ZERO
        } else {
            gap_terms.remove(0)
        };
        items.push(Item {
            id: take_id(next_item_id),
            pitch: note.pitch,
            // A take is played on a keyboard, and a key has no file to start
            // into. A row moved onto a wav instrument afterwards starts at the top of
            // it, which is where a note that was never given a start belongs.
            start: 0.0,
            dur,
            space,
        });
        for term in gap_terms {
            items.push(filler(next_item_id, note.pitch, term));
        }
    }

    (lead, items)
}

/// The next item id in a row, consumed.
fn take_id(next_item_id: &mut u64) -> u64 {
    let id = *next_item_id;
    *next_item_id += 1;
    id
}

/// A frame that plays nothing and only carries silence. Legal by construction:
/// a zero-length note is a placeholder the model already draws and skips.
fn filler(next_item_id: &mut u64, pitch: u8, space: Duration) -> Item {
    Item {
        id: take_id(next_item_id),
        pitch,
        start: 0.0,
        dur: ZERO,
        space,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration as StdDuration;

    use crate::gui::sequencer::Row;

    const WHOLE: i64 = UNITS_PER_WHOLE;
    const HALF: i64 = WHOLE / 2;
    const QUARTER: i64 = WHOLE / 4;
    const EIGHTH: i64 = WHOLE / 8;
    const SIXTEENTH: i64 = WHOLE / 16;

    /// A take at 120 BPM: a quarter note is half a second, so one unit is
    /// 1/512 s.
    const SPU: f64 = 0.5 / QUARTER as f64;

    fn note(at_secs: f64, dur_secs: f64, pitch: u8) -> RecordedNote {
        RecordedNote {
            at_secs,
            dur_secs,
            pitch,
        }
    }

    fn grid_note(at: i64, dur: i64, pitch: u8) -> GridNote {
        GridNote { at, dur, pitch }
    }

    /// A row built from one voice of a take, as the panel builds it.
    fn row_from(voice: &[GridNote]) -> Row {
        let mut row = Row::new(0, None);
        let (lead, items) = voice_items(voice, &mut row.next_item_id);
        row.lead = lead;
        row.items = items;
        row
    }

    /// The two select boxes cannot express every length — 3/8 is the first one
    /// a player hits — and pretending otherwise is what would make a recorded
    /// silence wrong.
    #[test]
    fn some_lengths_are_not_one_length() {
        assert!(duration_from_units(QUARTER + EIGHTH).is_none());
        assert_eq!(
            duration_from_units(WHOLE + EIGHTH).map(Duration::units),
            Some(WHOLE + EIGHTH)
        );
        // Past the whole-note ceiling there is nothing to express it with.
        assert!(duration_from_units(WHOLE * (MAX_WHOLES as i64 + 1)).is_none());
    }

    /// A silence is split into frames that add up to it **exactly**, so nothing
    /// after it moves. This is the whole reason placeholder frames exist.
    #[test]
    fn a_silence_is_split_exactly() {
        for units in [1, EIGHTH, QUARTER + EIGHTH, WHOLE * 3 + SIXTEENTH, 4321] {
            let terms = split_units(units);
            assert_eq!(
                terms.iter().map(|d| d.units()).sum::<i64>(),
                units,
                "{units} units did not add up"
            );
            assert!(terms.iter().all(|d| d.units() > 0));
        }
        assert!(split_units(0).is_empty());
        // Longest first, and no more terms than the length needs.
        assert_eq!(
            split_units(QUARTER + EIGHTH)
                .iter()
                .map(|d| d.units())
                .collect::<Vec<_>>(),
            vec![QUARTER, EIGHTH]
        );
    }

    /// A note's length is the one thing that has to be approximated — it sounds,
    /// so it cannot be split across frames. A tie goes to the shorter, which is
    /// the one that cannot overrun the note after it.
    #[test]
    fn a_note_length_takes_the_nearest_and_ties_go_short() {
        assert_eq!(nearest_duration(QUARTER).units(), QUARTER);
        assert_eq!(nearest_duration(QUARTER + EIGHTH).units(), QUARTER);
        assert_eq!(nearest_duration(QUARTER + EIGHTH + 1).units(), HALF);
        // Longer than anything the boxes can say: the longest they can.
        assert_eq!(
            nearest_duration(WHOLE * 100).units(),
            WHOLE * MAX_WHOLES as i64 + HALF
        );
    }

    /// Rounding is against the take's own zero, not against the note before, so
    /// a run of sloppy notes cannot accumulate half a step per note into a drift.
    #[test]
    fn rounding_does_not_accumulate() {
        // Four "quarter notes" at 120 BPM, none of them on time: 30 ms late,
        // 30 ms early, 20 ms late. Rounded against the note before instead of
        // against the take's zero, those errors would chain.
        let played: Vec<RecordedNote> = [0.0, 0.53, 0.97, 1.52]
            .iter()
            .map(|at| note(*at, 0.45, 60))
            .collect();
        let quantized = quantize(&played, SPU, SIXTEENTH);
        assert_eq!(
            quantized.iter().map(|n| n.at).collect::<Vec<_>>(),
            vec![0, QUARTER, QUARTER * 2, QUARTER * 3]
        );
    }

    /// A hit shorter than half a grid step is still a note that was played.
    #[test]
    fn a_short_hit_is_not_rounded_away() {
        let quantized = quantize(&[note(0.0, 0.005, 36)], SPU, SIXTEENTH);
        assert_eq!(quantized, vec![grid_note(0, SIXTEENTH, 36)]);
    }

    /// What a take is *for*: the frames a row ends up with put every note back
    /// at the position it was rounded to — exactly, however the silences had to
    /// be split to get there.
    #[test]
    fn a_recorded_row_puts_every_note_where_it_was_played() {
        // Starts at 0, 3/8 and 1 + 1/16 — the middle one is a position no
        // single space frame can express.
        let voice = [
            grid_note(0, EIGHTH, 60),
            grid_note(QUARTER + EIGHTH, SIXTEENTH, 64),
            grid_note(WHOLE + SIXTEENTH, QUARTER, 67),
        ];
        let row = row_from(&voice);
        let played: Vec<(i64, u8)> = row
            .planned_notes(1.0)
            .iter()
            .map(|n| (n.at_secs as i64, n.pitch))
            .collect();
        assert_eq!(
            played,
            vec![(0, 60), (QUARTER + EIGHTH, 64), (WHOLE + SIXTEENTH, 67)]
        );
        // The middle position needed a placeholder frame to reach it, so the
        // row holds more frames than it does notes.
        assert_eq!(row.items.len(), 4);
    }

    /// A note is never given more room than there is: rounding its length up
    /// must not make it run into the note after it.
    #[test]
    fn a_rounded_note_never_overruns_the_next_one() {
        // Played nearly a quarter long, but the next note is an eighth away.
        let voice = [
            grid_note(0, QUARTER, 60),
            grid_note(EIGHTH, EIGHTH, 62),
        ];
        let row = row_from(&voice);
        let notes = row.planned_notes(1.0);
        assert_eq!(notes[0].dur_secs as i64, EIGHTH);
        assert_eq!(notes[1].at_secs as i64, EIGHTH);
    }

    /// A row plays one note at a time, so notes that sound together have to
    /// become rows. A melody must not: first-fit keeps a single-note passage in
    /// one row.
    #[test]
    fn a_chord_becomes_a_row_each_and_a_melody_does_not() {
        let chord = [
            grid_note(0, QUARTER, 60),
            grid_note(0, QUARTER, 64),
            grid_note(0, QUARTER, 67),
        ];
        let voices = split_voices(&chord);
        assert_eq!(voices.len(), 3);
        assert_eq!(voices[0][0].pitch, 60);

        let melody = [
            grid_note(0, QUARTER, 60),
            grid_note(QUARTER, QUARTER, 62),
            grid_note(HALF, QUARTER, 64),
        ];
        assert_eq!(split_voices(&melody).len(), 1);

        // A note held under a whole line — it lasts through the notes that
        // follow, so it really is a second voice.
        let held = [
            grid_note(0, WHOLE, 36),
            grid_note(QUARTER, QUARTER, 60),
            grid_note(HALF, QUARTER, 62),
        ];
        let voices = split_voices(&held);
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0].len(), 1);
        assert_eq!(voices[1].len(), 2);
    }

    /// Letting go of a key a moment after pressing the next is a line, not a
    /// chord: it stays in one row, and the note that was held too long is cut
    /// back to where the next one starts.
    #[test]
    fn a_late_release_stays_in_one_row_and_is_trimmed() {
        // Three quarter notes, each held a sixteenth into the one after it.
        let sloppy = [
            grid_note(0, QUARTER + SIXTEENTH, 40),
            grid_note(QUARTER, QUARTER + SIXTEENTH, 43),
            grid_note(HALF, QUARTER, 45),
        ];
        let voices = split_voices(&sloppy);
        assert_eq!(voices.len(), 1, "a late release must not open a row");

        let row = row_from(&voices[0]);
        let played = row.planned_notes(1.0);
        assert_eq!(
            played
                .iter()
                .map(|n| (n.at_secs as i64, n.dur_secs as i64))
                .collect::<Vec<_>>(),
            vec![(0, QUARTER), (QUARTER, QUARTER), (HALF, QUARTER)]
        );
    }

    /// The keyboard's own timing is what a take is built from: note-off in both
    /// spellings, a re-struck key closing the note before it, and a key still
    /// held when the take ended closed at the end rather than dropped.
    #[test]
    fn a_take_is_read_off_the_keyboard_the_way_keyboards_send_it() {
        let base = Instant::now();
        let at = |ms: u64| base + StdDuration::from_millis(ms);
        let messages = vec![
            (at(0), [0x90, 60, 100]),
            // A note-off spelled as a note-on with no velocity, which is what
            // most keyboards actually send.
            (at(500), [0x90, 60, 0]),
            (at(500), [0x90, 64, 100]),
            // Struck again without a release: the first one has to end here.
            (at(1000), [0x90, 64, 100]),
            (at(1500), [0x80, 64, 0]),
            // Still held when the take ends.
            (at(1500), [0x90, 67, 100]),
        ];
        let notes = notes_from(
            &messages,
            |t| Some(t.duration_since(base).as_secs_f64()),
            2.0,
        );
        assert_eq!(
            notes,
            vec![
                note(0.0, 0.5, 60),
                note(0.5, 0.5, 64),
                note(1.0, 0.5, 64),
                note(1.5, 0.5, 67),
            ]
        );
    }

    /// A message the transport cannot place is one played before it made a
    /// sound; a key hit a hair early belongs at the top of the take, not at a
    /// negative time — held there, so what is trimmed is its front, not its end.
    #[test]
    fn unplaceable_messages_are_dropped_and_early_ones_pulled_to_zero() {
        let base = Instant::now();
        let at = |ms: u64| base + StdDuration::from_millis(ms);
        let messages = vec![
            (at(0), [0x90, 60, 100]),
            (at(100), [0x80, 60, 0]),
            (at(200), [0x90, 62, 100]),
            (at(700), [0x80, 62, 0]),
        ];
        // No clock at all until 150 ms in, and 200 ms of latency behind it.
        let notes = notes_from(
            &messages,
            |t| {
                let secs = t.duration_since(base).as_secs_f64();
                (secs >= 0.15).then_some(secs - 0.35)
            },
            1.0,
        );
        // The 60 was played before there was a clock and is gone; the 62 was
        // struck 150 ms before zero and starts at zero, ending where it ended.
        assert_eq!(notes, vec![note(0.0, 0.35, 62)]);
    }
}
