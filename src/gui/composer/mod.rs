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
//! Track Composer — arrange the registered Tracks in rows of frames and play them.
//!
//! **A row is a sequence, not a canvas.** It plays one Track (several rows may
//! share it) as a chain of frames laid left to right, each simply played after
//! the one before: nothing is positioned, dragged, or overlapping. Rows sound
//! together exactly to the extent that the lengths before a frame add up the
//! same — that is what makes a chord.
//!
//! **Frames come in pairs.** A *note* frame carries a pitch and a length; the
//! *space* behind it carries the silence that follows. They are one [`Item`], so
//! no code path can orphan a space or leave a note without one, and the space
//! has no delete button of its own. Zero length is normal — a placeholder ready
//! to be given one.
//!
//! **Every row also opens with a space** ([`Row::lead`]), which is how one row is
//! offset against another. It belongs to the row, not to a note, so deleting the
//! first note leaves it in place to lead the next; it starts at zero.
//!
//! **Length is three select boxes**: whole notes, then how many of a fraction
//! down to 1/256 — the nominator, so `3` of a `1/8` is the dotted quarter that
//! neither box could say on its own. Notes and spaces carry the same three.
//! Time is counted in [`UNITS_PER_WHOLE`]ths, so every length is a whole number
//! of units and the arithmetic stays exact. Playback lives in [`player`].
//!
//! **Frames are edited in blocks as well as one at a time.** A note frame's
//! select box puts it in the **block**, which is a stretch of frames spanning
//! any number of rows at once: "Span Rows" takes the same stretch of *time* out
//! of every row, "Clone" repeats the block in place on all of them, and
//! "Copy"/"Paste at End" carry it to the end of the composition. What keeps the
//! rows in step through all of it is that a copy occupies the block's window —
//! the time from its first note to the end of its last space — rather than the
//! frames' own extent; see [`ComposerPanel::clone_block`].
//!
//! **A repeat can be trimmed at either end.** "Repeat" loops the whole written
//! length by default; the two lengths beneath the transport say where it starts
//! and where it stops, each counted **from the beginning of the composition** —
//! lengths rather than timestamps, because that is the only vocabulary the
//! panel has, and from the same end as each other so the pair can be read
//! against each other at a glance. A stop of zero is the end of the
//! composition. Play with Repeat ticked starts at the window's start, so every
//! pass — the first one too — is the window. See
//! [`ComposerPanel::repeat_span_units`].
//!
//! **Rows can also be played in rather than written.** "Record & Play Once"
//! plays the composition through once and captures a MIDI keyboard against it,
//! rounding what was played onto a chosen note value and appending it as new
//! rows — see [`record`].

pub mod player;
pub mod project;
pub mod record;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;

use self::player::{
    CompositionPlayer, LoopSpan, PlannedNote, PreparedComposition, RowEdit, RowPlan,
};
use crate::midi::{add_midi_tap, gm_percussion_name, MidiTap, MidiTaps};
use super::registry::TrackRegistry;

/// Time resolution: a whole note is this many units. 256 makes every fraction in
/// [`Fraction`] — down to a 1/256 — a whole number of them, so lengths and the
/// positions they add up to are exact integer arithmetic.
const UNITS_PER_WHOLE: i64 = 256;
/// A beat is a quarter note.
const UNITS_PER_BEAT: i64 = UNITS_PER_WHOLE / 4;

/// On-screen width of one frame. Fixed: a frame carries up to four select boxes
/// and stops being usable below about this width, and a width proportional to
/// the length would make a 1/256 invisible.
const CARD_W: f32 = 108.0;
/// Width of a **wav row's** note frame, which is wider than the rest.
///
/// It carries a control none of the others do — the slider that says where in
/// the file the note starts — and that slider is the one thing on a frame whose
/// usefulness is measured in pixels: over a six-second recording, a track this
/// panel's ordinary width would move the start by sixty milliseconds a pixel,
/// which is most of the distance between landing on the beat and landing behind
/// it. Frames are laid side by side and nothing lines up across rows, so a
/// wider one costs nothing but its own space.
const WAV_CARD_W: f32 = 184.0;
/// Room kept for the number beside the start slider — the part of it that is
/// dragged a millisecond at a time, or typed into.
const START_VALUE_W: f32 = 70.0;
/// Height of one frame — a header line plus up to four select boxes. A box costs
/// 24 of it under the app's text sizes, and a card that no longer fits grows
/// past this, over the lane it is drawn in; grow the two together — and see
/// `every_frame_draws_its_nominator_inside_its_card`, which measures the card
/// against this rather than trusting the arithmetic.
const CARD_H: f32 = 124.0;
/// What a frame is filled with: an amber space, a note, and a note while it is
/// sounding.
const SPACE_FILL: egui::Color32 = egui::Color32::from_rgb(78, 62, 38);
const NOTE_FILL: egui::Color32 = egui::Color32::from_rgb(52, 66, 92);
const NOTE_SOUNDING_FILL: egui::Color32 = egui::Color32::from_rgb(74, 100, 138);
/// …and what a frame in the selected block is filled with: the same colour,
/// lit. A block runs across rows, so it has to be legible as *one* shape in a
/// panel full of cards — the fill says which frames, the [`SEL_STROKE`] round
/// them says they are one thing.
const SPACE_SEL_FILL: egui::Color32 = egui::Color32::from_rgb(108, 86, 50);
const NOTE_SEL_FILL: egui::Color32 = egui::Color32::from_rgb(70, 92, 130);
/// The border every frame in the block wears.
const SEL_STROKE: egui::Color32 = egui::Color32::from_rgb(238, 198, 120);
/// All of them together, which is how a layout test picks the cards out of what
/// the lane painted: they are the only filled rectangles in it.
const CARD_FILLS: [egui::Color32; 5] = [
    SPACE_FILL,
    NOTE_FILL,
    NOTE_SOUNDING_FILL,
    SPACE_SEL_FILL,
    NOTE_SEL_FILL,
];
/// Height of one row's lane. The frames plus a little air around them.
const ROW_H: f32 = CARD_H + 10.0;
/// The row head (name, track select, add-note, gain) takes this share of the
/// lanes' width, between the two bounds below. Every row's head is exactly the
/// same width, so every lane — and with it every lead frame — starts at the
/// same x: a head sized by its own contents put a row playing a long file name
/// a few hundred points to the right of its neighbours.
const HEAD_FRAC: f32 = 0.3;
/// Narrowest head: what the widest fixed line (add-note, gain, auto) needs.
const HEAD_W_MIN: f32 = 420.0;
/// Widest head: past this a longer track name is not worth the lane it costs.
const HEAD_W_MAX: f32 = 640.0;
/// Room beside a full-width box for the small button that goes with it (the
/// rename pencil, the row's bin).
const HEAD_BUTTON_W: f32 = 40.0;

/// The loudest a row may be set to, +34 dB. Well past unity: a drum sampled
/// quietly, or a resynthesised voice analysed well below full scale, has to
/// come up to a lead rather than the whole rest of the composition coming down
/// to it, and a source peaking at -30 dBFS needs more than eight times. The mix
/// limits at full scale, which is what limits how much of this is useful — past
/// the point where the peaks are being held down, more gain only compresses.
const GAIN_MAX: f32 = 50.0;

/// Where the row gain slider's travel starts, -40 dB. Below it the slider drops
/// to silence: the range spans four decades logarithmically, so unity keeps
/// half the travel instead of the two percent a linear run to [`GAIN_MAX`]
/// would leave it.
const GAIN_MIN_TRAVEL: f64 = 0.01;

/// Lowest and highest note offered, C0..B8.
const PITCH_MIN: u8 = 12;
const PITCH_MAX: u8 = 119;
/// A new note starts at middle C.
const DEFAULT_PITCH: u8 = 60;
/// …unless the row plays a drum kit, where middle C is a bongo and the note
/// anyone wants first is the kick.
const DEFAULT_DRUM_PITCH: u8 = 36;
/// Largest whole-note count a length may carry.
const MAX_WHOLES: u16 = 16;
/// …except in the repeat window, whose two lengths are counted from the very
/// beginning of the composition and so have to reach anywhere in a long piece.
const MAX_REPEAT_WHOLES: u16 = 2048;
/// Largest nominator — how many of its fraction a length may carry. Sixteen,
/// like [`MAX_WHOLES`]: beyond that the whole-note box says the same thing in
/// fewer parts.
const MAX_NUM: u8 = 16;

/// The fractional part of a length, as the fraction of a whole note it is named
/// for. `None` is a length that is a whole number of whole notes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fraction {
    None,
    Half,
    Quarter,
    Eighth,
    Sixteenth,
    ThirtySecond,
    SixtyFourth,
    HundredTwentyEighth,
    TwoHundredFiftySixth,
}

impl Fraction {
    const ALL: [Fraction; 9] = [
        Fraction::None,
        Fraction::Half,
        Fraction::Quarter,
        Fraction::Eighth,
        Fraction::Sixteenth,
        Fraction::ThirtySecond,
        Fraction::SixtyFourth,
        Fraction::HundredTwentyEighth,
        Fraction::TwoHundredFiftySixth,
    ];

    /// Length in grid units.
    fn units(self) -> i64 {
        match self {
            Fraction::None => 0,
            Fraction::Half => UNITS_PER_WHOLE / 2,
            Fraction::Quarter => UNITS_PER_WHOLE / 4,
            Fraction::Eighth => UNITS_PER_WHOLE / 8,
            Fraction::Sixteenth => UNITS_PER_WHOLE / 16,
            Fraction::ThirtySecond => UNITS_PER_WHOLE / 32,
            Fraction::SixtyFourth => UNITS_PER_WHOLE / 64,
            Fraction::HundredTwentyEighth => UNITS_PER_WHOLE / 128,
            Fraction::TwoHundredFiftySixth => 1,
        }
    }

    /// What the fraction is named after — the `8` of a `1/8`. [`Fraction::None`]
    /// is not a fraction at all, so it answers `1`, the denominator that changes
    /// nothing.
    fn denom(self) -> i64 {
        match self.units() {
            0 => 1,
            units => UNITS_PER_WHOLE / units,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Fraction::None => "—",
            Fraction::Half => "1/2",
            Fraction::Quarter => "1/4",
            Fraction::Eighth => "1/8",
            Fraction::Sixteenth => "1/16",
            Fraction::ThirtySecond => "1/32",
            Fraction::SixtyFourth => "1/64",
            Fraction::HundredTwentyEighth => "1/128",
            Fraction::TwoHundredFiftySixth => "1/256",
        }
    }

    /// How `num` of this fraction reads: three eighths as `3/8`. One of them is
    /// just the fraction's own name, and `None` has nothing to count.
    fn label_times(self, num: u8) -> String {
        match (self, num) {
            (Fraction::None, _) | (_, 1) => self.label().to_string(),
            _ => format!("{num}/{}", self.denom()),
        }
    }
}

/// A length: a whole-note count plus so many of a fraction, added together.
/// Each part is picked in its own select box, which is why they are stored apart
/// rather than as a single unit count.
///
/// The nominator is what lets a frame say `3/8`: the fraction box halves all the
/// way down, so every length it can name on its own is a power of two, and the
/// dotted values that most music is full of are not among them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Duration {
    /// Whole notes, `0..=`[`MAX_WHOLES`].
    wholes: u16,
    /// How many of [`Duration::frac`] — the nominator, `1..=`[`MAX_NUM`].
    ///
    /// Kept canonical by [`Duration::canonicalise`]: never zero, and forced back
    /// to one when there is no fraction to count. Two lengths that sound the
    /// same are then the same value, which is what lets the project file
    /// round-trip and `PartialEq` mean what the tests read it as.
    num: u8,
    frac: Fraction,
}

impl Duration {
    /// One of `frac` past `wholes` whole notes — the plain length, the one the
    /// nominator leaves alone.
    const fn new(wholes: u16, frac: Fraction) -> Self {
        Self { wholes, num: 1, frac }
    }

    /// `num` of `frac` past `wholes` whole notes, canonicalised.
    fn with_num(wholes: u16, num: u8, frac: Fraction) -> Self {
        let mut d = Self { wholes, num, frac };
        d.canonicalise();
        d
    }

    /// Restore the nominator's invariant after the select boxes have been at it:
    /// a fraction is counted at least once, and no fraction is counted at all.
    fn canonicalise(&mut self) {
        if self.frac == Fraction::None {
            self.num = 1;
        } else {
            self.num = self.num.max(1);
        }
    }

    fn units(self) -> i64 {
        self.wholes as i64 * UNITS_PER_WHOLE + self.num as i64 * self.frac.units()
    }

    /// The closest length the three boxes can name to `secs`, at `spu` seconds
    /// per grid unit — which is where the tempo comes into it.
    ///
    /// Every length is `wholes` whole notes plus `num` of a fraction, so for
    /// each of the (nominator, fraction) pairs there are, the whole-note count
    /// that gets closest is just the rest of the target rounded off. That makes
    /// this a short loop rather than a search over every length the boxes can
    /// say.
    ///
    /// Ties go to the plainest way of saying it — the coarsest fraction, then
    /// the smallest nominator — so half a whole note comes back as `1/2` rather
    /// than as `2/4` or `128/256`, which name the same length.
    fn closest_to(secs: f64, spu: f64) -> Duration {
        let target = if spu > 0.0 { (secs / spu).max(0.0) } else { 0.0 };
        let mut best = Duration::new(0, Fraction::None);
        let mut best_key = (f64::INFINITY, 0i64, 0u8);
        for frac in Fraction::ALL {
            // A fraction that is no fraction has nothing to count, so it is one
            // candidate rather than sixteen identical ones.
            let max_num = if frac == Fraction::None { 1 } else { MAX_NUM };
            for num in 1..=max_num {
                let counted = num as i64 * frac.units();
                let wholes = ((target - counted as f64) / UNITS_PER_WHOLE as f64)
                    .round()
                    .clamp(0.0, MAX_WHOLES as f64) as u16;
                let d = Duration::with_num(wholes, num, frac);
                let key = ((d.units() as f64 - target).abs(), frac.denom(), num);
                if key < best_key {
                    best_key = key;
                    best = d;
                }
            }
        }
        // A note with something left to play must not come back with no length
        // at all: a zero-length frame is skipped by the transport, and "fit this
        // to the file" answering with silence is worse than answering with the
        // shortest note there is. Only reachable for a remainder under half a
        // 1/256 — a few milliseconds.
        if target > 0.0 && best.units() == 0 {
            best = Duration::new(0, Fraction::TwoHundredFiftySixth);
        }
        best
    }

    /// The one length the three boxes can name that is exactly `units`, if
    /// there is one.
    ///
    /// A length is whole notes plus a nominator's worth of one fraction, so a
    /// unit count is nameable only when what is left over after the whole notes
    /// divides into at most [`MAX_NUM`] of some fraction: `255/256` does not, as
    /// nothing but the 1/256 divides it and that would take 255 of them. This is
    /// what lets a block edit put a silence *into a frame that already exists*
    /// rather than adding one — and [`Duration::split`] is what it falls back on
    /// when the answer is `None`.
    ///
    /// The coarsest fraction that fits is the one returned, so 128 units come
    /// back as `1/2` rather than as `128/256`, matching [`Duration::closest_to`].
    fn exact(units: i64) -> Option<Duration> {
        if units < 0 {
            return None;
        }
        let wholes = (units / UNITS_PER_WHOLE).min(MAX_WHOLES as i64);
        let rem = units - wholes * UNITS_PER_WHOLE;
        if rem == 0 {
            return Some(Duration::new(wholes as u16, Fraction::None));
        }
        for frac in Fraction::ALL {
            let f = frac.units();
            if f > 0 && rem % f == 0 && rem / f <= MAX_NUM as i64 {
                return Some(Duration::with_num(wholes as u16, (rem / f) as u8, frac));
            }
        }
        None
    }

    /// `units` spread over as few lengths as the boxes can name it in — what a
    /// silence too odd for one frame is laid out over.
    ///
    /// Whole by preference: a length [`Duration::exact`] can say is one entry,
    /// and only a remainder no fraction divides finely enough is chipped away
    /// coarsest-first. Each step takes more than half of what is left, so this
    /// is a handful of entries at worst and one entry nearly always.
    fn split(units: i64) -> Vec<Duration> {
        let mut left = units;
        let mut out = Vec::new();
        while left > 0 {
            if let Some(d) = Duration::exact(left) {
                out.push(d);
                break;
            }
            let wholes = (left / UNITS_PER_WHOLE).min(MAX_WHOLES as i64);
            let rem = left - wholes * UNITS_PER_WHOLE;
            let mut d = Duration::new(wholes as u16, Fraction::None);
            for frac in Fraction::ALL {
                let f = frac.units();
                if f > 0 && f <= rem {
                    let num = (rem / f).min(MAX_NUM as i64) as u8;
                    d = Duration::with_num(wholes as u16, num, frac);
                    break;
                }
            }
            // Nothing left that a 1/256 could take: unreachable while `left`
            // is positive, and a break rather than a hang if it ever is not.
            if d.units() == 0 {
                break;
            }
            left -= d.units();
            out.push(d);
        }
        out
    }

    /// How far this length is from `secs`, in seconds — what the fit could not
    /// name exactly. Signed: positive is longer than asked for.
    fn error_secs(self, secs: f64, spu: f64) -> f64 {
        self.units() as f64 * spu - secs
    }

    /// How the length reads in the frame's header, e.g. `1 + 3/8`.
    fn label(self) -> String {
        match (self.wholes, self.frac) {
            (0, Fraction::None) => "0".to_string(),
            (0, f) => f.label_times(self.num),
            (w, Fraction::None) => w.to_string(),
            (w, f) => format!("{w} + {}", f.label_times(self.num)),
        }
    }
}

/// What the block controls say when they are pressed with nothing selected.
const NO_BLOCK: &str = "Nothing selected — press ☐ on a note frame to start a block.";

/// How a stretch of time reads in the block bar: as a length when the length
/// boxes can name it, and as a count of whole notes when they cannot — a window
/// is the sum of whatever was selected and does not have to be nameable.
fn span_label(units: i64) -> String {
    match Duration::exact(units) {
        Some(d) => d.label(),
        None => format!("{:.2} whole", units as f64 / UNITS_PER_WHOLE as f64),
    }
}

/// Scientific pitch name of a MIDI note number (`60` → `C4`).
fn pitch_name(pitch: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!("{}{}", NAMES[(pitch % 12) as usize], pitch as i32 / 12 - 1)
}

/// What a note is called on a row, which depends on what the row plays: a pitch
/// on an instrument, and the drum it hits on a kit.
///
/// The drum name is a General MIDI one, and it is shown *beside* the pitch, not
/// instead of it: on a kit whose pads are not GM the note still plays whatever
/// is on the pad, and the pitch — the thing actually sent — stays visible.
fn note_label(pitch: u8, percussion: bool) -> String {
    match percussion.then(|| gm_percussion_name(pitch)).flatten() {
        Some(drum) => format!("{} · {drum}", pitch_name(pitch)),
        None => pitch_name(pitch),
    }
}

/// What the fit button on a wav note offers to do, spelled out: how much of the
/// file the note has left to play, the closest length the boxes can name to it,
/// and how far off that is.
///
/// The error is the whole point of saying it. A recording is whatever length it
/// is, and a bar of music is a whole number of note values, so the two agree
/// only by luck — the button gets as close as the grid allows and this says how
/// close, rather than leaving a few milliseconds of drift to be discovered on
/// the twentieth repeat.
fn fit_hint(left_secs: f64, fitted: Duration, spu: f64) -> String {
    let err_ms = fitted.error_secs(left_secs, spu) * 1000.0;
    let how_close = match err_ms.abs() {
        e if e < 0.5 => "exactly".to_string(),
        _ if err_ms > 0.0 => format!("{err_ms:.0} ms long"),
        _ => format!("{:.0} ms short", -err_ms),
    };
    format!(
        "Fit this note to the file.\n\nFrom where this note starts there is \
         {left_secs:.3} s of the recording left. At this tempo the closest length \
         these boxes can name is {}, which is {how_close}.\n\nThe fit is made at \
         the tempo it is pressed at: change the BPM and press it again.",
        fitted.label()
    )
}

/// A note and the space tied behind it — two frames on screen, one thing in the
/// model. They are stored together rather than as two entries in the row
/// because the space is not independently removable: it exists only as the
/// silence after *this* note, and goes when the note goes. Making that
/// structural means no code path can leave a space orphaned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Item {
    /// Stable within its row, so the egui widget ids survive frames being
    /// deleted around it.
    pub(crate) id: u64,
    pub(crate) pitch: u8,
    /// Where in the file this note starts, in seconds — a **wav row** only,
    /// where there is a file to start into. Zero everywhere else, and ignored:
    /// a plugin note starts where a note starts.
    ///
    /// This is what lines a recording up with the beat. A take does not begin
    /// on its first sample — there is a breath, a stick click, a moment of room
    /// before the downbeat — so a frame that always played from sample zero
    /// would put the *silence* on the beat and the music behind it. The slider
    /// on the frame moves the file under the note until the transient lands on
    /// it.
    pub(crate) start: f32,
    /// How long the note sounds.
    pub(crate) dur: Duration,
    /// The silence after it. Zero is legal and useful: the frame stays on
    /// screen as a placeholder, ready to be given a length, while the next note
    /// follows on immediately.
    pub(crate) space: Duration,
}

impl Item {
    /// The note and its space together — what the next note waits for.
    fn total_units(&self) -> i64 {
        self.dur.units() + self.space.units()
    }
}

/// One lane: a Track and the chain of frames played on it.
struct Row {
    /// Stable id, so egui widget state survives rows being removed above them.
    id: u64,
    /// What this row is **called**. A row nobody has named is named after a
    /// fresh UUID, so every row has a name from the moment it exists and no two
    /// rows start out sharing one; the box on the row's head replaces it with
    /// whatever the user would rather call it.
    ///
    /// A name is for reading, not for looking up: nothing addresses a row by it,
    /// so two rows *may* end up called the same thing, and a renamed row is the
    /// same row it was. That is deliberate — the id above is what identity is
    /// made of, and a name the user cannot repeat is a name they have to fight.
    name: String,
    /// The name as typed, before Rename commits it. Kept apart from `name` so a
    /// half-typed name is not yet the row's name, and so a box the user thinks
    /// better of is abandoned by simply not pressing the button.
    name_edit: String,
    /// The Track this row plays, by registry id. `None` only while the registry
    /// is empty.
    track_id: Option<u64>,
    /// How loud the row plays, `1.0` being the track's own level. Up to
    /// [`GAIN_MAX`], so a quiet source can be lifted rather than every other
    /// row having to be pulled down around it.
    gain: f32,
    /// The silence before the row's first note. Held by the row, not an item, so
    /// deleting the first note cannot take it along — it stays at the head and
    /// leads whichever note is first afterwards. Zero by default.
    lead: Duration,
    /// Play this row. On by default — a row exists to be heard — and off takes
    /// it out of the transport and out of an export, while leaving everything
    /// on it exactly where it is: the notes, the block, the track it plays.
    ///
    /// This is how a part is listened to without being deleted: mute the two
    /// rows that are drowning it, hear the one that is not working, put them
    /// back. Deleting the row and building it again is the alternative, and it
    /// loses the work.
    ///
    /// The lane is still drawn, and still counts towards the length of the
    /// composition: it is silent, not gone.
    enabled: bool,
    /// Re-export this row's LeSynth grid into the project folder on every save.
    /// On by default, so a project saved after an edit carries the edit; off
    /// pins whatever `.lsft` is already there.
    autosave: bool,
    /// Scroll this lane to keep the sounding frame in view while the transport
    /// runs. On by default: a row of any length is wider than the window, so a
    /// lane that does not follow shows the first few seconds of the composition
    /// and nothing of the rest. Off pins the lane where the user left it, which
    /// is what you want while editing one part of a long row as it plays.
    autoscroll: bool,
    /// The source a loaded project asked for and the app could not find. Kept
    /// whole, not just as a message, so saving the project again preserves the
    /// reference rather than quietly replacing it with "no track" — a save after
    /// a load must not be the thing that loses the file name.
    ///
    /// A row in this state is **not** auto-adopted onto another track by
    /// [`ComposerPanel::reconcile_tracks`]: the point is that the user is told
    /// what is missing and picks the replacement.
    missing: Option<project::TrackSource>,
    /// In play order: item `n` starts where item `n - 1`'s space ended.
    items: Vec<Item>,
    next_item_id: u64,
    /// The block selected on this row, as the ids of the frames at its two
    /// ends — see [`Row::sel_range`].
    ///
    /// Ids rather than positions: a frame deleted in front of the block would
    /// slide a range of *indices* silently onto other notes, and a whole block
    /// is the one thing in this panel that is edited without being looked at.
    /// The ends are stored as picked, anchor first, so shift-clicking keeps
    /// dragging the same end.
    sel: Option<(u64, u64)>,
}

impl Row {
    fn new(id: u64, track_id: Option<u64>) -> Self {
        // Named before it is anything else. A UUID says nothing about the row,
        // which is the point: it is a name that is already there to be replaced,
        // rather than a "Row 3" that goes on meaning row 3 after two rows above
        // it have gone.
        let name = uuid::Uuid::new_v4().to_string();
        Self {
            id,
            name: name.clone(),
            name_edit: name,
            track_id,
            gain: 1.0,
            lead: Duration::new(0, Fraction::None),
            enabled: true,
            autosave: true,
            autoscroll: true,
            missing: None,
            items: Vec::new(),
            next_item_id: 0,
            sel: None,
        }
    }

    /// Name the row outright — the box and the name both, so nothing is left
    /// looking half-edited. What loading a project and naming a new row use.
    fn set_name(&mut self, name: String) {
        self.name_edit = name.clone();
        self.name = name;
    }

    /// Take what is in the box as the row's name. Whitespace at either end goes,
    /// and a box holding nothing but that renames nothing: a row with no name at
    /// all could not be spoken about, and the UUID it already has is better than
    /// the blank that would replace it.
    ///
    /// Reports whether the name actually changed, which is what the status line
    /// is written from.
    fn commit_name(&mut self) -> bool {
        let name = self.name_edit.trim().to_string();
        if name.is_empty() || name == self.name {
            // Put the box back to what the row is called, so a stray space or an
            // abandoned edit does not sit there looking pending.
            self.name_edit = self.name.clone();
            return false;
        }
        self.set_name(name);
        true
    }

    /// What to call this row in a status line. A row nobody has named is called
    /// after a UUID, and thirty-six characters of hex tells the reader nothing
    /// they can use, so it is cut to its first block — enough to tell two rows
    /// apart in a sentence, while the box on the row still holds all of it.
    fn label(&self) -> String {
        let is_uuid = self.name.len() == 36 && uuid::Uuid::parse_str(&self.name).is_ok();
        if is_uuid {
            format!("{}…", &self.name[..8])
        } else {
            self.name.clone()
        }
    }

    /// The selected block as positions, first and last inclusive. `None` when
    /// nothing is selected on this row — or when what was selected has been
    /// deleted since, which is the whole reason the ends are held as ids.
    fn sel_range(&self) -> Option<(usize, usize)> {
        let (a, b) = self.sel?;
        let a = self.items.iter().position(|i| i.id == a)?;
        let b = self.items.iter().position(|i| i.id == b)?;
        Some((a.min(b), a.max(b)))
    }

    /// A click on a frame's select box. A box is drawn as a checkbox and must
    /// behave like one: a plain click on a frame **outside** the block grows
    /// the block to reach it, so clicking two boxes selects both and everything
    /// between — the block stays one stretch of time, which is the only thing
    /// it can be. A plain click on a frame already **in** the block drops the
    /// block, so one box both selects and deselects.
    ///
    /// `extend` (shift) instead drags the free end onto this frame, keeping the
    /// end it was anchored on — the way to shrink a block, or to move an end
    /// back past the other one.
    ///
    /// Rows are selected independently: clicking on one row never clears
    /// another, because a block that spans rows is built by clicking on each of
    /// them — or, faster, by marking one row and pressing “Span Rows”.
    fn select_item(&mut self, id: u64, extend: bool) {
        let Some((anchor, _)) = self.sel else {
            self.sel = Some((id, id));
            return;
        };
        if extend {
            self.sel = Some((anchor, id));
            return;
        }
        // Positions, not ids: which side of the block the click landed on is
        // what decides whether it grows or drops, and `sel_range` is also what
        // reports a block whose frames have been deleted since.
        let (Some((a, b)), Some(at)) =
            (self.sel_range(), self.items.iter().position(|i| i.id == id))
        else {
            self.sel = Some((id, id));
            return;
        };
        self.sel = match at {
            // Inside: the click is the second press on a box that is already
            // ticked, and lets the block go.
            _ if (a..=b).contains(&at) => None,
            // Outside: grow to it, anchored on the end that did not move, so a
            // shift-click after this one still drags the end just clicked.
            _ if at < a => Some((self.items[b].id, id)),
            _ => Some((self.items[a].id, id)),
        };
    }

    /// Every frame on the row, as one block — what the select box on the row
    /// head ticks. Clicking the first frame and shift-clicking the last does
    /// the same thing, but a row of two hundred frames is a scroll to the end
    /// and back, and a whole row is the commonest block there is: it is what a
    /// part *is*, and what gets cloned to repeat it.
    ///
    /// Rows stay independent, so ticking the box on several rows builds a block
    /// across all of them — every frame of each — which is the shape a repeat
    /// of a whole section wants.
    fn select_all(&mut self) {
        self.sel = match (self.items.first(), self.items.last()) {
            (Some(first), Some(last)) => Some((first.id, last.id)),
            _ => None,
        };
    }

    /// Whether the block on this row is the whole of it — what the row's select
    /// box shows. Derived rather than stored, so clicking frames unticks it on
    /// its own the moment the block is no longer everything.
    fn all_selected(&self) -> bool {
        !self.items.is_empty() && self.sel_range() == Some((0, self.items.len() - 1))
    }

    /// Where the transport is *for this lane*: nowhere, while the row is
    /// switched off. A row that is not playing must not be lit up as though it
    /// were, and must not drag its lane along under the user's hands — see
    /// [`Row::enabled`]. This is what turns off the sounding frame and, with
    /// it, the follow: neither has anything to point at.
    fn lane_playhead(&self, transport: Option<f64>) -> Option<f64> {
        transport.filter(|_| self.enabled)
    }

    /// Select the run of items `a..=b`, by position. Used by the block
    /// operations, which know where they put the frames they made.
    fn select_range(&mut self, a: usize, b: usize) {
        self.sel = match (self.items.get(a), self.items.get(b)) {
            (Some(first), Some(last)) => Some((first.id, last.id)),
            _ => None,
        };
    }

    /// Hand out the next item id.
    fn take_item_id(&mut self) -> u64 {
        let id = self.next_item_id;
        self.next_item_id += 1;
        id
    }

    /// Where the row's first note starts: the lead space, which counts only
    /// while there is a note for it to lead. An empty row has no first note and
    /// so no lead — and draws none.
    fn lead_units(&self) -> i64 {
        if self.items.is_empty() {
            0
        } else {
            self.lead.units()
        }
    }

    /// Total length of the row in units — the lead space plus the items, which
    /// simply add up.
    fn end_units(&self) -> i64 {
        self.lead_units() + self.items.iter().map(Item::total_units).sum::<i64>()
    }

    /// Append a note and, tied behind it, the space that separates it from
    /// whatever comes next.
    ///
    /// The new frame **repeats the last one whole** — its pitch, its note length
    /// and its space, whole notes, nominator and fraction alike. A row is nearly
    /// always built by repeating something: a run on one note, the same drum
    /// over and over, a rhythm of one length. What was chosen a moment ago is a
    /// far better guess than any fixed default, and carrying only *part* of it
    /// is the worst of both — seven select boxes to check every time, five of
    /// which have silently gone back to a default.
    ///
    /// This is one copy of the whole [`Item`], so a box added to a frame is
    /// carried by the same rule the moment it exists, with nothing here to
    /// remember to update.
    ///
    /// `default_pitch` is what the *first* note of a row starts on; the lengths
    /// it starts on are a quarter note and an eighth of silence — one of each,
    /// the nominator that changes nothing.
    fn add_note(&mut self, default_pitch: u8) {
        let id = self.next_item_id;
        self.next_item_id += 1;
        let item = match self.items.last() {
            Some(last) => Item { id, ..*last },
            None => Item {
                id,
                pitch: default_pitch,
                start: 0.0,
                dur: Duration::new(0, Fraction::Quarter),
                space: Duration::new(0, Fraction::Eighth),
            },
        };
        self.items.push(item);
    }

    /// Delete a note *and* the space tied to it — the only way either of them
    /// leaves the row.
    fn delete_item(&mut self, idx: usize) {
        if idx < self.items.len() {
            self.items.remove(idx);
        }
        // A block one of whose ends has just gone is no block. Dropping it here
        // rather than leaving `sel_range` to fail keeps "is anything selected"
        // one question with one answer.
        if self.sel_range().is_none() {
            self.sel = None;
        }
    }

    /// Where each note starts, in units — the running sum of everything before
    /// it. One entry per item; its space starts where the note ends.
    fn starts(&self) -> Vec<i64> {
        let mut at = self.lead_units();
        self.items
            .iter()
            .map(|i| {
                let start = at;
                at += i.total_units();
                start
            })
            .collect()
    }

    /// The row's notes in seconds, at `spu` seconds per grid unit. Spaces only
    /// advance the clock, and a note given no length at all is not played.
    fn planned_notes(&self, spu: f64) -> Vec<PlannedNote> {
        let mut at = self.lead_units();
        let mut out = Vec::new();
        for item in &self.items {
            let units = item.dur.units();
            if units > 0 {
                out.push(PlannedNote {
                    at_secs: at as f64 * spu,
                    dur_secs: units as f64 * spu,
                    pitch: item.pitch,
                    start_secs: item.start as f64,
                });
            }
            at += item.total_units();
        }
        out
    }
}

/// A **block**: the frames selected across the rows, resolved against the
/// composition.
///
/// A block is not a rectangle of notes — rows hold different numbers of frames
/// of different lengths, and nothing lines up across them by position. What
/// makes it one thing is the **window**: the stretch of time from the first
/// note in the selection to the end of the last space in it. Every row's part is
/// copied against that window rather than against its own extent, which is what
/// keeps a chord a chord and a row that comes in late coming in late.
struct Block {
    /// One entry per row that has something selected, in row order.
    rows: Vec<BlockRow>,
    /// Start of the window: the earliest selected note in any row.
    from: i64,
    /// End of it: where the last selected space in any row runs out.
    to: i64,
}

struct BlockRow {
    /// Where the row is in [`ComposerPanel::rows`].
    idx: usize,
    /// First and last selected item on it, inclusive.
    first: usize,
    last: usize,
}

impl Block {
    /// How long the window is, in units. Every copy of the block occupies
    /// exactly this, however much silence a given row leaves at either end.
    fn window(&self) -> i64 {
        self.to - self.from
    }

    fn frames(&self) -> usize {
        self.rows.iter().map(|r| r.last - r.first + 1).sum()
    }
}

/// A block lifted out of the composition and held for putting back — the
/// clipboard.
///
/// It keeps each row's frames *and* the silence in front of them (`lead`), so
/// pasting reproduces the alignment the block was copied with rather than
/// stacking every row flush against the paste point.
#[derive(Clone)]
struct Clip {
    /// The window the block covered — see [`Block`].
    window: i64,
    rows: Vec<ClipRow>,
}

#[derive(Clone)]
struct ClipRow {
    /// The row this came off, which is the row it goes back on. A block is
    /// copied *between times*, not between tracks: the frames carry pitches and
    /// file starts that mean what they mean on the track they were written for.
    row_id: u64,
    /// Units between the window's start and this row's first copied note.
    lead: i64,
    items: Vec<Item>,
}

impl Clip {
    fn frames(&self) -> usize {
        self.rows.iter().map(|r| r.items.len()).sum()
    }
}

/// Put `units` of extra silence into `space` — the space of whatever frame the
/// insertion follows — and report what would not fit.
///
/// A frame's length is three select boxes, which cannot name every unit count
/// (see [`Duration::exact`]): the rest comes back as lengths for placeholder
/// frames the caller appends. Silence that a block edit has computed must land
/// somewhere *exactly*, because it is the difference between two rows staying in
/// step and drifting apart by a fraction that grows with every repeat.
fn add_gap(space: &mut Duration, units: i64) -> Vec<Duration> {
    if units <= 0 {
        return Vec::new();
    }
    match Duration::exact(space.units() + units) {
        Some(d) => {
            *space = d;
            Vec::new()
        }
        None => Duration::split(units),
    }
}

/// A frame that is nothing but silence: a note of no length — which the
/// transport skips — carrying `space` behind it. What silence too odd for one
/// length box is spread over.
fn spacer(id: u64, pitch: u8, space: Duration) -> Item {
    Item { id, pitch, start: 0.0, dur: Duration::new(0, Fraction::None), space }
}

/// What the Composer wants the app to do with the project. The panel owns the
/// composition but not the plugin instances a save has to read the grids from,
/// nor the Tracks panel a load has to put them into, so the button records the
/// intent and [`ComposerPanel::take_request`] hands it over.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectRequest {
    /// Write the project to `dir`, creating it if needed.
    Save { dir: PathBuf, name: String },
    /// Read the manifest at this path and replace the composition with it.
    Load { file: PathBuf },
}

/// What the running transport has been told, so the panel can tell whether the
/// user has changed anything since.
struct LiveSnapshot {
    /// `(row id, track id)` per playing row — the part a live edit *cannot*
    /// change, because a different track means a different plugin to load.
    shape: Vec<(u64, Option<u64>)>,
    /// The part it can: notes and gain, per row.
    rows: Vec<RowEdit>,
    /// The stretch a repeat runs over — both ends, because either can be moved
    /// by an edit: the boxes name lengths off a composition that is itself
    /// being written.
    span: LoopSpan,
    /// Whether the user has already been told that the shape changed, so the
    /// status line is not rewritten on every frame.
    shape_warned: bool,
}

/// A take in progress: where the keyboard's messages are piling up, and what
/// they are timed against.
///
/// Nothing is written into the composition until it ends. A row is a chain of
/// lengths, so a note's frame cannot be built before the note after it is known
/// — and half a note appearing in a row while the user is still playing would
/// be a distraction, not feedback.
struct Recording {
    /// This take's own copy of the keyboard, stamped as it arrived. A copy, not
    /// the queue itself: an open editor drains that one destructively, and
    /// recording must not take notes away from the instrument the user is
    /// listening to while they play.
    tap: MidiTap,
    /// The clock for a take with no transport behind it (an empty composition
    /// has nothing to play, so there is no stream to ask). With one, the
    /// transport's own [`CompositionPlayer::heard_secs_at`] is used instead, so
    /// what was played lines up with what was heard.
    origin: Instant,
}

/// The Composer panel.
pub struct ComposerPanel {
    registry: TrackRegistry,
    rows: Vec<Row>,
    next_row_id: u64,
    tempo_bpm: f32,
    /// Project name as typed. Sanitised into the folder and manifest name only
    /// when saving, so what the user sees is what they wrote.
    project_name: String,
    /// Where this project was last saved to or loaded from, if anywhere.
    project_dir: Option<PathBuf>,
    /// Set by the Save/Load buttons and taken by the app, which owns the plugin
    /// instances the grids have to be exported from.
    request: Option<ProjectRequest>,
    status: String,
    player: Option<CompositionPlayer>,
    /// A Play that has been pressed but has not made a sound yet: the rows are
    /// loading their plugins on a thread of their own.
    preparing: Option<Receiver<Result<PreparedComposition, String>>>,
    /// Loop the composition instead of stopping at the end. Shared with the
    /// transport's audio callback, so the checkbox works on a running player.
    repeat: Arc<AtomicBool>,
    /// Where a repeat starts and where it stops, both as a length **from the
    /// beginning of the composition**, in the same three select boxes every
    /// other length in the panel is named by.
    ///
    /// Lengths rather than positions because that is the only vocabulary the
    /// panel has — a row is a chain of lengths, nothing in it sits at a
    /// timestamp — and from the same end as each other because a window read
    /// from two directions is a window nobody can check: "1/2" and "3/4" say
    /// which is in front of which at a glance, where a stop counted backwards
    /// has to be subtracted from a total before it can be compared with the
    /// start beside it.
    ///
    /// A stop of zero is the box untouched, and means the end of the
    /// composition — a repeat that stops before it starts is not a thing to
    /// have, so zero is free to mean the useful thing. See
    /// [`Self::repeat_to_units`].
    repeat_from: Duration,
    repeat_to: Duration,
    /// The composition as the looping transport last received it. What an edit
    /// is compared against, so an untouched frame publishes nothing.
    live_sent: Option<LiveSnapshot>,
    /// A running WAV export, which renders on its own thread and sends back the
    /// line to show when it is done. The GUI must not block on it: a long
    /// composition takes seconds, and a frozen window looks like a crash.
    export: Option<Receiver<String>>,
    /// Where a recording gets its own copy of the keyboard from.
    midi_taps: MidiTaps,
    /// The take being played in, if any.
    recording: Option<Recording>,
    /// The note value a take is rounded onto, in units.
    round_units: i64,
    /// The track a recorded row plays. `None` falls back to the first track
    /// there is, which is what an empty registry and a fresh panel both mean.
    record_track: Option<u64>,
    /// The last block copied, ready to be pasted. Not saved with the project: a
    /// clipboard is part of an editing session, not of the music.
    clip: Option<Clip>,
    /// How many copies “Clone” makes at once. A section is nearly always wanted
    /// four times rather than twice, and a repeat that takes one press is the
    /// difference between writing the arrangement and typing it.
    clone_times: u8,
    /// The name to give the next row, as typed into the box beside either Add
    /// Track Row button. Empty — which is how it starts and how it is left
    /// behind every row that takes it — means the row is named after a UUID.
    new_row_name: String,
}

impl ComposerPanel {
    pub fn new(registry: TrackRegistry, midi_taps: MidiTaps) -> Self {
        Self {
            registry,
            rows: Vec::new(),
            next_row_id: 0,
            tempo_bpm: 120.0,
            project_name: "Untitled".to_string(),
            project_dir: None,
            request: None,
            status: "Add a track row to start composing.".to_string(),
            player: None,
            preparing: None,
            repeat: Arc::new(AtomicBool::new(false)),
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            live_sent: None,
            export: None,
            midi_taps,
            recording: None,
            round_units: record::DEFAULT_ROUND_UNITS,
            record_track: None,
            clip: None,
            clone_times: 1,
            new_row_name: String::new(),
        }
    }

    /// Seconds one grid unit lasts.
    fn secs_per_unit(&self) -> f64 {
        60.0 / (self.tempo_bpm.max(1.0) as f64) / UNITS_PER_BEAT as f64
    }

    fn add_row(&mut self) {
        self.add_row_at(self.rows.len());
    }

    /// A new lane at `at` — the top button puts one in front of the list, the
    /// bottom one behind it. Nothing else in the composition is addressed by
    /// row position: a clip remembers the row **id** it came off, and a block
    /// is worked out afresh each frame, so the order is the user's to arrange.
    fn add_row_at(&mut self, at: usize) {
        let at = at.min(self.rows.len());
        let id = self.next_row_id;
        self.next_row_id += 1;
        let track = self.registry.first_id();
        let mut row = Row::new(id, track);
        // The box beside the button names the row that button makes, and is
        // emptied behind it: a name is typed for the row it was typed for, and
        // the row after it goes back to a UUID rather than quietly becoming a
        // second row of the same name.
        let typed = std::mem::take(&mut self.new_row_name).trim().to_string();
        if !typed.is_empty() {
            row.set_name(typed);
        }
        let label = row.label();
        self.rows.insert(at, row);
        self.status = match track.and_then(|t| self.registry.name_of(t)) {
            Some(name) => format!("Added row {label}, playing {name}."),
            None => format!("Added row {label} — no tracks exist yet to assign it to."),
        };
    }

    /// Keep every row pointed at a track that still exists: a row whose track was
    /// removed falls back to the first one available, and a row left without any
    /// (the registry went empty) adopts the first track added afterwards.
    ///
    /// A row whose loaded source went **missing** is left alone. Quietly moving
    /// it onto some other track would hide the one thing the user has to be told
    /// about, and would make the replacement look like it had always been there.
    fn reconcile_tracks(&mut self) {
        let first = self.registry.first_id();
        for row in &mut self.rows {
            if row.missing.is_some() {
                continue;
            }
            let valid = row.track_id.is_some_and(|id| self.registry.contains(id));
            if !valid {
                row.track_id = first;
            }
        }
        if !self.record_track.is_some_and(|id| self.registry.contains(id)) {
            self.record_track = first;
        }
    }

    /// End of the composition in units — the longest row.
    fn end_units(&self) -> i64 {
        self.rows.iter().map(Row::end_units).max().unwrap_or(0)
    }

    /// Where the stop box lands, in units from the beginning: what it says,
    /// held inside the composition — and the *end* of the composition when it
    /// says nothing.
    ///
    /// Zero has to mean something, and "stop at the beginning" is not a thing
    /// anyone wants: a repeat that stops where it starts has nothing in it. So
    /// an untouched box means the end, which is both the useful reading and the
    /// one that leaves a fresh panel looping the whole composition, as it always
    /// has. A stop asked for past the end is held at the end for the same
    /// reason a start is: there is no music out there to repeat, only silence
    /// the loop would sit in.
    fn repeat_to_units(&self) -> i64 {
        let end = self.end_units();
        match self.repeat_to.units() {
            0 => end,
            asked => asked.min(end),
        }
    }

    /// The stretch a repeat loops over, in units — start included, end
    /// excluded. Both ends are read off the composition as it stands now, so a
    /// window follows it as it is written.
    ///
    /// The two boxes are free to ask for nothing: a start on or past the stop,
    /// either way round. That is not a short loop — a window of no length is a
    /// transport with no samples to play — so it is answered with the whole
    /// composition, which is what a repeat did before there were boxes to ask
    /// with, and [`Self::repeat_span_ignored`] is what says so on screen.
    fn repeat_span_units(&self) -> (i64, i64) {
        let (from, to) = (self.repeat_from.units(), self.repeat_to_units());
        if from < to {
            (from, to)
        } else {
            (0, self.end_units())
        }
    }

    /// Whether the boxes are asking for something and being overruled — a window
    /// with nothing in it. Says nothing while they are both zero: that is not a
    /// window being ignored, it is no window at all.
    fn repeat_span_ignored(&self) -> bool {
        let asked = self.repeat_from.units() > 0 || self.repeat_to.units() > 0;
        asked && self.repeat_from.units() >= self.repeat_to_units()
    }

    /// The same stretch in seconds, which is what the transport loops on.
    fn repeat_span(&self) -> LoopSpan {
        let spu = self.secs_per_unit();
        let (from, to) = self.repeat_span_units();
        LoopSpan { start_secs: from as f64 * spu, end_secs: to as f64 * spu }
    }

    // ── Blocks ────────────────────────────────────────────────────────────
    //
    // A block is a stretch of frames selected across any number of rows at
    // once, and the reason it exists is that music repeats: a chorus, a bar of
    // drums, a two-bar figure played by three tracks together. Written frame by
    // frame, the second time round costs as much as the first, and the copy
    // drifts out of step with the original the moment one row is a 1/16 short.
    // Copying whole blocks makes the repeat one press, and pins the alignment
    // to arithmetic rather than to counting boxes by eye.

    /// The selected block, resolved — `None` when nothing is selected anywhere.
    fn block(&self) -> Option<Block> {
        let mut rows = Vec::new();
        let (mut from, mut to) = (i64::MAX, i64::MIN);
        for (idx, row) in self.rows.iter().enumerate() {
            let Some((first, last)) = row.sel_range() else {
                continue;
            };
            let starts = row.starts();
            from = from.min(starts[first]);
            to = to.max(starts[last] + row.items[last].total_units());
            rows.push(BlockRow { idx, first, last });
        }
        (!rows.is_empty()).then_some(Block { rows, from, to })
    }

    /// Select every frame in every row — the whole composition as one block.
    fn select_all(&mut self) {
        let mut rows = 0;
        for row in &mut self.rows {
            match row.items.len() {
                0 => row.sel = None,
                n => {
                    row.select_range(0, n - 1);
                    rows += 1;
                }
            }
        }
        self.status = match rows {
            0 => "Nothing to select — the rows are empty.".to_string(),
            n => format!("Selected everything on {n} row(s)."),
        };
    }

    fn clear_block(&mut self) {
        for row in &mut self.rows {
            row.sel = None;
        }
        self.status = "Selection cleared.".to_string();
    }

    /// Take the block's stretch of time out of **every** row: each row's notes
    /// that begin inside the window join the block.
    ///
    /// This is what makes a natural block one press. Mark a phrase on the row
    /// you can hear it in — the melody, the drums — and the same stretch of the
    /// arrangement, on every other track, comes with it. What the ear calls a
    /// section is a stretch of time, not a count of frames, and no two rows
    /// spend it on the same number of notes.
    ///
    /// The window can only grow: a row joining the block with a note that rings
    /// on past the end of it puts that end further out, so pressing this again
    /// may take in a little more. That is the same rule read twice, not a
    /// different one — and it is why the button says what the block spans.
    fn span_rows(&mut self) {
        let Some(block) = self.block() else {
            self.status = NO_BLOCK.to_string();
            return;
        };
        let (from, to) = (block.from, block.to);
        let mut rows = 0;
        let mut frames = 0;
        for row in &mut self.rows {
            let starts = row.starts();
            let inside = |s: &i64| *s >= from && *s < to;
            match (starts.iter().position(inside), starts.iter().rposition(inside)) {
                (Some(a), Some(b)) => {
                    row.select_range(a, b);
                    rows += 1;
                    frames += b - a + 1;
                }
                _ => row.sel = None,
            }
        }
        self.status =
            format!("Block spans {rows} row(s), {frames} frame(s) — {} of music.", span_label(to - from));
    }

    /// The one frame selected in the whole composition, as the row it is on and
    /// its position on that row. `None` unless exactly one frame is selected
    /// anywhere — a block of two says nothing about which one is meant.
    fn single_selected(&self) -> Option<(usize, usize)> {
        let mut found = None;
        for (idx, row) in self.rows.iter().enumerate() {
            let Some((first, last)) = row.sel_range() else {
                continue;
            };
            if first != last || found.is_some() {
                return None;
            }
            found = Some((idx, first));
        }
        found
    }

    /// Set one space on row `target` — its lead (`space == None`) or the space
    /// tied to item `space` — so the note behind it lands on the one selected
    /// note on *another* row: on where that note starts, or on where it ends
    /// with its own space included. The way to bring a part in on a cue, or
    /// after it, without counting the lengths in front of it by eye.
    ///
    /// Does nothing but say why — and leaves the cue selected, so it can be
    /// tried again — unless exactly one frame is selected and it is on a
    /// different row, and the space starts no later than the cue: a row cannot
    /// be lined up with itself, a cue already behind the space would need a
    /// negative length, and a space that moved its own cue would be chasing it.
    fn align_space(&mut self, target: usize, space: Option<usize>, to: AlignTo) {
        let Some((from, pos)) = self.single_selected() else {
            self.status = "Select exactly one note, on another row, to align the space with.".to_string();
            return;
        };
        if from == target || target >= self.rows.len() {
            self.status = "The selected note is on this row — select one on another row.".to_string();
            return;
        }
        let cue = {
            let row = &self.rows[from];
            let start = row.starts()[pos];
            match to {
                AlignTo::Start => start,
                AlignTo::End => start + row.items[pos].total_units(),
            }
        };
        let row = &self.rows[target];
        let begins = match space {
            None => 0,
            Some(idx) => match row.items.get(idx) {
                Some(item) => row.starts()[idx] + item.dur.units(),
                None => return,
            },
        };
        if begins > cue {
            self.status = format!(
                "{} already runs {} past the selected note's {} — nothing changed.",
                row.label(),
                span_label(begins - cue),
                to.edge()
            );
            return;
        }
        // A space is one length, so it can only be what the boxes can name.
        // Anything else would leave the row off its cue by a fraction, which is
        // the exact thing this button is for avoiding.
        let Some(len) = Duration::exact(cue - begins) else {
            self.status = format!(
                "The selected note's {} is {} on — further than one space can say.",
                to.edge(),
                span_label(cue - begins)
            );
            return;
        };
        let source = self.rows[from].label();
        // The cue has done its job: unticking it leaves the next space free to
        // be aligned with another note without clearing this one by hand first.
        self.rows[from].sel = None;
        let row = &mut self.rows[target];
        match space {
            None => row.lead = len,
            Some(idx) => row.items[idx].space = len,
        }
        self.status = format!(
            "Space on {} set to {} — in with the {} of the selected note on {source}.",
            row.label(),
            len.label(),
            to.edge()
        );
    }

    /// Hold the block for pasting. The frames are copied with the silence in
    /// front of them, per row, so what goes back keeps the shape that was taken.
    fn copy_block(&mut self) {
        let Some(block) = self.block() else {
            self.status = NO_BLOCK.to_string();
            return;
        };
        let rows: Vec<ClipRow> = block
            .rows
            .iter()
            .map(|br| {
                let row = &self.rows[br.idx];
                let starts = row.starts();
                ClipRow {
                    row_id: row.id,
                    lead: starts[br.first] - block.from,
                    items: row.items[br.first..=br.last].to_vec(),
                }
            })
            .collect();
        let clip = Clip { window: block.window(), rows };
        self.status = format!(
            "Copied {} frame(s) from {} row(s) — {} of music.",
            clip.frames(),
            clip.rows.len(),
            span_label(clip.window)
        );
        self.clip = Some(clip);
    }

    /// Put the clipboard back at the end of the composition, on the rows it came
    /// from and all of them at once.
    ///
    /// The paste point is the end of the **longest** row, not of each row: rows
    /// are played from zero together, so pasting each one at its own end would
    /// take a block that sounded together and deal it out as an echo. Every row
    /// is padded up to that point, plus the silence the block was copied with in
    /// front of it.
    ///
    /// A row that has since been removed is reported rather than guessed at: the
    /// frames on it name a pitch, a length and a place in a file that mean what
    /// they mean on the track they were written for.
    fn paste_block(&mut self) {
        let Some(clip) = self.clip.clone() else {
            self.status = "Nothing copied yet — select a block and press Copy.".to_string();
            return;
        };
        let at = self.end_units();
        let (mut rows, mut frames, mut gone) = (0, 0, 0);
        for crow in &clip.rows {
            if crow.items.is_empty() {
                continue;
            }
            let Some(row) = self.rows.iter_mut().find(|r| r.id == crow.row_id) else {
                gone += 1;
                continue;
            };
            // Never negative: `at` is the end of the longest row.
            let pad = (at + crow.lead - row.end_units()).max(0);
            let pitch = crow.items[0].pitch;
            let mut out: Vec<Item> = Vec::new();
            let extra = {
                let space = match row.items.last_mut() {
                    Some(last) => &mut last.space,
                    // An empty row has no frame to put the silence in — and its
                    // lead space *is* that silence, by definition.
                    None => &mut row.lead,
                };
                add_gap(space, pad)
            };
            for d in extra {
                let id = row.take_item_id();
                out.push(spacer(id, pitch, d));
            }
            for item in &crow.items {
                let id = row.take_item_id();
                out.push(Item { id, ..*item });
            }
            let end = row.items.len() + out.len();
            row.items.extend(out);
            // The block moves onto what was just pasted, which is what a second
            // press of Clone or Paste is nearly always meant to work on.
            row.select_range(end - crow.items.len(), end - 1);
            rows += 1;
            frames += crow.items.len();
        }
        self.status = match gone {
            0 => format!("Pasted {frames} frame(s) onto {rows} row(s) at the end of the composition."),
            n => format!(
                "Pasted {frames} frame(s) onto {rows} row(s) — {n} row(s) the block was \
                 copied from are gone."
            ),
        };
    }

    /// Repeat the block `times` over, in place: each copy lands directly behind
    /// the one before it, on every row at once, and whatever followed moves
    /// along by exactly as much on every row.
    ///
    /// The arithmetic is the whole point. A copy occupies the block's **window**
    /// — not the frames' own extent — so the silence a row leaves at the front
    /// of the window and the silence it leaves at the end are put back between
    /// one copy and the next. Every row therefore advances by the same amount
    /// per repeat, and a figure that sounded together on the first pass sounds
    /// together on the fourth.
    ///
    /// The selection moves onto the last copy made, so pressing Clone again
    /// repeats the repeat: the copy is measured the same as its original, which
    /// a selection left on the original — whose trailing space now carries the
    /// gap to the copy — would not be.
    fn clone_block(&mut self, times: u8) {
        let Some(block) = self.block() else {
            self.status = NO_BLOCK.to_string();
            return;
        };
        // The select box never offers it, and a repeat of nothing is nothing.
        if times == 0 {
            return;
        }
        let (window, frames, rows) = (block.window(), block.frames(), block.rows.len());
        for br in &block.rows {
            let row = &mut self.rows[br.idx];
            let starts = row.starts();
            let lead = starts[br.first] - block.from;
            let tail = block.to - (starts[br.last] + row.items[br.last].total_units());
            // What separates one copy from the next on this row: what it leaves
            // at the end of the window plus what it leaves at the start of it.
            let gap = lead + tail;
            let source: Vec<Item> = row.items[br.first..=br.last].to_vec();
            let pitch = source[0].pitch;
            let mut out: Vec<Item> = Vec::new();
            for _ in 0..times {
                let extra = {
                    let space = match out.last_mut() {
                        Some(last) => &mut last.space,
                        None => &mut row.items[br.last].space,
                    };
                    add_gap(space, gap)
                };
                for d in extra {
                    let id = row.take_item_id();
                    out.push(spacer(id, pitch, d));
                }
                for item in &source {
                    let id = row.take_item_id();
                    out.push(Item { id, ..*item });
                }
            }
            let at = br.last + 1;
            let end = at + out.len();
            row.items.splice(at..at, out);
            row.select_range(end - source.len(), end - 1);
        }
        self.status = format!(
            "Cloned {frames} frame(s) on {rows} row(s), ×{times} — {} of music each time.",
            span_label(window)
        );
    }

    /// Every row that has something to play, resolved to seconds. A row with no
    /// notes, with no track behind it, or switched off, simply is not in the
    /// composition — a disabled row costs no plugin instance either, which is
    /// what makes muting a heavy track worth doing.
    fn build_plans(&self) -> Vec<RowPlan> {
        let spu = self.secs_per_unit();
        let mut plans = Vec::new();
        for row in &self.rows {
            if !row.enabled {
                continue;
            }
            let Some(source) = row.track_id.and_then(|id| self.registry.playback_source(id)) else {
                continue;
            };
            let notes = row.planned_notes(spu);
            if notes.is_empty() {
                continue;
            }
            plans.push(RowPlan {
                row_id: row.id,
                source,
                gain: row.gain,
                notes,
            });
        }
        plans
    }

    /// Whether anything would play — without asking the registry for a playback
    /// source, which exports a LeSynth's live grid and is far too much work for
    /// a question this cheap to answer.
    fn has_playable_rows(&self) -> bool {
        let spu = self.secs_per_unit();
        self.rows
            .iter()
            .any(|row| row.enabled && row.track_id.is_some() && !row.planned_notes(spu).is_empty())
    }

    /// The composition as the transport would need it *now*. Cheap by design:
    /// it resolves note times and reads row gains, and deliberately does not
    /// touch the registry — asking that for a playback source exports a
    /// LeSynth's live grid, which is far too much work for a per-frame check.
    fn live_snapshot(&self) -> LiveSnapshot {
        let spu = self.secs_per_unit();
        let mut shape = Vec::new();
        let mut rows = Vec::new();
        for row in &self.rows {
            // Same filter as `build_plans`: a row with no track, no notes, or
            // switched off is not in the composition, so it is not in the
            // transport either. Switching one off while a repeat runs therefore
            // changes the *shape*, and is reported rather than applied — the
            // row's voice is a loaded plugin, and dropping it is not something
            // the audio thread can be asked to do mid-loop.
            if !row.enabled || row.track_id.is_none() {
                continue;
            }
            let notes = row.planned_notes(spu);
            if notes.is_empty() {
                continue;
            }
            shape.push((row.id, row.track_id));
            rows.push(RowEdit {
                row_id: row.id,
                gain: row.gain,
                notes,
            });
        }
        LiveSnapshot {
            shape,
            rows,
            span: self.repeat_span(),
            shape_warned: false,
        }
    }

    /// While a repeat is running, hand the transport anything the user has
    /// changed since it last heard from us. Applied at the next loop point.
    ///
    /// Only notes, gains and the loop length travel this way. Adding a row, or
    /// pointing one at another track, means a plugin to load — which cannot
    /// happen on the audio thread — so that is reported rather than applied.
    fn push_live_edits(&mut self) {
        let Some(player) = &self.player else { return };
        if !self.repeat.load(Ordering::Relaxed) {
            return;
        }
        let Some(sent) = &self.live_sent else { return };

        let snapshot = self.live_snapshot();
        if snapshot.shape != sent.shape {
            if !sent.shape_warned {
                self.status = "Rows or their tracks changed — press Play again to \
                               hear them; note edits still follow the loop."
                    .to_string();
                if let Some(sent) = &mut self.live_sent {
                    sent.shape_warned = true;
                }
            }
            return;
        }
        if snapshot.rows == sent.rows && snapshot.span == sent.span {
            return;
        }
        player.update_live(&snapshot.rows, snapshot.span);
        // One instance has one output, so rows sharing one are at one level. The
        // transport keeps the first row's; saying so beats a slider that visibly
        // moves and audibly does nothing.
        if player.gains_diverged() {
            self.status = "Rows sharing an instance were given different gains — \
                           press Play again to give them one each."
                .to_string();
        }
        self.live_sent = Some(snapshot);
    }

    /// Press Play: load the rows on a thread, and start the transport when they
    /// land ([`Self::poll_preparation`]).
    ///
    /// Loading is not quick — one plugin instance per row, and a plugin can take
    /// a second — so doing it here would freeze the window for as long as it
    /// takes, which is what makes a transport feel broken. The composition is
    /// snapshotted now, so what plays is what was on screen when the button went
    /// down, however long the loading takes.
    fn start_playback(&mut self, ctx: &egui::Context) {
        if self.preparing.is_some() {
            return;
        }
        let plans = self.build_plans();
        if plans.is_empty() {
            self.status = "Nothing to play — add a note to a row first.".to_string();
            return;
        }
        let rows = plans.len();
        let (tx, rx) = std::sync::mpsc::channel();
        // The context, so the rows can wake the window the moment they are
        // ready. Without it the transport waits for whenever the GUI next
        // happens to paint, which is a quarter of a second of silence for
        // nothing.
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(CompositionPlayer::prepare(plans).map_err(|e| format!("{e:#}")));
            ctx.request_repaint();
        });
        self.preparing = Some(rx);
        self.live_sent = Some(self.live_snapshot());
        self.status = format!("Loading {rows} row(s)…");
    }

    /// Start the transport once the rows have finished loading. Opening the
    /// stream is a couple of milliseconds, so that half stays here.
    fn poll_preparation(&mut self) {
        let Some(rx) = &self.preparing else { return };
        let ready = match rx.try_recv() {
            Ok(ready) => ready,
            Err(TryRecvError::Empty) => return,
            // The loading thread died without a word, which is still an answer.
            Err(TryRecvError::Disconnected) => Err("the rows stopped loading".to_string()),
        };
        self.preparing = None;

        let prepared = match ready {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("Playback failed: {e}");
                self.live_sent = None;
                // A take with no backing track and no clock to place it on is
                // not a take. Abandoning it here is better than recording
                // against a silence the user was not expecting.
                self.recording = None;
                return;
            }
        };
        let (loaded, total) = prepared.loaded_rows();
        let instances = prepared.instances();
        // The length may have been edited while the rows were loading; the loop
        // follows what is on screen now, as a live edit would.
        let span = self.repeat_span();
        match CompositionPlayer::start_prepared(prepared, span, self.repeat.clone()) {
            Ok(player) => {
                // Say when rows are sharing: it explains both the memory and why
                // one of them cannot be given its own level while it plays.
                // Only when there are instances to share — a composition of wav
                // rows loads none, and "on 0 shared instance(s)" is not news.
                let shared = match instances > 0 && instances < loaded {
                    true => format!(" on {instances} shared instance(s)"),
                    false => String::new(),
                };
                self.status = if loaded == total {
                    format!("Playing {loaded} row(s){shared}.")
                } else {
                    format!(
                        "Playing {loaded} of {total} row(s){shared} — the rest failed \
                         to load."
                    )
                };
                if let Some(rec) = &self.recording {
                    // The take starts where the sound does. Loading the rows
                    // took as long as it took, and anything played into the tap
                    // meanwhile was played against nothing — the transport's own
                    // clock ([`CompositionPlayer::heard_secs_at`]) takes over
                    // from here, and it starts at this moment.
                    if let Ok(mut tapped) = rec.tap.lock() {
                        tapped.clear();
                    }
                    self.status = format!("⏺ Recording over {loaded} row(s) — play your keyboard.");
                }
                self.player = Some(player);
            }
            Err(e) => {
                self.status = format!("Playback failed: {e:#}");
                self.live_sent = None;
                self.recording = None;
            }
        }
    }

    /// Press Record: play the composition through exactly once, and keep
    /// everything the MIDI keyboard sends while it does.
    ///
    /// Nothing is written into the composition until the pass ends
    /// ([`Self::finish_recording`]). A composition with nothing in it yet still
    /// records — there is simply no transport behind the take, and it runs from
    /// the button press until Stop.
    fn start_recording(&mut self, ctx: &egui::Context) {
        if self.recording.is_some() || self.player.is_some() || self.preparing.is_some() {
            return;
        }
        // Exactly once, as the button says: a loop would have the second pass
        // recorded over the first.
        self.repeat.store(false, Ordering::Relaxed);
        self.recording = Some(Recording {
            tap: add_midi_tap(&self.midi_taps),
            origin: Instant::now(),
        });
        if self.has_playable_rows() {
            self.start_playback(ctx);
            // `start_playback` says how many rows are loading; the take is only
            // armed once they are (see `poll_preparation`).
            self.status = format!("⏺ {}", self.status);
        } else {
            self.status = "⏺ Recording — nothing to play along to, so it runs from now \
                           until you press Stop."
                .to_string();
        }
    }

    /// End a take and append what was played to the composition, rounded onto
    /// the chosen note value.
    ///
    /// Returns the line to show, which the caller puts up *after* whatever
    /// stopping the transport had to say — "Stopped." is not the news here.
    /// Must be called while the player is still in hand: the transport is the
    /// clock the take is placed on.
    fn finish_recording(&mut self) -> Option<String> {
        let rec = self.recording.take()?;
        let messages = match rec.tap.lock() {
            Ok(mut tapped) => std::mem::take(&mut *tapped),
            // The tap is only ever locked for a push and for this; a poisoned
            // one means a MIDI callback panicked, and the take is gone with it.
            Err(_) => return Some("Recording failed: the keyboard tap was lost.".to_string()),
        };

        let notes = {
            let player = self.player.as_ref();
            let clock = |t: Instant| match player {
                Some(p) => p.heard_secs_at(t),
                None => Some(t.saturating_duration_since(rec.origin).as_secs_f64()),
            };
            let end_secs = clock(Instant::now()).unwrap_or(0.0);
            record::notes_from(&messages, clock, end_secs)
        };
        if notes.is_empty() {
            return Some(
                "Recorded nothing — no notes arrived. Connect a keyboard under MIDI \
                 and press Connect."
                    .to_string(),
            );
        }

        let grid = self.round_units;
        let quantized = record::quantize(&notes, self.secs_per_unit(), grid);
        let voices = record::split_voices(&quantized);
        let track = self.record_track.or_else(|| self.registry.first_id());
        for voice in &voices {
            let id = self.next_row_id;
            self.next_row_id += 1;
            let mut row = Row::new(id, track);
            let (lead, items) = record::voice_items(voice, &mut row.next_item_id);
            row.lead = lead;
            row.items = items;
            self.rows.push(row);
        }

        let played = notes.len();
        let rows = voices.len();
        let grid = record::round_label(grid);
        Some(match (track, rows) {
            // A take with no track behind it is still worth keeping: the rows
            // are there, and the select box on each is how they get a sound.
            (None, _) => format!(
                "Recorded {played} note(s) into {rows} new row(s), rounded to {grid} — \
                 no tracks exist yet, so pick one on each row."
            ),
            (Some(_), 1) => format!("Recorded {played} note(s) into a new row, rounded to {grid}."),
            // More rows than one means notes sounded together: a row plays one
            // note at a time, so a chord has to be spread across several. A key
            // merely let go of late does not land here — that stays one row.
            (Some(_), _) => format!(
                "Recorded {played} note(s), rounded to {grid} — spread over {rows} new rows, \
                 because notes sounded together."
            ),
        })
    }

    /// Render the whole composition offline and write it as a `.wav`.
    ///
    /// The render runs on its own thread — it loads its own plugin instances and
    /// works faster than real time, but "faster than real time" is still seconds
    /// for a long piece, and the panel stays usable meanwhile. The path is asked
    /// for first: a file dialog belongs on the GUI thread.
    fn export_wav(&mut self) {
        let plans = self.build_plans();
        if plans.is_empty() {
            self.status = "Nothing to export — add a note to a row first.".to_string();
            return;
        }
        let stem = project::sanitize_name(&self.project_name);
        let mut dialog = rfd::FileDialog::new()
            .set_title("Export the track composition as WAV")
            .add_filter("WAV audio", &["wav"])
            .set_file_name(format!("{stem}.wav"));
        if let Some(dir) = &self.project_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else { return };

        let (sample_rate, channels) = player::default_export_format();
        let shown = crate::file_label(&path);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let msg = match player::render_offline(plans, sample_rate, channels) {
                Ok((samples, loaded, total)) => {
                    let secs = samples.len() as f64 / (sample_rate * channels as f64).max(1.0);
                    match crate::audio::write_wav_i16(
                        &path,
                        &samples,
                        channels as u16,
                        sample_rate as u32,
                    ) {
                        Ok(()) if loaded == total => format!(
                            "Exported {:.1} s to {} ({} row(s), {:.0} Hz).",
                            secs,
                            crate::file_label(&path),
                            loaded,
                            sample_rate
                        ),
                        // Say which rows are missing from the file rather than
                        // let a silent instrument be discovered on playback.
                        Ok(()) => format!(
                            "Exported {:.1} s to {} — only {loaded} of {total} row(s) \
                             loaded, the rest are missing from the file.",
                            secs,
                            crate::file_label(&path)
                        ),
                        Err(e) => format!("Export failed: {e}"),
                    }
                }
                Err(e) => format!("Export failed: {e}"),
            };
            let _ = tx.send(msg);
        });
        self.status = format!("Exporting to {shown}… (rendering offline)");
        self.export = Some(rx);
    }

    /// The optional name for the next row, beside the button that makes it.
    ///
    /// Both Add Track Row buttons are served by the same box: there is one name
    /// being typed at a time, whichever end of the list the row is going on, so
    /// a name typed at the bottom is still there if the row is added at the top.
    /// `salt` only keeps the two boxes' widget state apart.
    fn new_row_name_ui(&mut self, ui: &mut egui::Ui, salt: &'static str) {
        ui.label("name");
        ui.add(
            egui::TextEdit::singleline(&mut self.new_row_name)
                .id_salt(salt)
                .desired_width(150.0)
                .hint_text("(a UUID)"),
        )
        .on_hover_text(
            "What to call the row this button makes. Optional: a row left \
             unnamed is named after a UUID, which is unique without anyone \
             having to think of anything.\n\nThe box empties into the row it \
             names, so the next row gets a UUID again — and any row can be \
             renamed afterwards, on the row itself.",
        );
    }

    /// What the space bar does: stop whatever the transport is doing — a
    /// play, a Play still loading, a take — or, when it is doing nothing, play.
    pub fn toggle_transport(&mut self, ctx: &egui::Context) {
        if self.player.is_some() || self.preparing.is_some() || self.recording.is_some() {
            self.stop_playback();
        } else {
            self.start_playback(ctx);
        }
    }

    fn stop_playback(&mut self) {
        // The take is closed first: the transport is the clock it is placed on,
        // so it cannot be read after the player has gone.
        let recorded = self.finish_recording();
        self.live_sent = None;
        // Dropping the receiver abandons a Play that is still loading: the
        // thread finishes on its own and drops the instances it made, which is
        // exactly where that work belongs — not here.
        let was_preparing = self.preparing.take().is_some();
        if self.player.take().is_some() || was_preparing {
            self.status = "Stopped.".to_string();
        }
        // What was recorded is the news, not that the transport stopped.
        if let Some(msg) = recorded {
            self.status = msg;
        }
    }

    /// Where the transport is, in grid units, while it is running. The frame
    /// containing it is the one lit up in each lane.
    fn playhead_units(&self) -> Option<f64> {
        let player = self.player.as_ref()?;
        Some(player.position_secs() / self.secs_per_unit())
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.reconcile_tracks();
        // A finished composition stops itself, so the transport does not sit on
        // "playing" over silence.
        // A Play whose rows have finished loading starts here, so the button
        // press and the first sound are separate frames and the window never
        // stops painting between them.
        self.poll_preparation();
        if self.player.as_ref().is_some_and(CompositionPlayer::is_finished) {
            // Before the player goes: a take is placed on the transport's clock.
            let recorded = self.finish_recording();
            self.player = None;
            self.live_sent = None;
            self.status = recorded.unwrap_or_else(|| "Finished.".to_string());
        }
        // Anything the user changed while a repeat is running goes in at the
        // next loop point.
        self.push_live_edits();
        // A finished export reports itself. A dropped sender means the render
        // thread died without a word, which is still an answer the user needs.
        if let Some(rx) = &self.export {
            match rx.try_recv() {
                Ok(msg) => {
                    self.status = msg;
                    self.export = None;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.status = "Export failed: the render stopped unexpectedly.".to_string();
                    self.export = None;
                }
            }
        }

        let tracks = self.registry.list();

        ui.horizontal_wrapped(|ui| {
            if ui
                .button("➕ Add Track Row")
                .on_hover_text("A new lane at the top of the list, playing the first available track")
                .clicked()
            {
                self.add_row_at(0);
            }
            self.new_row_name_ui(ui, "new_row_name_top");
            // One switch for all of them, next to the button that makes them.
            // It shows what is true — unticked the moment any row is off — and
            // a click makes every row agree with it, which is how a part is
            // soloed by hand and how the rest come back afterwards.
            let mut all = self.rows.iter().all(|r| r.enabled);
            if ui
                .add_enabled(
                    !self.rows.is_empty(),
                    egui::Checkbox::new(&mut all, "Enable all"),
                )
                .on_hover_text(
                    "Play every row.\n\nUnticked while any row is switched off; \
                     ticking it turns them all back on, and unticking it silences \
                     the composition without touching a note of it.",
                )
                .changed()
            {
                for row in &mut self.rows {
                    row.enabled = all;
                }
                self.status = if all {
                    "Every row enabled.".to_string()
                } else {
                    "Every row switched off — nothing will play.".to_string()
                };
            }
            if tracks.is_empty() {
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(
                        "No tracks yet — create one under Tracks, or publish a subtrack \
                         from Resynthesis with “Add as Track”.",
                    )
                    .color(egui::Color32::from_rgb(220, 180, 120)),
                );
            }
        });
        ui.add_space(6.0);
        self.project_ui(ui);

        if !self.rows.is_empty() {
            ui.add_space(6.0);
            self.block_ui(ui);
            ui.add_space(4.0);
            self.lanes_ui(ui, &tracks);
            // The same button again, where the list ends: a composition is
            // built downwards, and reaching back to the top to add the next
            // lane — then dragging nothing, since rows do not move — is the
            // long way round.
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button("➕ Add Track Row")
                    .on_hover_text(
                        "A new lane at the end of the list, playing the first available track",
                    )
                    .clicked()
                {
                    self.add_row();
                }
                self.new_row_name_ui(ui, "new_row_name_bottom");
            });
        }

        ui.add_space(8.0);
        ui.separator();
        self.transport_ui(ui);
    }

    /// Take whatever the Save/Load buttons asked for, if anything. The app calls
    /// this after drawing and performs the request — see [`ProjectRequest`].
    pub fn take_request(&mut self) -> Option<ProjectRequest> {
        self.request.take()
    }

    /// The composition as it would be saved. `source_of` maps a row's registry
    /// id onto the source to record for it, which only the app can decide: it
    /// knows which tracks are LeSynth, where their grids went, and where a
    /// custom VST was loaded from.
    pub fn to_project(
        &self,
        name: &str,
        mut source_of: impl FnMut(Option<u64>) -> project::TrackSource,
    ) -> project::Project {
        project::Project {
            name: name.to_string(),
            tempo_bpm: self.tempo_bpm,
            repeat_from: self.repeat_from,
            repeat_to: self.repeat_to,
            rows: self
                .rows
                .iter()
                .map(|row| project::ProjectRow {
                    name: row.name.clone(),
                    track_name: row
                        .track_id
                        .and_then(|id| self.registry.name_of(id))
                        .or_else(|| row.missing.as_ref().map(|s| s.describe()))
                        .unwrap_or_default(),
                    // An unresolved row round-trips the source it was looking
                    // for; anything else asks the app what its track is now.
                    source: match &row.missing {
                        Some(src) => src.clone(),
                        None => source_of(row.track_id),
                    },
                    gain: row.gain,
                    lead: row.lead,
                    enabled: row.enabled,
                    autosave: row.autosave,
                    autoscroll: row.autoscroll,
                    items: row.items.clone(),
                })
                .collect(),
        }
    }

    /// Replace the composition with a loaded project. `resolved` is one entry
    /// per row, in order: the registry id the app managed to bind that row's
    /// source to, or `None` if it could not be found.
    pub fn apply_project(
        &mut self,
        project: &project::Project,
        dir: PathBuf,
        resolved: &[Option<u64>],
    ) {
        self.stop_playback();
        self.tempo_bpm = project.tempo_bpm;
        self.repeat_from = project.repeat_from;
        self.repeat_to = project.repeat_to;
        self.project_name = project.name.clone();
        self.project_dir = Some(dir);
        self.rows.clear();
        for (i, prow) in project.rows.iter().enumerate() {
            let id = self.next_row_id;
            self.next_row_id += 1;
            let track_id = resolved.get(i).copied().flatten();
            let mut row = Row::new(id, track_id);
            // A project written before rows had names leaves this empty, and the
            // row keeps the UUID it was just given: unnamed rows are named, not
            // left blank, however old the file they came out of.
            if !prow.name.is_empty() {
                row.set_name(prow.name.clone());
            }
            row.gain = prow.gain;
            row.lead = prow.lead;
            row.enabled = prow.enabled;
            row.autosave = prow.autosave;
            row.autoscroll = prow.autoscroll;
            // Nothing to bind means the source is gone. Keep what it was so the
            // row can say so and the user knows what to replace.
            row.missing = match (track_id, &prow.source) {
                (None, project::TrackSource::None) => None,
                (None, src) => Some(src.clone()),
                (Some(_), _) => None,
            };
            for item in &prow.items {
                let item_id = row.next_item_id;
                row.next_item_id += 1;
                row.items.push(Item { id: item_id, ..*item });
            }
            self.rows.push(row);
        }
        let missing = self.rows.iter().filter(|r| r.missing.is_some()).count();
        self.status = match missing {
            0 => format!("Loaded {} — {} row(s).", project.name, self.rows.len()),
            n => format!(
                "Loaded {} — {} row(s), {n} with a missing source: pick a new one \
                 in the row's select box.",
                project.name,
                self.rows.len()
            ),
        };
    }

    /// Which rows want their grid re-exported on the next save, as registry ids.
    pub fn autosave_track_ids(&self) -> Vec<u64> {
        self.rows
            .iter()
            .filter(|r| r.autosave)
            .filter_map(|r| r.track_id)
            .collect()
    }

    /// Record where a save actually landed, so the next one goes to the same
    /// folder rather than asking again.
    pub fn set_project_dir(&mut self, dir: PathBuf, name: String) {
        self.project_dir = Some(dir);
        self.project_name = name;
    }

    pub fn set_status(&mut self, status: String) {
        self.status = status;
    }

    /// Name, Save and Load. The name is the folder's name and the manifest's,
    /// so there is one thing to type and no separate "save as" dialog to keep in
    /// step with it: saving under a new name writes a new folder beside the old.
    fn project_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Project");
            ui.add(
                egui::TextEdit::singleline(&mut self.project_name)
                    .desired_width(180.0)
                    .hint_text("Untitled"),
            )
            .on_hover_text(
                "The project's name — also the folder it is saved in and the \
                 .rsyn file inside it.\n\nSaving under a new name writes a new \
                 folder next to the old one and leaves that one untouched.",
            );

            let stem = project::sanitize_name(&self.project_name);
            if ui
                .button("💾 Save Project")
                .on_hover_text(format!(
                    "Write {stem}/{stem}.{ext}, with a .lsft beside it for every \
                     LeSynth Fourier track a row plays.\n\nShortcut: Ctrl+S.",
                    ext = project::EXTENSION
                ))
                .clicked()
            {
                self.request_save(&stem);
            }
            if ui
                .button("📂 Load Project")
                .on_hover_text("Open a .rsyn project (or an old .gmstn one) — this replaces the rows below.")
                .clicked()
            {
                if let Some(file) = rfd::FileDialog::new()
                    .add_filter("ReSynth Studio project", &[project::EXTENSION, project::LEGACY_EXTENSION])
                    .add_filter("All files", &["*"])
                    .pick_file()
                {
                    self.request = Some(ProjectRequest::Load { file });
                }
            }
            if let Some(dir) = &self.project_dir {
                ui.label(
                    egui::RichText::new(crate::file_label(dir))
                        .small()
                        .color(egui::Color32::from_gray(130)),
                );
            }
        });
    }

    /// What Ctrl+S does: exactly what the Save Project button does.
    pub fn save_project(&mut self) {
        let stem = project::sanitize_name(&self.project_name);
        self.request_save(&stem);
    }

    /// Where to save. An existing project saves in place unless the name has
    /// changed, in which case the new name gets its own folder beside it; a
    /// project that has never been saved asks where to put its folder.
    fn request_save(&mut self, stem: &str) {
        let existing = self.project_dir.clone().filter(|d| {
            d.file_name().is_some_and(|n| n == std::ffi::OsStr::new(stem))
        });
        let parent = match &existing {
            Some(dir) => return self.request = Some(ProjectRequest::Save {
                dir: dir.clone(),
                name: stem.to_string(),
            }),
            None => self
                .project_dir
                .as_ref()
                .and_then(|d| d.parent().map(PathBuf::from)),
        };
        let parent = match parent {
            Some(p) => Some(p),
            None => rfd::FileDialog::new()
                .set_title("Where should the project folder go?")
                .pick_folder(),
        };
        let Some(parent) = parent else { return };
        self.request = Some(ProjectRequest::Save {
            dir: parent.join(stem),
            name: stem.to_string(),
        });
    }

    /// Write the project back where it already is, asking nothing.
    ///
    /// This is what a change made *in* a saved project uses — a rename, so far.
    /// It is deliberately narrower than the Save button: a project that has
    /// never been saved, or one whose name has been typed over since, is left
    /// alone rather than made to answer a folder dialog nobody opened. Nothing
    /// is lost by that — the Save button is still there, and it is the one that
    /// decides where a folder goes.
    ///
    /// A save the user has already asked for this frame is not overwritten
    /// either: the queue holds one request, and theirs is the one that named a
    /// folder.
    fn save_in_place(&mut self) {
        if self.request.is_some() {
            return;
        }
        let stem = project::sanitize_name(&self.project_name);
        let saved_here = self
            .project_dir
            .clone()
            .filter(|d| d.file_name().is_some_and(|n| n == std::ffi::OsStr::new(&stem)));
        if let Some(dir) = saved_here {
            self.request = Some(ProjectRequest::Save { dir, name: stem });
        }
    }

    /// One strip per row: the head on the left, the chain of frames scrolling on
    /// the right.
    /// The block bar — what is selected across the rows, and everything that can
    /// be done to it.
    ///
    /// It sits above the lanes rather than in each row's head because a block is
    /// the one thing in this panel that is *not* a property of one row: it is
    /// one selection spanning as many rows as were clicked, and the operations
    /// on it act on all of them at once.
    fn block_ui(&mut self, ui: &mut egui::Ui) {
        // Resolved once, before the buttons borrow the panel: it is what tells
        // them whether there is anything to act on, and what the bar reports.
        let block = self.block();
        let has = block.is_some();
        let (rows, frames, window) = match &block {
            Some(b) => (b.rows.len(), b.frames(), b.window()),
            None => (0, 0, 0),
        };
        let clip = self
            .clip
            .as_ref()
            .map(|c| format!("clipboard: {} frame(s), {}", c.frames(), span_label(c.window)));
        let mut clone_now = false;
        egui::Frame::new()
            .fill(egui::Color32::from_gray(26))
            .inner_margin(egui::Margin::same(6))
            .corner_radius(4.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Block").strong().color(SEL_STROKE));
                    if ui
                        .button("▣ All")
                        .on_hover_text("Select every frame in every row")
                        .clicked()
                    {
                        self.select_all();
                    }
                    if ui
                        .add_enabled(has, egui::Button::new("↔ Span Rows"))
                        .on_hover_text(
                            "Take the same stretch of time out of every row.\n\n\
                             Mark a phrase on the row you can hear it in and press this: \
                             every note that begins inside it, on every other track, \
                             joins the block. A section is a stretch of time, not a count \
                             of frames — no two rows spend it on the same number of notes.",
                        )
                        .clicked()
                    {
                        self.span_rows();
                    }
                    if ui
                        .add_enabled(has, egui::Button::new("✖ Clear"))
                        .on_hover_text("Select nothing")
                        .clicked()
                    {
                        self.clear_block();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(has, egui::Button::new("🗐 Copy"))
                        .on_hover_text(
                            "Hold the block, with the silence in front of each row's part \
                             of it, so what is pasted keeps the shape that was taken",
                        )
                        .clicked()
                    {
                        self.copy_block();
                    }
                    if ui
                        .add_enabled(clip.is_some(), egui::Button::new("📋 Paste at End"))
                        .on_hover_text(
                            "Put the copied block back at the end of the composition, on \
                             the rows it came from and all of them at once.\n\n\
                             Every row is padded up to the end of the longest one, so a \
                             block that sounded together is pasted sounding together.",
                        )
                        .clicked()
                    {
                        self.paste_block();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(has, egui::Button::new("🔁 Clone"))
                        .on_hover_text(
                            "Repeat the block in place — each copy directly behind the \
                             last, on every row at once, with everything that followed \
                             moved along by the same amount on every row.\n\n\
                             A copy takes exactly as long as the block's window, so the \
                             tracks stay in step however many times it is pressed. The \
                             selection moves onto the copy, so pressing it again repeats \
                             the repeat.",
                        )
                        .clicked()
                    {
                        clone_now = true;
                    }
                    egui::ComboBox::from_id_salt("clone_times")
                        .width(64.0)
                        .selected_text(format!("× {}", self.clone_times))
                        .show_ui(ui, |ui| {
                            for n in 1..=MAX_NUM {
                                ui.selectable_value(&mut self.clone_times, n, format!("× {n}"));
                            }
                        })
                        .response
                        .on_hover_text("How many copies one press of Clone makes");
                    ui.separator();
                    ui.label(
                        egui::RichText::new(if has {
                            format!(
                                "{frames} frame(s) on {rows} row(s) · {}",
                                span_label(window)
                            )
                        } else {
                            "nothing selected — press ☐ on a note frame".to_string()
                        })
                        .color(if has {
                            SEL_STROKE
                        } else {
                            egui::Color32::from_gray(140)
                        }),
                    );
                    if let Some(clip) = clip {
                        ui.separator();
                        ui.label(egui::RichText::new(clip).color(egui::Color32::from_gray(150)));
                    }
                });
            });
        // Outside the closure: cloning rewrites the rows the bar was drawn from.
        if clone_now {
            self.clone_block(self.clone_times);
        }
    }

    fn lanes_ui(&mut self, ui: &mut egui::Ui, tracks: &[(u64, String)]) {
        let mut remove_row: Option<usize> = None;
        // Deferred like the removal, and for the same reason: the status line
        // belongs to the panel, and the rows are borrowed for the whole loop.
        let mut renamed: Option<String> = None;
        // Which space's align button was pressed — the row, the space on it
        // (`None` for the lead) and which edge of the cue — applied after the
        // loop, which has every row borrowed.
        let mut align: Option<(usize, Option<usize>, AlignTo)> = None;
        // The row the one selected note is on, if exactly one is: the align
        // buttons on every other row's spaces are live, and only those.
        let cue_row = self.single_selected().map(|(idx, _)| idx);
        // One width for every head, from the width there is — less the row
        // frame's margins, which are inside what the lanes are given.
        let head_w = ((ui.available_width() - 12.0) * HEAD_FRAC).clamp(HEAD_W_MIN, HEAD_W_MAX);
        let playhead = self.playhead_units();
        // What a grid unit is worth in seconds, for the frames that have to turn
        // a stretch of a recording into a length.
        let spu = self.secs_per_unit();
        // Resolved before the loop: `self.rows` is borrowed mutably inside it,
        // and the registry is what knows a drum track from an instrument.
        let percussion: Vec<bool> = self
            .rows
            .iter()
            .map(|r| r.track_id.is_some_and(|id| self.registry.is_percussion(id)))
            .collect();
        // And which rows play an audio file rather than an instrument — with how
        // long that file runs, which is the range of the start slider on its
        // frames. `None` is an ordinary instrument row.
        let wav: Vec<Option<f32>> = self
            .rows
            .iter()
            .map(|r| r.track_id.and_then(|id| self.registry.wav_secs(id)))
            .collect();

        for (idx, row) in self.rows.iter_mut().enumerate() {
            let row_id = row.id;
            // The transport reaches only the lanes that are playing: a row that
            // is switched off has no sounding frame to light and nothing to
            // follow, whatever its own boxes say.
            let row_playhead = row.lane_playhead(playhead);
            ui.push_id(row_id, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_gray(30))
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal_top(|ui| {
                            // ── Row head ──────────────────────────────────
                            ui.allocate_ui_with_layout(
                                egui::vec2(head_w, ROW_H),
                                egui::Layout::top_down(egui::Align::Min),
                                |ui| {
                                    // Exactly `head_w`, however short or long
                                    // its contents: this is what lines the
                                    // lanes up.
                                    ui.set_min_width(head_w);
                                    ui.set_max_width(head_w);
                                    ui.horizontal(|ui| {
                                        // The row's own name, and the form that
                                        // changes it. The box holds a draft
                                        // while it is being typed in, and the
                                        // draft becomes the name the moment the
                                        // box is done with: Enter, the button
                                        // beside it, or simply clicking away.
                                        // Leaving the box is how most people
                                        // finish typing, and a name that only
                                        // counted if you found the button is a
                                        // name the project would not have.
                                        let typed = ui
                                            .add(
                                                egui::TextEdit::singleline(&mut row.name_edit)
                                                    .id_salt(("row_name", row_id))
                                                    .desired_width(head_w - HEAD_BUTTON_W)
                                                    .hint_text("row name"),
                                            )
                                            .on_hover_text(format!(
                                                "This row is called “{}”.\n\nType another \
                                                 name and press Enter, or just click away \
                                                 — either way the row takes it, and a \
                                                 saved project is written again with it. \
                                                 Escape puts the box back.\n\nA row's \
                                                 name is for reading — nothing is looked \
                                                 up by it.",
                                                row.name
                                            ));
                                        // One ending covers both: the Enter
                                        // that gives focus up, and the click
                                        // somewhere else that takes it away.
                                        // Escape ends the box too, and is the
                                        // one ending that means "forget it" —
                                        // so the draft goes back to the name
                                        // rather than over it.
                                        let done = typed.lost_focus();
                                        let escaped =
                                            ui.input(|i| i.key_pressed(egui::Key::Escape));
                                        if done && escaped {
                                            row.name_edit = row.name.clone();
                                        }
                                        let finished = done && !escaped;
                                        // The button is live only while the box
                                        // holds a name that would change
                                        // something, so it says whether there is
                                        // an edit waiting on it.
                                        let pending = {
                                            let want = row.name_edit.trim();
                                            !want.is_empty() && want != row.name
                                        };
                                        let pressed = ui
                                            .add_enabled(pending, egui::Button::new("✏"))
                                            .on_hover_text(
                                                "Rename this row to what is in the box — \
                                                 the same as pressing Enter in it, or \
                                                 clicking away from it.\n\nGreyed out \
                                                 while the box holds the name the row \
                                                 already has, or nothing at all: a row \
                                                 always has a name, and the UUID it was \
                                                 given is better than a blank.",
                                            )
                                            .clicked();
                                        if (pressed || finished) && row.commit_name() {
                                            renamed = Some(row.name.clone());
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        // A row whose loaded source is missing
                                        // says so in the box itself, and says
                                        // what it was: picking anything from the
                                        // list is how it gets repaired.
                                        let label = match (&row.missing, row.track_id) {
                                            (Some(want), _) => {
                                                format!("⚠ missing: {}", want.describe())
                                            }
                                            (None, Some(id)) => tracks
                                                .iter()
                                                .find(|(t, _)| *t == id)
                                                .map(|(_, n)| n.clone())
                                                .unwrap_or_else(|| "— no track —".to_string()),
                                            (None, None) => "— no track —".to_string(),
                                        };
                                        // Salt = `row_id` (so two rows never
                                        // share a popup) + the track count. An
                                        // `egui::Area` measures itself only on
                                        // the first pass a given id is shown and
                                        // then caps the Ui, so the ScrollArea
                                        // clips the list and it measures the same
                                        // again — a box first opened with two
                                        // tracks stayed two tall. A new count is
                                        // a new Area, measured afresh.
                                        //
                                        // Cut to the head's width: a source's
                                        // name is often a file name plus a
                                        // UUID, and the box would otherwise
                                        // grow to all of it. The full name is
                                        // in the list, which is as wide as it
                                        // needs to be.
                                        let combo_w = head_w - HEAD_BUTTON_W;
                                        let full = label.clone();
                                        ui.allocate_ui(
                                            egui::vec2(combo_w, ui.spacing().interact_size.y),
                                            |ui| {
                                        egui::ComboBox::from_id_salt(("track", row_id, tracks.len()))
                                            .width(combo_w)
                                            .truncate()
                                            .selected_text(label)
                                            .show_ui(ui, |ui| {
                                                if tracks.is_empty() {
                                                    ui.label("— no track —");
                                                }
                                                for (id, name) in tracks {
                                                    if ui
                                                        .selectable_label(
                                                            row.missing.is_none()
                                                                && row.track_id == Some(*id),
                                                            name,
                                                        )
                                                        .clicked()
                                                    {
                                                        row.track_id = Some(*id);
                                                        // Repaired: the row is
                                                        // an ordinary row again.
                                                        row.missing = None;
                                                    }
                                                }
                                            })
                                            .response
                                            .on_hover_text(full);
                                            },
                                        );
                                        if ui
                                            .button("🗑")
                                            .on_hover_text("Remove this row")
                                            .clicked()
                                        {
                                            remove_row = Some(idx);
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        if ui
                                            .button("➕ Add Note")
                                            .on_hover_text(
                                                "Append a note frame and, behind it, a \
                                                 space frame for the silence that \
                                                 follows.\n\nThe new frame repeats the \
                                                 last one in the row — pitch, note \
                                                 length and space — so a run of the \
                                                 same thing takes one click each.",
                                            )
                                            .clicked()
                                        {
                                            row.add_note(if percussion[idx] {
                                                DEFAULT_DRUM_PITCH
                                            } else {
                                                DEFAULT_PITCH
                                            });
                                        }
                                        ui.label("Gain");
                                        ui.spacing_mut().slider_width = 76.0;
                                        ui.add(
                                            egui::Slider::new(&mut row.gain, 0.0..=GAIN_MAX)
                                                .logarithmic(true)
                                                .smallest_positive(GAIN_MIN_TRAVEL)
                                                .fixed_decimals(2)
                                                .show_value(true),
                                        )
                                        .on_hover_text(
                                            "How loud this row plays in the mix. 1.00 is \
                                             the track's own level; above it the row is \
                                             boosted, so a quiet drum can hold its own \
                                             against a loud lead.\n\nThe travel is \
                                             logarithmic, a fader rather than a ruler: \
                                             unity sits mid-way and the top is 50.00, \
                                             enough to lift a source analysed thirty \
                                             decibels down.\n\nA boost that takes the \
                                             row past full scale is limited rather than \
                                             clipped: the peaks are turned down smoothly \
                                             instead of being cut flat, so it stops \
                                             getting louder without turning to fuzz.",
                                        );
                                        ui.checkbox(&mut row.autosave, "auto")
                                            .on_hover_text(
                                                "Re-export this row's LeSynth Fourier \
                                                 grid into the project folder every time \
                                                 the project is saved, so the project \
                                                 carries what you last edited.\n\n\
                                                 Off keeps the .lsft already in the \
                                                 folder, which pins the sound as it was \
                                                 when it was first written.",
                                            );
                                    });
                                    ui.horizontal(|ui| {
                                        // Every frame on the row, in one press.
                                        // Derived from the block, so clicking a
                                        // frame unticks it by itself.
                                        let mut all = row.all_selected();
                                        if ui
                                            .add_enabled(
                                                !row.items.is_empty(),
                                                egui::Checkbox::new(&mut all, "select all"),
                                            )
                                            .on_hover_text(
                                                "Take every frame on this row into the \
                                                 block — the whole part, however long \
                                                 it is.\n\n\
                                                 Rows are selected independently, so \
                                                 ticking this on several rows blocks \
                                                 all of them at once, which is what \
                                                 “⧉ Clone” repeats. Untick it, or click \
                                                 any frame's ☐, to let the block go.",
                                            )
                                            .changed()
                                        {
                                            if all {
                                                row.select_all();
                                            } else {
                                                row.sel = None;
                                            }
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut row.enabled, "Enabled")
                                            .on_hover_text(
                                                "Play this row. On for every new row — \
                                                 a lane exists to be heard.\n\n\
                                                 Off takes the row out of the transport \
                                                 and out of an exported wav, and leaves \
                                                 everything on it where it is: the \
                                                 notes, the block, the track it plays. \
                                                 That is how a part is listened to \
                                                 without the two rows drowning it, and \
                                                 how they come back afterwards.\n\n\
                                                 Switched while a repeat is running, it \
                                                 takes effect on the next Play: the \
                                                 row's voice is a loaded plugin, not a \
                                                 level.",
                                            );
                                        ui.add_enabled(
                                            row.enabled,
                                            egui::Checkbox::new(&mut row.autoscroll, "⏵ follow"),
                                        )
                                            .on_hover_text(
                                                "While the transport is playing — \
                                                 including a take — keep this lane \
                                                 scrolled to the frame that is \
                                                 sounding, centred.\n\n\
                                                 On by default: a row of any length is \
                                                 wider than the window, so a lane that \
                                                 does not follow shows the first few \
                                                 seconds and nothing of the rest. Turn \
                                                 it off to keep the lane where you put \
                                                 it while you edit one part of a long \
                                                 row as it plays.\n\n\
                                                 A row that is not Enabled is not \
                                                 playing, so there is nothing to \
                                                 follow: this is ignored until it is \
                                                 switched back on.",
                                            );
                                    });
                                },
                            );

                            // ── The chain of frames ───────────────────────
                            egui::ScrollArea::horizontal()
                                .id_salt("lane")
                                // Room for the frames *and* the scrollbar under
                                // them, which would otherwise clip the cards.
                                .max_height(ROW_H + 12.0)
                                .show(ui, |ui| {
                                    let pressed = Self::chain_ui(
                                        ui,
                                        row,
                                        row_playhead,
                                        percussion[idx],
                                        wav[idx],
                                        spu,
                                        cue_row.is_some_and(|c| c != idx),
                                    );
                                    if let Some((space, to)) = pressed {
                                        align = Some((idx, space, to));
                                    }
                                });
                        });
                    });
            });
            ui.add_space(4.0);
        }

        if let Some((idx, space, to)) = align {
            self.align_space(idx, space, to);
        }
        if let Some(name) = renamed {
            self.status = format!("Renamed a row to {name}.");
            // A name the project cannot be closed and reopened with is not the
            // row's name yet as far as the file is concerned, so the rename is
            // written straight back to where the project already lives.
            self.save_in_place();
        }
        if let Some(idx) = remove_row {
            if idx < self.rows.len() {
                let row = self.rows.remove(idx);
                self.status = format!("Removed row {}.", row.label());
            }
        }
    }

    /// A row's frames, left to right in play order — the row's lead space, then
    /// each item as its note frame followed by its tied space frame. Editing a
    /// frame's length or pitch is immediate; everything after it simply shifts,
    /// because a frame's position is nothing but the sum of the lengths before
    /// it.
    ///
    /// `can_align` lights the align buttons on every space; the one pressed is
    /// handed back — the space (`None` for the lead) and the cue's edge — for
    /// the panel to apply, since the cue is on another row.
    #[allow(clippy::too_many_arguments)]
    fn chain_ui(
        ui: &mut egui::Ui,
        row: &mut Row,
        playhead: Option<f64>,
        percussion: bool,
        wav: Option<f32>,
        spu: f64,
        can_align: bool,
    ) -> Option<(Option<usize>, AlignTo)> {
        // Follow the transport only while there *is* one for this lane — a row
        // that is switched off is handed no playhead at all — and only if the
        // lane was asked to: scrolling a lane every frame is exactly what stops
        // the user scrolling it themselves.
        let follow = row.autoscroll && playhead.is_some();
        // Deferred: removing a frame rewrites the list this loop walks.
        let mut pending_delete: Option<usize> = None;
        // Likewise the block: the row's selection is read for every frame in
        // the loop, so a click changes it once the row has been drawn.
        let mut pending_select: Option<(u64, bool)> = None;
        let mut pending_align: Option<(Option<usize>, AlignTo)> = None;
        let sel = row.sel_range();
        // Taken before the frames are drawn, so an edit made in one frame moves
        // the ones behind it only on the next pass — never mid-loop.
        let starts = row.starts();
        let row_id = row.id;

        ui.horizontal_top(|ui| {
            ui.set_min_height(ROW_H);
            if row.items.is_empty() {
                ui.label(
                    egui::RichText::new("Empty row — press “➕ Add Note”.")
                        .color(egui::Color32::from_gray(120)),
                );
            }
            // The lead space, ahead of the first note. It belongs to the row,
            // so it stays put when that note is deleted and leads the next one
            // instead. Drawn only when there is a note for it to lead.
            if !row.items.is_empty() {
                let lead = Self::frame_ui(
                    ui,
                    row_id,
                    LEAD_SPACE_ID,
                    Frame::Space,
                    &mut row.lead,
                    // The lead belongs to the row, not to an item: there is no
                    // block it could be part of, and nothing sounds in it.
                    FrameState { sounding: false, percussion, selected: false, can_align },
                );
                if let Some(to) = lead.align {
                    pending_align = Some((None, to));
                }
            }
            for (idx, (item, start)) in row.items.iter_mut().zip(starts).enumerate() {
                let selected = sel.is_some_and(|(a, b)| idx >= a && idx <= b);
                let sounding = playhead
                    .is_some_and(|p| p >= start as f64 && p < (start + item.dur.units()) as f64);
                // A wav row's note has no pitch to pick: the file plays at the
                // pitch it was recorded at. What it has instead is a start —
                // where in the file this note begins.
                let note = Self::frame_ui(
                    ui,
                    row_id,
                    item.id,
                    match wav {
                        Some(len_secs) => Frame::Wav {
                            start: &mut item.start,
                            len_secs,
                            spu,
                        },
                        None => Frame::Note(&mut item.pitch),
                    },
                    &mut item.dur,
                    FrameState { sounding, percussion, selected, can_align: false },
                );
                if note.deleted {
                    pending_delete = Some(idx);
                }
                if let Some(extend) = note.select {
                    pending_select = Some((item.id, extend));
                }
                // Centred, not merely "into view": a frame scrolled to the edge
                // is one the next note leaves again immediately, and centring
                // shows what is coming as well as what is playing.
                if follow && sounding {
                    ui.scroll_to_rect(note.rect, Some(egui::Align::Center));
                }
                // The space tied to it, drawn right behind it and carrying no
                // delete button of its own: it leaves only with its note.
                let space = Self::frame_ui(
                    ui,
                    row_id,
                    item.id,
                    Frame::Space,
                    &mut item.space,
                    // A space is never *sounding* — it is the silence — but it
                    // is in the block whenever its note is: they are one item.
                    FrameState { sounding: false, percussion, selected, can_align },
                );
                if let Some(to) = space.align {
                    pending_align = Some((Some(idx), to));
                }
            }
        });

        if let Some((id, extend)) = pending_select {
            row.select_item(id, extend);
        }
        if let Some(idx) = pending_delete {
            row.delete_item(idx);
        }
        pending_align
    }

    /// One frame. Reports whether its delete button was pressed and where it
    /// landed, which is what a lane following the transport scrolls to.
    ///
    /// [`Frame`] decides which frame this is — the blue note, the blue note of a
    /// row that plays a file (the same frame with no pitch box on it), or the
    /// amber space, which carries the same length boxes but *no delete button*:
    /// a space leaves only with its note, and the row's lead space never leaves
    /// at all.
    ///
    /// `percussion` names the notes after what they hit on a drum kit.
    fn frame_ui(
        ui: &mut egui::Ui,
        row_id: u64,
        id: u64,
        kind: Frame<'_>,
        dur: &mut Duration,
        state: FrameState,
    ) -> FrameOut {
        let FrameState { sounding, percussion, selected, can_align } = state;
        let space = matches!(kind, Frame::Space);
        let (fill, stroke, header) = match (space, sounding) {
            (true, _) => (
                SPACE_FILL,
                egui::Color32::from_rgb(186, 146, 84),
                egui::Color32::from_rgb(226, 190, 130),
            ),
            (false, true) => (
                NOTE_SOUNDING_FILL,
                PLAYHEAD,
                egui::Color32::from_rgb(245, 225, 220),
            ),
            (false, false) => (
                NOTE_FILL,
                egui::Color32::from_rgb(110, 150, 200),
                egui::Color32::from_rgb(200, 218, 240),
            ),
        };

        // In the block: the same card, lit, and ringed so the block reads as one
        // shape across the rows. A frame that is *sounding* keeps the
        // playhead's colours — where the transport is beats what is selected,
        // and the border says the rest.
        let (fill, stroke, stroke_w) = match (selected, sounding) {
            (false, _) => (fill, stroke, 1.0),
            (true, true) => (fill, SEL_STROKE, 2.0),
            (true, false) => (
                if space { SPACE_SEL_FILL } else { NOTE_SEL_FILL },
                SEL_STROKE,
                2.0,
            ),
        };

        let mut deleted = false;
        let mut select = None;
        let mut align = None;
        let card_w = kind.width();
        let placed = ui.allocate_ui_with_layout(
            egui::vec2(card_w, CARD_H),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                egui::Frame::new()
                    .fill(fill)
                    .stroke(egui::Stroke::new(stroke_w, stroke))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::same(4))
                    .show(ui, |ui| {
                        let inner_w = card_w - 10.0;
                        ui.set_width(inner_w);
                        ui.set_min_height(CARD_H - 10.0);
                        ui.spacing_mut().item_spacing.y = 3.0;
                        ui.spacing_mut().button_padding = egui::vec2(6.0, 2.0);
                        // egui's own row height, which the card's height was
                        // measured against: the app's is a full-padding button
                        // tall, and would push the card past CARD_H.
                        ui.spacing_mut().interact_size.y =
                            egui::style::Spacing::default().interact_size.y;

                        // Header: what the frame is and how long. The note frame
                        // also carries the delete button, laid out from the
                        // right so it keeps its corner and a long title
                        // truncates instead of pushing the card wider than its
                        // neighbours.
                        ui.horizontal(|ui| {
                            // The block's handle, on the left of the header: a
                            // note *and its space* are one item, so the box is
                            // on the note and the space follows it in and out
                            // of the block — the same rule the delete button
                            // already follows.
                            if !space
                                && ui
                                    .add(
                                        egui::Button::new(if selected { "☑" } else { "☐" })
                                            .small()
                                            .frame(false),
                                    )
                                    .on_hover_text(
                                        "Select this frame into the block.\n\n\
                                         Click another frame on the row to take everything \
                                         between them; click one that is already in the \
                                         block to drop it. Shift-click moves the end last \
                                         clicked, which is how a block shrinks. Rows are \
                                         selected independently, so a block can span as \
                                         many as you like — or mark one row and press \
                                         “↔ Span Rows”.",
                                    )
                                    .clicked()
                            {
                                select = Some(ui.input(|i| i.modifiers.shift));
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if !space
                                        && ui
                                            .add(egui::Button::new("✖").small().frame(false))
                                            .on_hover_text(
                                                "Delete this note and the space tied to it",
                                            )
                                            .clicked()
                                    {
                                        deleted = true;
                                    }
                                    // Align with the one selected note on
                                    // another row — right to left, so end then
                                    // start reads start, end on the card.
                                    if space {
                                        for to in [AlignTo::End, AlignTo::Start] {
                                            if ui
                                                .add_enabled(
                                                    can_align,
                                                    egui::Button::new(to.icon())
                                                        .small()
                                                        .frame(false),
                                                )
                                                .on_hover_text(to.hint())
                                                .on_disabled_hover_text(format!(
                                                    "{}\n\nSelect exactly one note, on \
                                                     another row, first.",
                                                    to.title()
                                                ))
                                                .clicked()
                                            {
                                                align = Some(to);
                                            }
                                        }
                                    }
                                    // Fit the length to the file. Only a wav
                                    // note has a length that *something* is the
                                    // right answer for: an instrument sounds for
                                    // as long as it is asked to, but a recording
                                    // has an end, and a note that stops short of
                                    // it or rings on past it into silence is
                                    // almost never what was meant.
                                    if let Frame::Wav { start, len_secs, spu } = &kind {
                                        let left = (*len_secs - **start).max(0.0) as f64;
                                        let fitted = Duration::closest_to(left, *spu);
                                        if ui
                                            .add(egui::Button::new("⏱").small().frame(false))
                                            .on_hover_text(fit_hint(left, fitted, *spu))
                                            .clicked()
                                        {
                                            *dur = fitted;
                                        }
                                    }
                                    // The lead space says so, because it is the
                                    // one space that is not the silence after
                                    // some note and does not leave with one.
                                    let title = match &kind {
                                        Frame::Note(p) => {
                                            format!("{} · {}", note_label(**p, percussion), dur.label())
                                        }
                                        // No pitch to name it by, so it is named
                                        // after what it plays: the file, whole.
                                        Frame::Wav { .. } => format!("wav · {}", dur.label()),
                                        Frame::Space if id == LEAD_SPACE_ID => {
                                            format!("lead · {}", dur.label())
                                        }
                                        Frame::Space => format!("space · {}", dur.label()),
                                    };
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(title).small().color(header),
                                        )
                                        .truncate(),
                                    );
                                },
                            );
                        });

                        match kind {
                            Frame::Note(pitch) => {
                                // The popup measures itself once per id and keeps
                                // that width forever, so the id carries
                                // `percussion` — a row moved from a synth to a
                                // drum kit gets a fresh popup instead of one
                                // sized for bare pitches.
                                egui::ComboBox::from_id_salt(("pitch", row_id, id, percussion))
                                    .width(inner_w)
                                    .height(260.0)
                                    .selected_text(note_label(*pitch, percussion))
                                    .show_ui(ui, |ui| {
                                        for p in PITCH_MIN..=PITCH_MAX {
                                            ui.selectable_value(
                                                pitch,
                                                p,
                                                note_label(p, percussion),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(if percussion {
                                        "Which drum to hit. The name is General MIDI's, \
                                         which is the map nearly every kit follows; the \
                                         note is what is actually sent."
                                    } else {
                                        "Pitch"
                                    });
                            }
                            // Where in the file this note starts. In the pitch
                            // box's place, because it answers the same question
                            // for a recording that a pitch answers for an
                            // instrument: *what* is sounding here.
                            //
                            // A slider to find it by ear, and a value to set it
                            // exactly: the track moves it in tens of
                            // milliseconds, which is the wrong side of a beat,
                            // and the number beside it can be dragged a
                            // millisecond at a time or typed outright.
                            Frame::Wav { start, len_secs, .. } => {
                                // A file that has been replaced by a shorter one
                                // must not leave a note starting past its end.
                                let last = len_secs.max(0.001);
                                *start = start.clamp(0.0, last);
                                // The value box takes the rest of the card; the
                                // default slider width would overrun it.
                                ui.spacing_mut().slider_width = inner_w - START_VALUE_W;
                                ui.add(
                                    egui::Slider::new(start, 0.0..=last)
                                        .fixed_decimals(3)
                                        .step_by(0.001)
                                        .suffix("s"),
                                )
                                .on_hover_text(
                                    "Where in the file this note starts.\n\nA take does \
                                     not begin on its first sample — there is room, a \
                                     breath, a stick click before the downbeat — so this \
                                     is what puts the sound itself on the beat rather \
                                     than the silence in front of it. Drag the slider to \
                                     find it, then drag the number for milliseconds, or \
                                     double-click it to type one.\n\nThe note plays from \
                                     here for its own length; past the end of the file it \
                                     falls silent.",
                                );
                            }
                            Frame::Space => {}
                        }

                        // Whole notes and fraction both reach zero, and for a
                        // space that is the point: a 0-length space is a
                        // placeholder that keeps its frame without putting any
                        // silence in the row.
                        egui::ComboBox::from_id_salt(("wholes", row_id, id, space))
                            .width(inner_w)
                            .height(260.0)
                            .selected_text(format!("{} whole", dur.wholes))
                            .show_ui(ui, |ui| {
                                for w in 0..=MAX_WHOLES {
                                    ui.selectable_value(&mut dur.wholes, w, format!("{w} whole"));
                                }
                            })
                            .response
                            .on_hover_text("Whole notes — the whole part of the length");

                        // The nominator, drawn over the fraction it counts, so
                        // the two boxes read down the card as the fraction they
                        // make: `× 3` over `1/8` is three eighths.
                        //
                        // Disabled while there is no fraction: it would multiply
                        // nothing, and a box that says `× 3` over a length that
                        // is a whole number of whole notes is simply a lie about
                        // what is playing.
                        ui.add_enabled_ui(dur.frac != Fraction::None, |ui| {
                            egui::ComboBox::from_id_salt(("num", row_id, id, space))
                                .width(inner_w)
                                .height(260.0)
                                .selected_text(format!("× {}", dur.num))
                                .show_ui(ui, |ui| {
                                    for n in 1..=MAX_NUM {
                                        ui.selectable_value(&mut dur.num, n, format!("× {n}"));
                                    }
                                })
                                .response
                                .on_hover_text(
                                    "How many of the fraction below — the nominator. \
                                     Three of a 1/8 is 3/8, the dotted quarter the \
                                     fraction box cannot name on its own, and it counts \
                                     for a space exactly as it does for a note.\n\n\
                                     Needs a fraction to count: pick one below first.",
                                );
                        });

                        egui::ComboBox::from_id_salt(("frac", row_id, id, space))
                            .width(inner_w)
                            .selected_text(dur.frac.label())
                            .show_ui(ui, |ui| {
                                for f in Fraction::ALL {
                                    ui.selectable_value(&mut dur.frac, f, f.label());
                                }
                            })
                            .response
                            .on_hover_text(
                                "Fractional part of the length, down to a 1/256 — taken \
                                 as many times as the nominator above says, and added to \
                                 the whole notes",
                            );

                        // The boxes are free to leave the nominator counting a
                        // fraction that is no longer there; the length is not.
                        dur.canonicalise();
                    });
            },
        );
        FrameOut {
            deleted,
            select,
            align,
            rect: placed.response.rect,
        }
    }

    /// Where a repeat starts and where it stops — both a length from the
    /// beginning of the composition.
    ///
    /// **Why lengths and not positions.** It is the only vocabulary the panel
    /// has: a row is a chain of lengths, nothing in it sits at a timestamp, and
    /// a repeat marked in seconds would be a second way of saying when.
    ///
    /// **Why both from the same end.** So the two can be read against each
    /// other. `1/2` and `3/4` say which is in front of which at a glance; a stop
    /// counted back from the end has to be subtracted from a total nobody has in
    /// their head before it can be compared with the start sitting next to it.
    ///
    /// **Why a line of its own.** Six select boxes do not fit in a transport bar
    /// that already carries three buttons, a tempo and two recording controls —
    /// and the pair belong together, read as one window. Read apart, a start gets
    /// moved onto a stop nobody looked at, which is the mistake the seconds at
    /// the end of the line are there to catch.
    fn repeat_window_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            // Greyed out during a take, exactly as Repeat is: a take is one
            // pass, so there is no window for it to run in.
            let recording = self.recording.is_some();
            // A horizontal region inside the greying-out, or every box in it
            // sits a few pixels below the labels — see `length_boxes`.
            ui.add_enabled_ui(!recording, |ui| {
                ui.horizontal(|ui| {
                    ui.label("🔁 Repeat from").on_hover_text(
                        "How far into the composition a repeat starts — a length \
                         from the beginning, in the same three boxes a note is \
                         written with.\n\nLeave it at zero to loop from the top. \
                         Play with Repeat ticked starts right here, the first \
                         pass included.",
                    );
                    length_boxes(ui, "repeat_from", &mut self.repeat_from);

                    ui.add_space(8.0);
                    ui.label("to").on_hover_text(
                        "Where a repeat stops — also a length from the beginning \
                         of the composition, so the two boxes can be read against \
                         each other rather than off opposite ends.\n\nZero means \
                         the end of the composition, which is what a repeat loops \
                         to until it is told otherwise; a stop past the end is \
                         held there, since out that far there is only silence to \
                         repeat. Releases ring on across the wrap either way.",
                    );
                    length_boxes(ui, "repeat_to", &mut self.repeat_to);
                });
            });

            // What the two lengths actually came to, which is the whole reason
            // they are next to each other: a window is two boxes and a
            // composition, and only this says what the three of them make.
            ui.add_space(10.0);
            let spu = self.secs_per_unit();
            let (from, to) = self.repeat_span_units();
            let (from_secs, to_secs) = (from as f64 * spu, to as f64 * spu);
            if self.repeat_span_ignored() {
                ui.label(
                    egui::RichText::new(format!(
                        "⚠ nothing between them — looping all {:.1} s",
                        to_secs - from_secs
                    ))
                    .color(RECORD),
                )
                .on_hover_text(
                    "The start is not in front of the stop: a window of no length \
                     is not a short loop, it is a transport with nothing to play. \
                     The whole composition is looped until one of the two boxes \
                     leaves room for the other.",
                );
            } else {
                // An untouched stop box says nothing on its own, so the reading
                // says what it came to mean. It is the one thing about the line
                // that has to be learned rather than read, and this is where it
                // is cheapest to learn it.
                let end = if self.repeat_to.units() == 0 { ", to the end" } else { "" };
                ui.label(
                    egui::RichText::new(format!(
                        "loops {from_secs:.1} – {to_secs:.1} s ({:.1} s{end})",
                        to_secs - from_secs
                    ))
                    .color(egui::Color32::from_gray(160)),
                )
                .on_hover_text(
                    "Where the two lengths land in the composition as it stands \
                     now. It follows the tempo, and a stop left at zero follows \
                     the music: write another bar and the loop reaches it.\n\n\
                     A window can be changed while a repeat is playing — like a \
                     note edit, it comes in at the next wrap.",
                );
            }
        });
    }

    /// Play / stop, tempo, and where the transport currently is.
    fn transport_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let playing = self.player.is_some();
            let loading = self.preparing.is_some();
            let recording = self.recording.is_some();
            if ui
                .add_enabled(!playing && !loading, egui::Button::new("▶ Play"))
                .on_hover_text("Play every row from the start.\n\nSpace plays, and stops again.")
                .clicked()
            {
                self.start_playback(ui.ctx());
            }
            if ui
                .add_enabled(
                    !playing && !loading && !recording,
                    egui::Button::new(
                        egui::RichText::new("⏺ Record & Play Once").color(RECORD),
                    ),
                )
                .on_hover_text(
                    "Play the composition through exactly once and record your MIDI \
                     keyboard over it.\n\nWhat you play is appended as new rows when the \
                     pass ends (or when you press Stop), with every note and every \
                     silence rounded onto the note value in “Round”. Notes that sound \
                     *together* go into a row each — a row plays one note at a time — \
                     so a chord comes out as a row per voice; a key merely let go of \
                     late stays in the line and is trimmed.\n\nTo hear yourself while \
                     you play, open the track's editor: the keyboard feeds it as usual, \
                     and recording takes a copy rather than the events themselves.",
                )
                .clicked()
            {
                self.start_recording(ui.ctx());
            }
            // Stop also abandons a Play still loading its rows, and ends a take.
            if ui
                .add_enabled(playing || loading || recording, egui::Button::new("■ Stop"))
                .on_hover_text("Stop — or press Space.")
                .clicked()
            {
                self.stop_playback();
            }
            // Live: ticking this mid-play loops from the next pass (or straight
            // away, if the composition is already into its release tail).
            // Not during a take: "once" is the whole of what Record promises,
            // and a second pass would be recorded over the first.
            let mut repeat = self.repeat.load(Ordering::Relaxed);
            if ui
                .add_enabled_ui(!recording, |ui| ui.checkbox(&mut repeat, "🔁 Repeat"))
                .inner
                .on_hover_text(
                    "Play the composition over and over. By default the loop is \
                     the whole written length; the two lengths on the line below \
                     trim it at either end.\n\nReleases ring on across the loop, \
                     and it can be switched on and off while it plays — untick it \
                     and the current pass is the last one.\n\nUnavailable while \
                     recording: a take is one pass.",
                )
                .changed()
            {
                self.repeat.store(repeat, Ordering::Relaxed);
            }
            let exporting = self.export.is_some();
            if ui
                .add_enabled(!exporting, egui::Button::new("🎵 Export WAV…"))
                .on_hover_text(
                    "Render the whole composition offline and write it as a \
                     16-bit .wav — every row, mixed exactly as the transport \
                     plays it.\n\nThe render runs in the background; the panel \
                     stays usable.",
                )
                .clicked()
            {
                self.export_wav();
            }
            ui.add_space(12.0);
            ui.label("Tempo");
            ui.add(
                egui::DragValue::new(&mut self.tempo_bpm)
                    .range(20.0..=300.0)
                    .speed(1.0)
                    .suffix(" BPM"),
            )
            .on_hover_text("A beat is a quarter note; every note length scales with this.");

            // What a take is rounded onto, and where it lands. Both belong here
            // rather than in a dialog: they are read the moment a take ends, so
            // the rounding can still be reconsidered while the rows load.
            ui.add_space(12.0);
            ui.label("Round");
            egui::ComboBox::from_id_salt("record_round")
                .width(72.0)
                // All nine note values at once: the default popup height cuts
                // the list off at about six and scrolls the rest out of sight.
                .height(9.0 * 44.0)
                .selected_text(record::round_label(self.round_units))
                .show_ui(ui, |ui| {
                    for (label, units) in record::ROUND_CHOICES {
                        ui.selectable_value(&mut self.round_units, units, label);
                    }
                })
                .response
                .on_hover_text(
                    "The note value a recorded take is rounded onto — every note \
                     start, every note length and every silence snaps to the nearest \
                     multiple of it.\n\nA position is rounded exactly: a silence that \
                     no single length can express is carried on placeholder frames \
                     behind the space, so nothing after it drifts. A note's *length* \
                     takes the nearest length the two select boxes can express, which \
                     is the one thing that can come out a fraction short.",
                );
            ui.label("into");
            let record_label = self
                .record_track
                .and_then(|id| self.registry.name_of(id))
                .unwrap_or_else(|| "— no track —".to_string());
            egui::ComboBox::from_id_salt(("record_track", self.registry.list().len()))
                .width(150.0)
                .selected_text(record_label)
                .show_ui(ui, |ui| {
                    if self.registry.list().is_empty() {
                        ui.label("— no track —");
                    }
                    for (id, name) in self.registry.list() {
                        ui.selectable_value(&mut self.record_track, Some(id), name);
                    }
                })
                .response
                .on_hover_text("The track the recorded rows will play.");

            ui.add_space(12.0);
            let length_secs = self.end_units() as f64 * self.secs_per_unit();
            if loading {
                ui.label(
                    egui::RichText::new("⏳ loading rows…").color(PLAYHEAD),
                );
            }
            // A take says how much of it there is so far. Counting note-ons in
            // the tap rather than draining it keeps the one conversion of a take
            // in one place — where it ends.
            if let Some(rec) = &self.recording {
                let notes = rec
                    .tap
                    .lock()
                    .map_or(0, |t| t.iter().filter(|(_, m)| is_note_on(*m)).count());
                ui.label(
                    egui::RichText::new(format!("⏺ {notes} note(s)")).color(RECORD),
                );
            }
            match &self.player {
                // While looping, the length that means anything is the loop's,
                // not "last note plus release".
                Some(p) if self.repeat.load(Ordering::Relaxed) => {
                    // Both ends when the loop is a window, so the readout says
                    // what is being repeated rather than only how far through
                    // the composition the playhead is.
                    let (start, end) = (p.loop_start_secs(), p.loop_secs());
                    let of = if start > 0.0 {
                        format!("{start:.1} – {end:.1} s")
                    } else {
                        format!("{end:.1} s")
                    };
                    ui.label(
                        egui::RichText::new(format!("🔁 {:.1} s / {of}", p.position_secs()))
                            .color(PLAYHEAD),
                    )
                }
                Some(p) => ui.label(
                    egui::RichText::new(format!(
                        "▶ {:.1} s / {:.1} s",
                        p.position_secs(),
                        p.total_secs
                    ))
                    .color(PLAYHEAD),
                ),
                None => ui.label(
                    egui::RichText::new(format!("{length_secs:.1} s"))
                        .color(egui::Color32::from_gray(160)),
                ),
            };
        });
        ui.add_space(4.0);
        self.repeat_window_ui(ui);
        ui.add_space(4.0);
        ui.label(egui::RichText::new(&self.status).color(egui::Color32::from_gray(170)));

        // While playing, repaint fast enough for the sounding frame to light up
        // on time; otherwise stay idle like the rest of the app. An export has
        // nothing to animate but must not sit finished on screen unnoticed, so
        // it only asks for the occasional frame.
        if self.player.is_some() || self.preparing.is_some() || self.recording.is_some() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(16));
        } else if self.export.is_some() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
        }
    }
}

/// The three select boxes that name a length — whole notes, how many of a
/// fraction, and which fraction — laid along a line. Answers whether they
/// changed it.
///
/// The same three a frame carries, in the same order and saying the same
/// things, because a length is a length wherever it is asked for: anyone who
/// can write a dotted quarter into a row can write one into a repeat's start
/// without learning a second way to say it. Only the layout differs — a card
/// stacks them, a toolbar lays them out — which is why this is a function of
/// its own rather than a call into [`ComposerPanel::frame_ui`]. The one
/// difference is the whole-note count, which goes up to [`MAX_REPEAT_WHOLES`]
/// here and so is typed or dragged rather than picked off a list.
///
/// Change is reported by comparing the length rather than by asking the boxes:
/// a combo answers `changed()` for the *button*, not for the popup (see the
/// track select box above), and picking the value that was already there is not
/// a change to anything but the widget.
fn length_boxes(ui: &mut egui::Ui, salt: &'static str, dur: &mut Duration) -> bool {
    let before = *dur;
    // A value to drag or type rather than a list to pick from: a repeat window
    // reaches thousands of whole notes into a long piece.
    ui.add(
        egui::DragValue::new(&mut dur.wholes)
            .range(0..=MAX_REPEAT_WHOLES)
            .speed(0.25)
            .suffix(" whole"),
    )
    .on_hover_text("Whole notes — the whole part of the length. Drag or type.");
    // The greying-out gets a horizontal region of its own: a scope nested in a
    // line inherits the *line's* full height and centres what is in it there, so
    // one wrapped straight round a box drops it a few pixels below its
    // neighbours — which on a row of six boxes reads as a staircase.
    ui.add_enabled_ui(dur.frac != Fraction::None, |ui| {
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt((salt, "num"))
                .width(60.0)
                .height(260.0)
                .selected_text(format!("× {}", dur.num))
                .show_ui(ui, |ui| {
                    for n in 1..=MAX_NUM {
                        ui.selectable_value(&mut dur.num, n, format!("× {n}"));
                    }
                })
                .response
                .on_hover_text(
                    "How many of the fraction beside it — three of a 1/8 is 3/8, \
                     the dotted quarter the fraction box cannot name on its \
                     own.\n\nNeeds a fraction to count: pick one first.",
                );
        });
    });
    egui::ComboBox::from_id_salt((salt, "frac"))
        .width(72.0)
        .selected_text(dur.frac.label())
        .show_ui(ui, |ui| {
            for f in Fraction::ALL {
                ui.selectable_value(&mut dur.frac, f, f.label());
            }
        })
        .response
        .on_hover_text(
            "Fractional part of the length, down to a 1/256 — taken as many times \
             as the nominator says, and added to the whole notes",
        );
    // The boxes are free to leave the nominator counting a fraction that is no
    // longer there; the length is not.
    dur.canonicalise();
    *dur != before
}

/// Widget id of a row's lead space frame. Item ids count up from zero and are
/// handed out one at a time, so this cannot collide with one.
const LEAD_SPACE_ID: u64 = u64::MAX;

/// What drawing one frame reported back to its lane.
/// Which of a row's three frames [`ComposerPanel::frame_ui`] is drawing.
///
/// A note carries a pitch, unless the row plays an audio file: a wav track has
/// the pitch it was recorded at, so its frames offer a length and nothing else.
/// Both are the blue note frame, and both leave with their delete button; a
/// space is the amber one, which has none.
enum Frame<'a> {
    Note(&'a mut u8),
    /// A note on a wav row: where in the file it starts, how long that file runs
    /// — which is the range the start can be moved over — and seconds per grid
    /// unit, which is the only frame that needs the tempo: fitting a length to a
    /// stretch of a recording is a conversion out of seconds.
    Wav { start: &'a mut f32, len_secs: f32, spu: f64 },
    Space,
}

impl Frame<'_> {
    /// How wide the card is. Only the wav note is different, and only because of
    /// the slider on it — see [`WAV_CARD_W`].
    fn width(&self) -> f32 {
        match self {
            Frame::Wav { .. } => WAV_CARD_W,
            _ => CARD_W,
        }
    }
}

/// Which edge of the one selected note a space's align button lines the note
/// behind the space up with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AlignTo {
    /// Where the cue note starts: the two notes come in together.
    Start,
    /// Where the cue note ends, *its space included*: this note comes in on
    /// whatever would have followed it.
    End,
}

impl AlignTo {
    fn icon(self) -> &'static str {
        match self {
            AlignTo::Start => "⤵",
            AlignTo::End => "⏭",
        }
    }

    /// What the edge is called in the status line.
    fn edge(self) -> &'static str {
        match self {
            AlignTo::Start => "start",
            AlignTo::End => "end",
        }
    }

    fn title(self) -> &'static str {
        match self {
            AlignTo::Start => "Align with start",
            AlignTo::End => "Align to end",
        }
    }

    fn hint(self) -> String {
        let what = match self {
            AlignTo::Start => "starts where the selected note starts",
            AlignTo::End => "starts where the selected note ends, with its space included",
        };
        format!(
            "{}: set this space so the note behind it {what}.\n\n\
             Live only while exactly one note is selected, and on another row. If \
             this space already starts past that point, nothing changes and the \
             note stays selected; it is unticked once the space is set.",
            self.title()
        )
    }
}

/// What a frame is besides its length — the four things it is drawn against,
/// travelling together because three bare `bool`s at a call site say nothing
/// about which is which.
#[derive(Clone, Copy)]
struct FrameState {
    /// The transport is inside this note.
    sounding: bool,
    /// Its row plays a drum kit, so its pitches are named after drums.
    percussion: bool,
    /// It is in the selected block.
    selected: bool,
    /// Exactly one note is selected, on another row, so a space's align buttons
    /// have a cue to align with.
    can_align: bool,
}

struct FrameOut {
    /// Its delete button was pressed.
    deleted: bool,
    /// Its select box was pressed, and whether shift was down — which extends
    /// the block onto this frame instead of starting a new one.
    select: Option<bool>,
    /// A space's align button was pressed, and which edge of the cue it was.
    align: Option<AlignTo>,
    /// Where it landed, so a lane that follows the transport can scroll the
    /// sounding frame into the middle of itself.
    rect: egui::Rect,
}

/// Colour of the transport's position readout, and of the frame sounding under
/// it.
const PLAYHEAD: egui::Color32 = egui::Color32::from_rgb(240, 120, 110);

/// Colour of the record button and of a take's note count.
const RECORD: egui::Color32 = egui::Color32::from_rgb(235, 90, 90);

/// Whether a MIDI message starts a note. The zero-velocity note-on most
/// keyboards send instead of a note-off does not.
fn is_note_on(msg: [u8; 3]) -> bool {
    msg[0] & 0xF0 == 0x90 && msg[2] > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUARTER: i64 = UNITS_PER_WHOLE / 4;
    const EIGHTH: i64 = UNITS_PER_WHOLE / 8;

    /// `(pitch, note length, space length)` per item.
    fn row_with(items: &[(u8, Duration, Duration)]) -> Row {
        let mut row = Row::new(0, None);
        for (pitch, dur, space) in items {
            let id = row.next_item_id;
            row.next_item_id += 1;
            row.items.push(Item {
                id,
                pitch: *pitch,
                start: 0.0,
                dur: *dur,
                space: *space,
            });
        }
        row
    }

    /// Shorthand for a length with no whole-note part.
    const fn frac(f: Fraction) -> Duration {
        Duration::new(0, f)
    }

    /// A panel of several rows, each with its own id — which is what a paste
    /// looks a row up by.
    fn panel_with_rows(rows: &[&[(u8, Duration, Duration)]]) -> ComposerPanel {
        let mut panel = ComposerPanel::new(TrackRegistry::default(), crate::midi::new_midi_taps());
        for items in rows {
            let id = panel.next_row_id;
            panel.next_row_id += 1;
            let mut row = row_with(items);
            row.id = id;
            panel.rows.push(row);
        }
        panel
    }

    /// Where each row's notes start, in units — spacer frames, which have no
    /// length, left out. What every block edit is measured by: two rows are in
    /// step exactly to the extent that these agree.
    fn note_starts(panel: &ComposerPanel) -> Vec<Vec<i64>> {
        panel
            .rows
            .iter()
            .map(|row| {
                let mut at = row.lead_units();
                let mut out = Vec::new();
                for item in &row.items {
                    if item.dur.units() > 0 {
                        out.push(at);
                    }
                    at += item.total_units();
                }
                out
            })
            .collect()
    }

    /// A panel with one row of `items`, playing track `track`.
    fn panel_with(items: &[(u8, Duration, Duration)], track: Option<u64>) -> ComposerPanel {
        let mut panel = ComposerPanel::new(TrackRegistry::default(), crate::midi::new_midi_taps());
        let mut row = row_with(items);
        row.track_id = track;
        panel.rows.push(row);
        panel
    }

    /// Adding a note repeats the frame before it **whole** — pitch, note length
    /// and space, whole notes, nominator and fraction alike. Building a row means
    /// repeating something far more often than changing it — a hi-hat line, a run
    /// on one drum, a rhythm of one length — so none of the seven select boxes
    /// should have to be touched again for the next frame.
    #[test]
    fn a_new_note_carries_on_from_the_one_before_it() {
        let mut row = Row::new(0, None);
        // The first note of a row has nothing to follow, so it takes the
        // defaults: middle-of-the-road lengths and the caller's pitch.
        row.add_note(DEFAULT_DRUM_PITCH);
        assert_eq!(row.items[0].pitch, DEFAULT_DRUM_PITCH);
        assert_eq!(row.items[0].dur, Duration::new(0, Fraction::Quarter));
        assert_eq!(row.items[0].space, Duration::new(0, Fraction::Eighth));

        // Change every box of it — the nominator of the note and of the space
        // included — and everything added after follows all of them.
        row.items[0].pitch = 42;
        row.items[0].dur = Duration::with_num(2, 3, Fraction::Sixteenth);
        row.items[0].space = Duration::with_num(1, 5, Fraction::Half);
        row.add_note(DEFAULT_DRUM_PITCH);
        row.add_note(DEFAULT_DRUM_PITCH);
        for i in [1, 2] {
            assert_eq!(row.items[i].pitch, 42, "the pitch was not carried over");
            assert_eq!(
                row.items[i].dur,
                Duration::with_num(2, 3, Fraction::Sixteenth),
                "the note length was not carried over"
            );
            assert_eq!(
                row.items[i].space,
                Duration::with_num(1, 5, Fraction::Half),
                "the space was not carried over"
            );
            assert_eq!(row.items[i].dur.num, 3, "the note's nominator was not carried over");
            assert_eq!(row.items[i].space.num, 5, "the space's nominator was not carried over");
        }

        // It follows the *last* frame, not the first.
        row.items[2].pitch = 46;
        row.items[2].dur = Duration::with_num(0, 7, Fraction::ThirtySecond);
        row.add_note(DEFAULT_DRUM_PITCH);
        assert_eq!(row.items[3].pitch, 46);
        assert_eq!(row.items[3].dur, Duration::with_num(0, 7, Fraction::ThirtySecond));

        // Every frame still gets an id of its own, or two of them would share
        // their widgets.
        let ids: std::collections::HashSet<u64> = row.items.iter().map(|i| i.id).collect();
        assert_eq!(ids.len(), row.items.len(), "two frames share an id");

        // A row emptied of notes starts from the defaults again — one of each
        // fraction, the nominator that changes nothing.
        row.items.clear();
        row.add_note(DEFAULT_PITCH);
        assert_eq!(row.items[0].pitch, DEFAULT_PITCH);
        assert_eq!(row.items[0].dur, Duration::new(0, Fraction::Quarter));
        assert_eq!(row.items[0].space, Duration::new(0, Fraction::Eighth));
        assert_eq!(row.items[0].dur.num, 1);
        assert_eq!(row.items[0].space.num, 1);
    }

    /// Pressing Play must not wait for the plugins.
    ///
    /// The Composer loads one instance per row, and that is hundreds of
    /// milliseconds — seconds, for a big synth or a long composition. Doing it
    /// in the button's own call freezes the window until the first sound, which
    /// is what "the transport lags" means. So Play hands the loading to a thread
    /// and returns: the test is that it comes back with no transport yet, and
    /// that one turns up on a later frame.
    #[test]
    fn pressing_play_hands_the_loading_to_a_thread() {
        let registry = TrackRegistry::default();
        let plugin = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("internal_plugins")
            .join("liblesynth_fourier.so");
        if !plugin.exists() {
            println!("no internal plugin — nothing to load");
            return;
        }
        let id = registry.add(
            "LeSynth",
            plugin,
            Some(crate::vst::class_ids::FOURIER_SYNTH),
            true,
            None,
        );
        let mut panel = ComposerPanel::new(registry, crate::midi::new_midi_taps());
        let mut row = row_with(&[(60, frac(Fraction::Quarter), frac(Fraction::Eighth))]);
        row.track_id = Some(id);
        panel.rows.push(row);

        let pressed = std::time::Instant::now();
        panel.start_playback(&egui::Context::default());
        let button_took = pressed.elapsed();
        assert!(
            panel.preparing.is_some(),
            "Play did not start loading: {}",
            panel.status
        );
        assert!(
            panel.player.is_none(),
            "Play waited for the transport instead of handing it off"
        );
        assert!(
            button_took < std::time::Duration::from_millis(100),
            "the button itself took {button_took:?}"
        );

        // The transport turns up on a later frame, without the button's help.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while panel.player.is_none() && std::time::Instant::now() < deadline {
            panel.poll_preparation();
            if panel.status.starts_with("Playback failed") {
                println!("nothing to play through ({}) — the rest needs a device", panel.status);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(panel.player.is_some(), "the transport never started: {}", panel.status);
        assert!(panel.status.starts_with("Playing"), "status reads {:?}", panel.status);

        panel.stop_playback();
        assert!(panel.player.is_none() && panel.preparing.is_none());
    }

    /// A drum track names its notes after what they hit; an instrument track
    /// leaves them as pitches. The name is *added* to the pitch, never swapped
    /// for it — a kit that is not General MIDI still shows what is being sent.
    #[test]
    fn a_drum_row_labels_its_notes_with_the_drum() {
        assert_eq!(note_label(36, false), "C2");
        assert_eq!(note_label(36, true), "C2 · Bass Drum (Kick)");
        assert_eq!(note_label(42, true), "F#2 · Closed Hi-Hat");
        // Outside the General MIDI percussion range there is nothing to add.
        assert_eq!(note_label(100, true), "E7");
        // And the first note on a drum row is the kick, not middle C's bongo.
        assert_eq!(gm_percussion_name(DEFAULT_DRUM_PITCH), Some("Bass Drum (Kick)"));
        assert_eq!(gm_percussion_name(DEFAULT_PITCH), Some("Hi Bongo"));
    }

    /// What a live edit may and may not change, which is what decides whether the
    /// looping transport can take it or the user has to press Play again.
    ///
    /// The panel compares these snapshots every frame while a repeat runs, so
    /// "nothing changed" has to come out equal — otherwise the transport would be
    /// handed the same composition over and over.
    #[test]
    fn a_live_snapshot_separates_note_edits_from_track_changes() {
        let items = [(60, frac(Fraction::Quarter), frac(Fraction::Eighth))];
        let panel = panel_with(&items, Some(7));
        let before = panel.live_snapshot();
        assert_eq!(
            before.rows,
            panel.live_snapshot().rows,
            "an untouched composition must snapshot equal, or every frame \
             republishes it"
        );

        // A note edit: same rows and tracks, different schedule — the transport
        // can swap this in at the loop point.
        let mut edited = panel_with(&items, Some(7));
        edited.rows[0].items[0].pitch = 64;
        let after = edited.live_snapshot();
        assert_eq!(after.shape, before.shape, "a pitch change is not structural");
        assert_ne!(after.rows, before.rows, "the pitch change was not noticed");

        // Tempo moves every note *and* the loop length.
        let mut faster = panel_with(&items, Some(7));
        faster.tempo_bpm = 180.0;
        let after = faster.live_snapshot();
        assert_eq!(after.shape, before.shape, "tempo is not structural");
        assert_ne!(after.rows, before.rows, "the tempo change did not move the notes");
        assert_ne!(after.span, before.span, "the loop length did not follow");

        // A different track means a different plugin, which an audio callback
        // cannot load: structural, and reported rather than applied.
        let moved = panel_with(&items, Some(8));
        assert_ne!(
            moved.live_snapshot().shape,
            before.shape,
            "pointing a row at another track must count as structural"
        );

        // A row with nothing to play is in neither the plans nor the snapshot.
        let silent = panel_with(&[(60, frac(Fraction::None), frac(Fraction::Eighth))], Some(7));
        assert!(silent.live_snapshot().rows.is_empty());
    }

    /// The requirement: a row opens with a space before its very first note, and
    /// everything the row plays waits for it.
    #[test]
    fn a_row_opens_with_a_space_before_its_first_note() {
        let mut row = row_with(&[
            (60, frac(Fraction::Quarter), frac(Fraction::None)),
            (62, frac(Fraction::Quarter), frac(Fraction::None)),
        ]);
        // Zero by default, so a row still starts at time zero.
        assert_eq!(row.lead.units(), 0);
        assert_eq!(row.starts(), vec![0, QUARTER]);

        row.lead = frac(Fraction::Eighth);
        assert_eq!(
            row.starts(),
            vec![EIGHTH, EIGHTH + QUARTER],
            "the lead space must push every note in the row, not just the first"
        );
        assert_eq!(row.end_units(), EIGHTH + 2 * QUARTER);
        let notes = row.planned_notes(1.0);
        assert_eq!(notes[0].at_secs, EIGHTH as f64);
        assert_eq!(notes[1].at_secs, (EIGHTH + QUARTER) as f64);
    }

    /// The other half of it: deleting the first note must leave the lead space
    /// where it is, leading whichever note is first afterwards. It belongs to
    /// the row, so there is no path that can take it away with a note.
    #[test]
    fn deleting_the_first_note_keeps_the_lead_space_for_the_next_one() {
        let mut row = row_with(&[
            (60, frac(Fraction::Quarter), frac(Fraction::None)),
            (62, frac(Fraction::Half), frac(Fraction::None)),
        ]);
        row.lead = frac(Fraction::Eighth);

        row.delete_item(0);

        assert_eq!(row.lead, frac(Fraction::Eighth), "the lead space went with the note");
        assert_eq!(row.items.len(), 1);
        assert_eq!(row.items[0].pitch, 62);
        assert_eq!(
            row.starts(),
            vec![EIGHTH],
            "the note that is first now must wait through the same lead space"
        );

        // And deleting the last note leaves the lead intact for the next one
        // added, though an empty row has no first note to lead and so no delay.
        row.delete_item(0);
        assert!(row.items.is_empty());
        assert_eq!(row.lead, frac(Fraction::Eighth));
        assert_eq!(row.end_units(), 0, "an empty row has nothing to wait for");
        row.add_note(DEFAULT_PITCH);
        assert_eq!(row.starts(), vec![EIGHTH]);
    }

    /// The requirement: one press of "Add Note" leaves a note *and* the space
    /// behind it.
    #[test]
    fn adding_a_note_appends_the_note_and_a_space_behind_it() {
        let mut row = row_with(&[]);
        row.add_note(DEFAULT_PITCH);
        assert_eq!(row.items.len(), 1);
        assert_eq!(row.items[0].pitch, DEFAULT_PITCH);
        assert!(row.items[0].dur.units() > 0);
        assert!(row.items[0].space.units() > 0);

        row.add_note(DEFAULT_PITCH);
        assert_eq!(row.items.len(), 2);
        // Ids stay unique, so two items never share a widget id.
        assert_ne!(row.items[0].id, row.items[1].id);
    }

    /// The space is tied to its note: there is no way to remove one and keep the
    /// other, and deleting the note takes its space with it.
    #[test]
    fn a_space_can_only_leave_with_the_note_it_is_tied_to() {
        let mut row = row_with(&[
            (60, frac(Fraction::Quarter), frac(Fraction::Half)),
            (64, frac(Fraction::Quarter), frac(Fraction::Eighth)),
        ]);
        assert_eq!(
            row.end_units(),
            QUARTER + UNITS_PER_WHOLE / 2 + QUARTER + EIGHTH
        );

        // Delete the first note: its half-note space goes too, so the row loses
        // both lengths and the second note starts at zero.
        row.delete_item(0);
        assert_eq!(row.items.len(), 1);
        assert_eq!(row.items[0].pitch, 64);
        assert_eq!(row.end_units(), QUARTER + EIGHTH);
        assert_eq!(row.starts(), vec![0]);
    }

    /// A space of length zero is legal — the placeholder case: the frame stays,
    /// the row does not grow, and the next note follows on immediately.
    #[test]
    fn a_space_may_be_zero_length() {
        let spu = 1.0;
        let row = row_with(&[
            (60, frac(Fraction::Quarter), Duration::new(0, Fraction::None)),
            (64, frac(Fraction::Quarter), frac(Fraction::Eighth)),
        ]);
        // The zero space is still an item's space — nothing is dropped …
        assert_eq!(row.items[0].space.units(), 0);
        assert_eq!(row.items[0].space.label(), "0");
        // … and it adds no time: the second note starts as the first one ends.
        let notes = row.planned_notes(spu);
        assert_eq!(notes[1].at_secs, QUARTER as f64);
        assert_eq!(row.end_units(), QUARTER + QUARTER + EIGHTH);
    }

    /// A length is the whole part plus the fractional part, and 1/256 is the
    /// finest value the boxes offer.
    #[test]
    fn a_length_is_its_whole_part_plus_its_fraction() {
        assert_eq!(Duration::new(0, Fraction::None).units(), 0);
        assert_eq!(Duration::new(1, Fraction::None).units(), UNITS_PER_WHOLE);
        assert_eq!(
            Duration::new(1, Fraction::Half).units(),
            UNITS_PER_WHOLE + UNITS_PER_WHOLE / 2
        );
        // A dotted half: 1/2 + 1/4.
        assert_eq!(
            Duration::new(0, Fraction::Half).units() + Duration::new(0, Fraction::Quarter).units(),
            UNITS_PER_WHOLE * 3 / 4
        );
        // Every fraction is a whole number of units, halving down to a 1/256.
        assert_eq!(Fraction::TwoHundredFiftySixth.units(), 1);
        for pair in Fraction::ALL[1..].windows(2) {
            assert_eq!(pair[1].units(), pair[0].units() / 2);
        }
        assert_eq!(Duration::new(2, Fraction::Eighth).label(), "2 + 1/8");
        assert_eq!(Duration::new(0, Fraction::Eighth).label(), "1/8");
        assert_eq!(Duration::new(3, Fraction::None).label(), "3");
    }

    /// The nominator counts the fraction: `3` of a `1/8` is the dotted quarter
    /// the halving fraction box cannot name on its own, and it reads as `3/8`
    /// wherever the length is shown.
    #[test]
    fn a_nominator_counts_the_fraction_it_sits_over() {
        let dotted_quarter = Duration::with_num(0, 3, Fraction::Eighth);
        assert_eq!(dotted_quarter.units(), UNITS_PER_WHOLE * 3 / 8);
        assert_eq!(
            dotted_quarter.units(),
            Duration::new(0, Fraction::Quarter).units()
                + Duration::new(0, Fraction::Eighth).units(),
        );
        assert_eq!(dotted_quarter.label(), "3/8");
        assert_eq!(Duration::with_num(2, 3, Fraction::Eighth).label(), "2 + 3/8");
        assert_eq!(
            Duration::with_num(2, 3, Fraction::Eighth).units(),
            UNITS_PER_WHOLE * 2 + UNITS_PER_WHOLE * 3 / 8,
        );
        // One of a fraction is the fraction, named the way it always was.
        assert_eq!(
            Duration::with_num(0, 1, Fraction::Eighth),
            Duration::new(0, Fraction::Eighth),
        );
        assert_eq!(Duration::with_num(0, 1, Fraction::Eighth).label(), "1/8");
        // Every fraction knows what it is named after, which is what makes the
        // nominator's `3/8` — and the file's token — say the right thing.
        for f in Fraction::ALL {
            let whole = if f == Fraction::None { 0 } else { UNITS_PER_WHOLE };
            assert_eq!(f.units() * f.denom(), whole);
        }
        // It reaches the top of the box without overflowing the grid.
        assert_eq!(
            Duration::with_num(MAX_WHOLES, MAX_NUM, Fraction::Half).units(),
            UNITS_PER_WHOLE * (MAX_WHOLES as i64 + MAX_NUM as i64 / 2),
        );
    }

    /// A nominator with no fraction under it would multiply nothing, so the
    /// select boxes never leave one behind: two lengths that sound the same are
    /// the same value, which is what the file's round trip and every `assert_eq!`
    /// on a `Duration` rest on.
    #[test]
    fn a_nominator_without_a_fraction_is_canonicalised_away() {
        // What the boxes can do: count a fraction, then take the fraction away.
        let mut d = Duration::with_num(2, 5, Fraction::Eighth);
        d.frac = Fraction::None;
        d.canonicalise();
        assert_eq!(d, Duration::new(2, Fraction::None));
        assert_eq!(d.units(), UNITS_PER_WHOLE * 2);
        assert_eq!(d.label(), "2");
        // And a nominator is never zero: a fraction is counted at least once.
        let mut none = Duration::with_num(0, 0, Fraction::Sixteenth);
        assert_eq!(none.units(), UNITS_PER_WHOLE / 16);
        none.num = 0;
        none.canonicalise();
        assert_eq!(none.num, 1);
    }

    /// Frames are played one after another: a note starts where the previous
    /// note's space ended, and a space is silence of exactly its own length.
    #[test]
    fn frames_play_in_sequence_and_spaces_are_the_silences() {
        let spu = 1.0; // one second per unit keeps the arithmetic readable
        let row = row_with(&[
            (60, frac(Fraction::Quarter), frac(Fraction::Eighth)),
            (64, frac(Fraction::Quarter), frac(Fraction::None)),
        ]);
        let notes = row.planned_notes(spu);
        assert_eq!(notes.len(), 2); // the spaces are not played
        assert_eq!(notes[0].at_secs, 0.0);
        assert_eq!(notes[0].dur_secs, QUARTER as f64);
        // The second note waits out the quarter *and* the eighth of silence.
        assert_eq!(notes[1].at_secs, (QUARTER + EIGHTH) as f64);
        assert_eq!(notes[1].pitch, 64);
        assert_eq!(row.end_units(), QUARTER + EIGHTH + QUARTER);

        // A note left at length zero is silence, not a click — but its space
        // still counts, so what follows keeps its place.
        let row = row_with(&[
            (60, frac(Fraction::None), frac(Fraction::Eighth)),
            (62, frac(Fraction::Half), frac(Fraction::None)),
        ]);
        let notes = row.planned_notes(spu);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch, 62);
        assert_eq!(notes[0].at_secs, EIGHTH as f64);
    }

    /// Every row starts at zero, so rows sound together exactly when the lengths
    /// before a note add up the same — that is the only thing that makes a chord
    /// now that nothing is positioned by hand.
    #[test]
    fn equal_lengths_before_a_note_make_it_sound_with_another_row() {
        let spu = 0.5 / UNITS_PER_BEAT as f64; // 120 BPM

        // Two eighths played back to back, then the note.
        let dense = row_with(&[
            (60, frac(Fraction::Eighth), frac(Fraction::None)),
            (62, frac(Fraction::Eighth), frac(Fraction::None)),
            (64, Duration::new(1, Fraction::None), frac(Fraction::None)),
        ]);
        // A note cut short and a long space instead — the same moment, reached
        // by a different route.
        let sparse = row_with(&[
            (59, frac(Fraction::Eighth), frac(Fraction::Eighth)),
            (67, Duration::new(1, Fraction::None), frac(Fraction::None)),
        ]);

        let third = dense.planned_notes(spu)[2].at_secs;
        let second = sparse.planned_notes(spu)[1].at_secs;
        assert_eq!(third, second);
        assert_eq!(third, (EIGHTH + EIGHTH) as f64 * spu);
        // …and a row whose lengths do not add up the same does not join them.
        let off = row_with(&[
            (59, frac(Fraction::Sixteenth), frac(Fraction::None)),
            (67, Duration::new(1, Fraction::None), frac(Fraction::None)),
        ]);
        assert_ne!(off.planned_notes(spu)[1].at_secs, third);
    }

    /// Deleting from the middle closes the gap: everything behind moves earlier
    /// by exactly the note *and* space that went with it.
    #[test]
    fn deleting_a_frame_pulls_the_rest_forward() {
        let mut row = row_with(&[
            (60, frac(Fraction::Quarter), frac(Fraction::None)),
            (62, frac(Fraction::Quarter), frac(Fraction::Quarter)),
            (64, frac(Fraction::Quarter), frac(Fraction::None)),
        ]);
        assert_eq!(row.starts(), vec![0, QUARTER, 3 * QUARTER]);
        row.delete_item(1); // takes its own quarter of silence with it
        assert_eq!(row.items.len(), 2);
        assert_eq!(row.starts(), vec![0, QUARTER]);
        assert_eq!(row.end_units(), 2 * QUARTER);
    }

    #[test]
    fn pitch_names_follow_scientific_notation() {
        assert_eq!(pitch_name(60), "C4");
        assert_eq!(pitch_name(61), "C#4");
        assert_eq!(pitch_name(PITCH_MIN), "C0");
        assert_eq!(pitch_name(PITCH_MAX), "B8");
    }

    fn registry_with(names: &[&str]) -> (TrackRegistry, Vec<u64>) {
        let registry = TrackRegistry::default();
        let ids = names
            .iter()
            .map(|n| {
                registry.add(
                    *n,
                    std::path::PathBuf::from("/nonexistent/plugin.so"),
                    None,
                    false,
                    None,
                )
            })
            .collect();
        (registry, ids)
    }

    /// A row must never point at a track that is gone: the requirement is that
    /// deleting the track behind a row silently re-points it, that an empty
    /// registry leaves the select box on its placeholder, and that a track added
    /// afterwards is adopted.
    #[test]
    fn a_row_follows_the_track_list_as_tracks_come_and_go() {
        let (registry, ids) = registry_with(&["one", "two"]);
        let mut panel = ComposerPanel::new(registry.clone(), crate::midi::new_midi_taps());
        panel.add_row();
        assert_eq!(panel.rows[0].track_id, Some(ids[0]));

        // The track this row plays is deleted → it takes the next available one.
        registry.remove(ids[0]);
        panel.reconcile_tracks();
        assert_eq!(panel.rows[0].track_id, Some(ids[1]));

        // Every track deleted → the placeholder, not a dangling id.
        registry.remove(ids[1]);
        panel.reconcile_tracks();
        assert_eq!(panel.rows[0].track_id, None);

        // A track exists again → the row picks it up.
        let late = registry.add(
            "late",
            std::path::PathBuf::from("/nonexistent/plugin.so"),
            None,
            false,
            None,
        );
        panel.reconcile_tracks();
        assert_eq!(panel.rows[0].track_id, Some(late));
    }

    /// Lay the whole panel out for real (headless egui, no window) in the states
    /// that have layout of their own: no rows, an empty row, a row of note and
    /// space frames, two rows, and a row whose track list has gone empty.
    /// Catches the panics a layout test can catch — duplicate widget ids, bad
    /// rects — which no amount of model testing would.
    #[test]
    fn the_panel_lays_out_in_every_state_without_panicking() {
        let (registry, ids) = registry_with(&["one"]);
        let mut panel = ComposerPanel::new(registry.clone(), crate::midi::new_midi_taps());
        let ctx = egui::Context::default();
        let frame = |panel: &mut ComposerPanel| {
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
            });
        };

        frame(&mut panel); // no rows: just the add button and the transport

        // The transport in its recording state — the note-count readout, the
        // rounding and target select boxes, and Repeat disabled beneath them.
        // Started with no rows, so nothing is loading behind it.
        panel.start_recording(&ctx);
        frame(&mut panel);
        let _ = panel.finish_recording();

        panel.add_row();
        frame(&mut panel); // a row with no frames yet

        panel.rows[0].add_note(DEFAULT_PITCH);
        panel.rows[0].add_note(DEFAULT_PITCH);
        frame(&mut panel);

        // The lead space is drawn too, and must lay out at a real length as
        // well as at its zero default.
        panel.rows[0].lead = Duration::new(MAX_WHOLES, Fraction::TwoHundredFiftySixth);
        frame(&mut panel);

        // Two rows on the same track, which the spec allows. The longest length
        // the boxes can name, and a zero-length placeholder space, are the two
        // extremes a frame has to lay out at.
        panel.add_row();
        panel.rows[1].add_note(DEFAULT_PITCH);
        panel.rows[1].items[0].dur =
            Duration::with_num(MAX_WHOLES, MAX_NUM, Fraction::TwoHundredFiftySixth);
        panel.rows[1].items[0].space = Duration::new(0, Fraction::None);
        frame(&mut panel);

        // A deleted item must not leave a stale widget id behind — neither for
        // its note frame nor for the space that went with it.
        panel.rows[0].delete_item(0);
        frame(&mut panel);

        registry.remove(ids[0]);
        frame(&mut panel);
        assert!(panel.rows.iter().all(|r| r.track_id.is_none()));
    }
    /// A recording is whatever length it is; a length in this panel is a whole
    /// number of note values at the current tempo. The fit is the closest the
    /// boxes can come — and which length that is depends on the tempo, which is
    /// the whole reason the button re-derives it rather than storing an answer.
    #[test]
    fn a_length_is_fitted_to_a_stretch_of_a_recording() {
        // 120 BPM: a beat is half a second, so a whole note is two.
        let spu = 0.5 / UNITS_PER_BEAT as f64;
        assert_eq!(Duration::closest_to(2.0, spu), Duration::new(1, Fraction::None));
        assert_eq!(Duration::closest_to(1.0, spu), Duration::new(0, Fraction::Half));
        assert_eq!(Duration::closest_to(3.0, spu), Duration::new(1, Fraction::Half));
        // Three eighths — the dotted quarter, which only the nominator can name.
        assert_eq!(
            Duration::closest_to(0.75, spu),
            Duration::with_num(0, 3, Fraction::Eighth)
        );
        // The same stretch of file at twice the tempo is twice the note value.
        let fast = 0.25 / UNITS_PER_BEAT as f64;
        assert_eq!(Duration::closest_to(1.0, fast), Duration::new(1, Fraction::None));

        // Ties go to the plainest: half a whole note is `1/2`, not `2/4`.
        assert_eq!(Duration::closest_to(1.0, spu).frac, Fraction::Half);
        assert_eq!(Duration::closest_to(1.0, spu).num, 1);

        // Nothing left of the file is no length at all — but a sliver of it is
        // the shortest note there is, never a frame that plays nothing.
        assert_eq!(Duration::closest_to(0.0, spu), Duration::new(0, Fraction::None));
        assert_eq!(Duration::closest_to(-1.0, spu), Duration::new(0, Fraction::None));
        assert_eq!(
            Duration::closest_to(0.0001, spu),
            Duration::new(0, Fraction::TwoHundredFiftySixth)
        );

        // A recording longer than the boxes can say gets the longest they can.
        assert_eq!(
            Duration::closest_to(1000.0, spu),
            Duration::with_num(MAX_WHOLES, MAX_NUM, Fraction::Half)
        );
    }

    /// What the fit could not name is the number the hint has to show: a file
    /// whose length is not a note value lands short or long, and by how much is
    /// the difference between a loop that holds and one that drifts.
    #[test]
    fn a_fit_that_is_not_exact_says_which_way_it_missed() {
        let spu = 0.5 / UNITS_PER_BEAT as f64;
        // D5.wav: 5.81 s, which no length at 120 BPM names — the nominator stops
        // at sixteen, so the nearest is 2 + 7/8 (5.75 s), 60 ms short.
        let fitted = Duration::closest_to(5.81, spu);
        assert_eq!(fitted, Duration::with_num(2, 7, Fraction::Eighth));
        assert!((fitted.error_secs(5.81, spu) + 0.06).abs() < 1e-6);
        assert!(fit_hint(5.81, fitted, spu).contains("60 ms short"), "{}", fit_hint(5.81, fitted, spu));

        // An exact fit says so rather than showing a rounded zero.
        let exact = Duration::closest_to(2.0, spu);
        assert_eq!(exact.error_secs(2.0, spu), 0.0);
        assert!(fit_hint(2.0, exact, spu).contains("exactly"));
    }

    /// A row playing a wav track has no pitch to pick — the file sounds at the
    /// pitch it was recorded at — so its note frame draws the *start* of the
    /// file in the pitch box's place, then the same three length boxes as every
    /// other frame. The space behind it is untouched: a silence is a silence
    /// whatever sounds around it.
    #[test]
    fn a_wav_row_s_note_frame_offers_a_start_and_no_pitch() {
        let registry = TrackRegistry::default();
        let id =
            registry.add_wav("a whole take.wav", std::path::PathBuf::from("/x/take.wav"), 5.8);
        let mut panel = ComposerPanel::new(registry, crate::midi::new_midi_taps());
        panel.add_row();
        assert_eq!(panel.rows[0].track_id, Some(id));
        panel.rows[0].add_note(DEFAULT_PITCH);
        panel.rows[0].items[0].dur = Duration::with_num(0, 3, Fraction::Eighth);
        panel.rows[0].items[0].space = Duration::new(0, Fraction::Quarter);
        // A start past the top of the file, so the slider is read for what it
        // carries rather than for a zero it would show either way.
        panel.rows[0].items[0].start = 1.25;

        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
        });
        let (mut texts, mut cards) = (Vec::new(), Vec::new());
        fn walk(
            sh: &egui::Shape,
            texts: &mut Vec<(String, egui::Pos2)>,
            cards: &mut Vec<egui::Rect>,
        ) {
            match sh {
                egui::Shape::Text(t) => texts.push((t.galley.text().to_string(), t.pos)),
                egui::Shape::Rect(r) if CARD_FILLS.contains(&r.fill) => cards.push(r.rect),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, texts, cards)),
                _ => {}
            }
        }
        for cs in &out.shapes {
            walk(&cs.shape, &mut texts, &mut cards);
        }

        // No pitch anywhere in the lane — not in a box, not in a header.
        assert!(
            !texts.iter().any(|(t, _)| t == &note_label(DEFAULT_PITCH, false)),
            "a wav row still offers a pitch: {texts:?}"
        );
        // The fit button is on the note and only on the note: a space is not
        // playing a file, so there is nothing for it to be the length of.
        assert_eq!(
            texts.iter().filter(|(t, _)| t == "⏱").count(),
            1,
            "the fit button is not on exactly the one note frame: {texts:?}"
        );
        // Every card still fits the height it is given; the wav note is the one
        // that is allowed to be wider, and only up to its own width.
        assert_eq!(cards.len(), 3, "the lead space, the note and its space: {cards:?}");
        for card in &cards {
            assert!(
                card.height() <= CARD_H && card.width() <= WAV_CARD_W,
                "a frame grew to {:?}, past the {WAV_CARD_W}x{CARD_H} it is given",
                card.size()
            );
        }

        // The note frame says what it plays and how long for, and carries the
        // start over the three length boxes; the space carries the three alone.
        for (header, width, boxes) in [
            ("wav · 3/8", WAV_CARD_W, vec!["1.250s", "0 whole", "× 3", "1/8"]),
            ("space · 1/4", CARD_W, vec!["0 whole", "× 1", "1/4"]),
        ] {
            let found: Vec<egui::Pos2> =
                texts.iter().filter(|(t, _)| t == header).map(|(_, p)| *p).collect();
            assert_eq!(found.len(), 1, "{header:?} is painted {} times: {texts:?}", found.len());
            let top = found[0];
            let mut column: Vec<&(String, egui::Pos2)> = texts
                .iter()
                .filter(|(_, p)| {
                    p.y > top.y && p.y < top.y + CARD_H && p.x <= top.x && p.x > top.x - width
                })
                .collect();
            column.sort_by(|a, b| a.1.y.total_cmp(&b.1.y));
            let painted: Vec<&str> = column.iter().map(|(t, _)| t.as_str()).collect();
            assert_eq!(painted, boxes, "the {header:?} card's boxes: {texts:?}");
        }
    }

    /// The start belongs to the *file*, so it cannot point past the end of one.
    /// A track swapped for a shorter recording, or a hand-edited project, must
    /// come back inside it rather than play silence.
    #[test]
    fn a_start_past_the_end_of_the_file_is_pulled_back_into_it() {
        let registry = TrackRegistry::default();
        registry.add_wav("short.wav", std::path::PathBuf::from("/x/short.wav"), 2.0);
        let mut panel = ComposerPanel::new(registry, crate::midi::new_midi_taps());
        panel.add_row();
        panel.rows[0].add_note(DEFAULT_PITCH);
        panel.rows[0].items[0].start = 30.0;

        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
        });
        assert_eq!(panel.rows[0].items[0].start, 2.0);
    }

    /// The nominator is a *box on the card*, and a card is a fixed size. It has
    /// to be drawn — on the space frame and the lead space as much as on the
    /// note — and it has to be drawn **inside** its own card: a [`CARD_H`] left
    /// at three boxes would push the fraction box out from under it and into the
    /// frame below, which nothing asserted about a `Duration` would ever notice.
    #[test]
    fn every_frame_draws_its_nominator_inside_its_card() {
        let (registry, _ids) = registry_with(&["one"]);
        let mut panel = ComposerPanel::new(registry, crate::midi::new_midi_taps());
        panel.add_row();
        panel.rows[0].add_note(DEFAULT_PITCH);
        // A nominator on the note, a different one on its space, and the lead
        // space left at its default — so each card is told apart by what it
        // paints, not by where it happens to sit.
        panel.rows[0].items[0].dur = Duration::with_num(0, 3, Fraction::Eighth);
        panel.rows[0].items[0].space = Duration::with_num(1, 5, Fraction::Sixteenth);

        // Under the app's own style, not egui's defaults: the boxes are sized
        // by the text in them, and this is the only way the number this test
        // measures is the number on screen.
        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
        });
        fn walk(
            sh: &egui::Shape,
            texts: &mut Vec<(String, egui::Pos2)>,
            cards: &mut Vec<egui::Rect>,
        ) {
            match sh {
                egui::Shape::Text(t) => texts.push((t.galley.text().to_string(), t.pos)),
                // A card is a filled rounded rect drawn in one of the three
                // frame colours — the only shapes in the panel that are.
                egui::Shape::Rect(r) if CARD_FILLS.contains(&r.fill) => cards.push(r.rect),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, texts, cards)),
                _ => {}
            }
        }
        let (mut texts, mut cards) = (Vec::new(), Vec::new());
        for cs in &out.shapes {
            walk(&cs.shape, &mut texts, &mut cards);
        }

        // The card is a fixed size, and the boxes have to fit *in* it: a card
        // that grew past [`CARD_H`] overflows the lane it is drawn in, and the
        // one below it. This is the assertion that fails when a box is added and
        // the constant is not.
        assert_eq!(cards.len(), 3, "the lead space, the note and its space: {cards:?}");
        for card in &cards {
            assert!(
                card.height() <= CARD_H && card.width() <= CARD_W,
                "a frame grew to {:?}, past the {CARD_W}x{CARD_H} it is given",
                card.size()
            );
        }
        let at = |what: &str| -> egui::Pos2 {
            let found: Vec<egui::Pos2> =
                texts.iter().filter(|(t, _)| t == what).map(|(_, p)| *p).collect();
            assert_eq!(found.len(), 1, "{what:?} is painted {} times: {texts:?}", found.len());
            found[0]
        };

        // Every card: its header, the card it is painted on, and then
        // everything painted on that card beneath it, top to bottom. Reading the column back whole is what makes this a test
        // of the *card* rather than of one string: a box that landed on the
        // neighbouring card, or one that did not draw at all, changes the list.
        for (header, boxes) in [
            ("lead · 0", vec!["0 whole", "× 1", "—"]),
            ("C4 · 3/8", vec!["C4", "0 whole", "× 3", "1/8"]),
            ("space · 1 + 5/16", vec!["1 whole", "× 5", "1/16"]),
        ] {
            let top = at(header);
            let card = *cards
                .iter()
                .find(|c| c.contains(top))
                .unwrap_or_else(|| panic!("{header:?} is painted on no card: {cards:?}"));
            let mut column: Vec<&(String, egui::Pos2)> = texts
                .iter()
                .filter(|(_, p)| p.y > top.y && card.contains(*p))
                .collect();
            column.sort_by(|a, b| a.1.y.total_cmp(&b.1.y));
            let painted: Vec<&str> = column.iter().map(|(t, _)| t.as_str()).collect();
            assert_eq!(painted, boxes, "the {header:?} card's boxes: {texts:?}");
            // One column, not two cards' worth read as one.
            assert!(
                column.iter().all(|(_, p)| (p.x - column[0].1.x).abs() < 1.0),
                "the {header:?} card's boxes are not in one column: {column:?}"
            );
            // The last box, its text and all, still inside the card it belongs
            // to: CARD_H has to grow with the number of boxes.
            let last = column.last().expect("a card draws its boxes").1.y;
            assert!(
                last + 20.0 <= top.y + CARD_H,
                "the {header:?} card's last box falls out of the bottom of the \
                 card: {last} > {}",
                top.y + CARD_H - 20.0
            );
        }
    }

    /// A take with nothing to play against is still a take: it runs from the
    /// button press, and what was played comes back as rows of the composition —
    /// rounded, on the track the recorder was pointed at.
    #[test]
    fn a_take_becomes_rows_on_the_composition() {
        let (registry, ids) = registry_with(&["one"]);
        let mut panel = ComposerPanel::new(registry, crate::midi::new_midi_taps());
        let ctx = egui::Context::default();
        panel.reconcile_tracks();
        panel.start_recording(&ctx);
        // Nothing to play along to, so no transport was started — the take is
        // timed from the press instead.
        assert!(panel.player.is_none() && panel.preparing.is_none());
        let tap = panel.recording.as_ref().expect("a take is running").tap.clone();

        // The keyboard, as the MIDI callback delivers it: two quarter notes at
        // the default 120 BPM, the second a beat after the first.
        let now = Instant::now();
        let at = |ms: u64| now + std::time::Duration::from_millis(ms);
        tap.lock().unwrap().extend([
            (at(0), [0x90, 60, 100]),
            (at(480), [0x80, 60, 0]),
            (at(500), [0x90, 64, 100]),
            (at(990), [0x80, 64, 0]),
        ]);

        let status = panel.finish_recording().expect("the take reports itself");
        assert!(status.contains("2 note(s)"), "{status}");
        assert_eq!(panel.rows.len(), 1);
        assert_eq!(panel.rows[0].track_id, Some(ids[0]));
        assert_eq!(
            panel.rows[0].items.iter().map(|i| i.pitch).collect::<Vec<_>>(),
            vec![60, 64]
        );
        // Rounded to 1/16 and back to seconds: a quarter note each, a beat apart.
        let played = panel.rows[0].planned_notes(panel.secs_per_unit());
        assert_eq!(played.len(), 2);
        assert!((played[0].at_secs - 0.0).abs() < 1e-9, "{played:?}");
        assert!((played[1].at_secs - 0.5).abs() < 1e-9, "{played:?}");
        assert!((played[0].dur_secs - 0.5).abs() < 1e-9, "{played:?}");
        // The take is over: the tap is emptied and nothing is left running.
        assert!(panel.recording.is_none());
        assert!(tap.lock().unwrap().is_empty());
    }

    /// A row's track select box must offer every track in the registry, not the
    /// ones that happened to exist when the row was added.
    ///
    /// Driven with a real click on the real popup Area, and — crucially — the
    /// box is **opened before the tracks are created**, which is what used to
    /// freeze it (an Area measures itself once per id). The assertion is on the
    /// text actually painted, so it fails whether the list is stale, the popup is
    /// clipped away, or two rows collide on one widget id.
    #[test]
    fn a_rows_select_box_offers_tracks_created_after_the_row() {
        let registry = TrackRegistry::default();
        for n in 1..=2 {
            registry.add(format!("t{n}"), std::path::PathBuf::from("/x"), None, false, None);
        }
        let mut panel = ComposerPanel::new(registry.clone(), crate::midi::new_midi_taps());
        let ctx = egui::Context::default();
        // The app's own spacing and text sizes: how tall a popup item is decides
        // how much of the list a stuck popup can still show, so the default
        // style hides the defect this exists to catch.
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(10.0, 6.0);
        {
            use egui::{FontFamily::Proportional, FontId, TextStyle};
            style.text_styles = [
                (TextStyle::Heading, FontId::new(18.0, Proportional)),
                (TextStyle::Body, FontId::new(14.0, Proportional)),
                (TextStyle::Button, FontId::new(14.0, Proportional)),
                (TextStyle::Monospace, FontId::new(13.0, egui::FontFamily::Monospace)),
                (TextStyle::Small, FontId::new(11.0, Proportional)),
            ]
            .into();
        }
        ctx.set_style(style);

        // Every string the frame paints, popups included, with where it landed.
        let run = |panel: &mut ComposerPanel, input: egui::RawInput| -> Vec<String> {
            let out = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| panel.ui(ui));
                });
            });
            let mut texts = Vec::new();
            fn walk(sh: &egui::Shape, out: &mut Vec<String>) {
                match sh {
                    egui::Shape::Text(t) => {
                        out.push(format!("{}@{},{}", t.galley.text(), t.pos.x, t.pos.y))
                    }
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            for cs in &out.shapes {
                walk(&cs.shape, &mut texts);
            }
            texts
        };
        let click = |at: egui::Pos2| {
            let mut i = egui::RawInput::default();
            i.events = vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                },
            ];
            i
        };
        /// `text@x,y`, as `walk` writes it.
        fn painted(s: &str) -> Option<(&str, f32, f32)> {
            let (name, pos) = s.rsplit_once('@')?;
            let (nx, ny) = pos.split_once(',')?;
            Some((name, nx.parse().ok()?, ny.parse().ok()?))
        }

        /// A registry track's name, which is what this test's popups list.
        fn is_track_name(name: &str) -> bool {
            name.strip_prefix('t').is_some_and(|n| n.parse::<u32>().is_ok())
        }

        // Which **tracks** the popup opened under `y` is listing: the items sit
        // in the popup's own left column, below the button that opened it.
        //
        // Track names specifically, not "every string painted there": a popup is
        // an `Area` and egui does not clip it to itself when it fits, so nothing
        // in the painted output distinguishes it from the row head behind it —
        // and the head's controls land at the same x, a pixel away in y. A
        // filter on position alone silently starts counting whatever control the
        // head grows next.
        let listed = |texts: &[String], y: f32| -> Vec<String> {
            texts
                .iter()
                .filter_map(|s| {
                    let (name, nx, ny) = painted(s)?;
                    // The popup's own column: its frame indents its items past
                    // the row controls behind it.
                    (is_track_name(name)
                        && (29.0..40.0).contains(&nx)
                        && ny > y + 10.0
                        && ny < y + 260.0)
                        .then(|| name.to_string())
                })
                .collect()
        };

        // Where the row heads' select boxes are, found from the painted track
        // name rather than hardcoded — the panel grows a header now and then,
        // and a test that silently clicks past it proves nothing.
        let boxes = |texts: &[String]| -> Vec<egui::Pos2> {
            texts
                .iter()
                .filter_map(|s| {
                    let (name, nx, ny) = painted(s)?;
                    (is_track_name(name) && (20.0..28.0).contains(&nx))
                        .then(|| egui::pos2(nx + 66.0, ny + 2.0))
                })
                .collect()
        };

        panel.add_row();
        let laid_out = run(&mut panel, egui::RawInput::default());

        // The first row's select box, opened once while only one track exists.
        // Opening it is the whole point: that first view is what its popup used
        // to be stuck at.
        let box1 = *boxes(&laid_out).first().expect("the row's select box is drawn");
        run(&mut panel, click(box1));
        let open = run(&mut panel, egui::RawInput::default());
        assert_eq!(
            listed(&open, box1.y),
            vec!["t1", "t2"],
            "the popup did not open over the first row's select box — the click \
             missed, and this test is not testing anything: {open:?}"
        );

        // Close it, add a second row, then create another track.
        run(&mut panel, click(egui::pos2(900.0, 900.0)));
        let closed = run(&mut panel, egui::RawInput::default());
        assert!(
            listed(&closed, box1.y).is_empty(),
            "the popup was still open, so reopening it proves nothing: {closed:?}"
        );
        panel.add_row();
        run(&mut panel, egui::RawInput::default());
        for n in 3..=4 {
            registry.add(format!("t{n}"), std::path::PathBuf::from("/x"), None, false, None);
        }
        let laid_out = run(&mut panel, egui::RawInput::default());
        let box2 = *boxes(&laid_out).get(1).expect("the second row's select box is drawn");

        // The first row must now offer them too. It is a fresh popup Area, so it
        // takes a sizing pass to appear — hence the second frame.
        run(&mut panel, click(box1));
        run(&mut panel, egui::RawInput::default());
        let reopened = run(&mut panel, egui::RawInput::default());
        assert_eq!(
            listed(&reopened, box1.y),
            vec!["t1", "t2", "t3", "t4"],
            "tracks created after this row's select box was first opened are \
             missing from it: {reopened:?}"
        );

        // And the row added later, which never had a popup of its own.
        run(&mut panel, click(egui::pos2(900.0, 900.0)));
        run(&mut panel, egui::RawInput::default());
        run(&mut panel, click(box2));
        run(&mut panel, egui::RawInput::default());
        let second = run(&mut panel, egui::RawInput::default());
        assert_eq!(
            listed(&second, box2.y),
            vec!["t1", "t2", "t3", "t4"],
            "the second row's select box is missing a track: {second:?}"
        );
    }

    /// The requirement's second half: a project whose grid has been deleted must
    /// load with that row marked, must not have some other track silently
    /// adopted onto it, must still say what it wanted, and must **survive being
    /// saved again** — losing the file name on the next save would make the
    /// damage permanent.
    #[test]
    fn a_missing_source_is_kept_until_the_user_replaces_it() {
        let (registry, ids) = registry_with(&["other"]);
        let mut panel = ComposerPanel::new(registry.clone(), crate::midi::new_midi_taps());
        let loaded = project::Project {
            name: "Song".to_string(),
            tempo_bpm: 100.0,
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            rows: vec![project::ProjectRow {
                name: "the voice".to_string(),
                track_name: "Voice".to_string(),
                source: project::TrackSource::LeSynth { file: "Voice.lsft".to_string(), state: None },
                gain: 0.5,
                lead: Duration::new(0, Fraction::Eighth),
                enabled: true,
                autosave: false,
                autoscroll: true,
                items: vec![Item {
                    id: 0,
                    pitch: 62,
                    start: 0.0,
                    dur: Duration::new(0, Fraction::Quarter),
                    space: Duration::new(0, Fraction::None),
                }],
            }],
        };
        // The app could not bind it: the grid is gone.
        panel.apply_project(&loaded, std::path::PathBuf::from("/tmp/Song"), &[None]);

        assert_eq!(panel.rows.len(), 1);
        assert!(panel.rows[0].missing.is_some(), "the row must be marked");
        assert_eq!(panel.rows[0].track_id, None);
        assert_eq!(panel.rows[0].gain, 0.5);
        assert!(!panel.rows[0].autosave, "autosave must round-trip");
        assert_eq!(panel.rows[0].items.len(), 1);

        // A track exists, and the row must NOT be quietly moved onto it.
        panel.reconcile_tracks();
        assert_eq!(panel.rows[0].track_id, None, "an unresolved row was adopted silently");

        // Saving again keeps the reference rather than erasing it.
        let again = panel.to_project("Song", |_| project::TrackSource::None);
        assert_eq!(
            again.rows[0].source,
            project::TrackSource::LeSynth { file: "Voice.lsft".to_string(), state: None },
            "a re-save lost what the row was looking for"
        );
        assert_eq!(again.rows[0].track_name, "Voice.lsft");

        // The user picks a replacement: the row becomes ordinary again.
        panel.rows[0].track_id = Some(ids[0]);
        panel.rows[0].missing = None;
        panel.reconcile_tracks();
        assert_eq!(panel.rows[0].track_id, Some(ids[0]));
        let fixed = panel.to_project("Song", |_| project::TrackSource::LeSynthDefault);
        assert_eq!(fixed.rows[0].source, project::TrackSource::LeSynthDefault);
        assert!(fixed.rows[0].enabled, "a row that plays must be saved as one");
    }

    /// Loading replaces the composition, and everything on a row that is not the
    /// notes has to come back too — a project that forgets the tempo or a row's
    /// lead is not a saved project.
    #[test]
    fn a_loaded_project_restores_every_row_setting() {
        let (registry, ids) = registry_with(&["a", "b"]);
        let mut panel = ComposerPanel::new(registry, crate::midi::new_midi_taps());
        panel.add_row();
        let loaded = project::Project {
            name: "Two".to_string(),
            tempo_bpm: 88.0,
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            rows: vec![
                project::ProjectRow {
                    name: "first".to_string(),
                    track_name: "a".to_string(),
                    source: project::TrackSource::LeSynthDefault,
                    gain: 1.5,
                    lead: Duration::new(1, Fraction::Sixteenth),
                    // Switched off in the saved project: a row that was muted
                    // must come back muted, or a loaded project plays something
                    // the user had silenced.
                    enabled: false,
                    autosave: true,
                    autoscroll: true,
                    items: vec![],
                },
                project::ProjectRow {
                    // Left unnamed, the way a project saved before rows had
                    // names has every row: this one must still come back with a
                    // name of its own.
                    name: String::new(),
                    track_name: "b".to_string(),
                    source: project::TrackSource::LeSynthDefault,
                    gain: 0.25,
                    lead: Duration::new(0, Fraction::None),
                    enabled: true,
                    autosave: false,
                    autoscroll: true,
                    items: vec![Item {
                        id: 0,
                        pitch: 70,
                        start: 0.0,
                        dur: Duration::new(0, Fraction::Half),
                        space: Duration::new(0, Fraction::Quarter),
                    }],
                },
            ],
        };
        panel.apply_project(&loaded, std::path::PathBuf::from("/tmp/Two"), &[
            Some(ids[0]),
            Some(ids[1]),
        ]);

        assert_eq!(panel.rows.len(), 2, "the old row must be replaced, not appended to");
        assert_eq!(panel.tempo_bpm, 88.0);
        assert_eq!(panel.rows[0].gain, 1.5);
        assert_eq!(panel.rows[0].lead, Duration::new(1, Fraction::Sixteenth));
        assert_eq!(panel.rows[1].track_id, Some(ids[1]));
        assert!(!panel.rows[0].enabled, "a row saved switched off must load switched off");
        assert!(panel.rows[1].enabled);
        assert!(!panel.rows[1].autosave);
        assert_eq!(panel.rows[1].items[0].pitch, 70);
        // Row ids stay unique, so two rows never share egui widget state.
        assert_ne!(panel.rows[0].id, panel.rows[1].id);
        // The name the project saved comes back on the row it was saved for…
        assert_eq!(panel.rows[0].name, "first");
        assert_eq!(panel.rows[0].name_edit, "first", "the box shows what the row is called");
        // …and a row that was saved without one is still named, by the UUID it
        // was given on the way in — an unnamed row is not a blank row.
        assert!(
            uuid::Uuid::parse_str(&panel.rows[1].name).is_ok(),
            "a row loaded from a project with no name should fall back to a UUID, got {:?}",
            panel.rows[1].name
        );
        // And only rows with autosave on ask for a fresh grid.
        assert_eq!(panel.autosave_track_ids(), vec![ids[0]]);
    }

    /// A block is built by clicking frames: the boxes are checkboxes, so a
    /// second box joins the first rather than replacing it, a box already in
    /// the block lets the block go, and shift moves one end exactly. Rows are
    /// independent, because a block that spans rows is exactly what this is for.
    #[test]
    fn a_block_is_clicked_open_and_clicked_shut() {
        let mut row = row_with(&[
            (60, frac(Fraction::Quarter), frac(Fraction::None)),
            (62, frac(Fraction::Quarter), frac(Fraction::None)),
            (64, frac(Fraction::Quarter), frac(Fraction::None)),
        ]);
        let ids: Vec<u64> = row.items.iter().map(|i| i.id).collect();

        row.select_item(ids[1], false);
        assert_eq!(row.sel_range(), Some((1, 1)), "a plain click selects one frame");

        row.select_item(ids[1], false);
        assert_eq!(row.sel_range(), None, "the same click again drops the block");

        // Two plain clicks: the second box does not untick the first. This is
        // the whole of the complaint that changed the rule.
        row.select_item(ids[0], false);
        row.select_item(ids[2], false);
        assert_eq!(
            row.sel_range(),
            Some((0, 2)),
            "a plain click on a second box must grow the block, not restart it"
        );

        // Backwards too: the block grows to a box in front of it.
        row.select_item(ids[1], false);
        assert_eq!(row.sel_range(), None, "a box inside the block drops the block");
        row.select_item(ids[2], false);
        row.select_item(ids[0], false);
        assert_eq!(row.sel_range(), Some((0, 2)), "a box in front of the block grows it");

        // Growing anchors on the end that did not move, so shift after a plain
        // click drags the end just clicked — here, back off the block again.
        row.select_item(ids[1], true);
        assert_eq!(row.sel_range(), Some((1, 2)), "shift shrinks from the end last clicked");

        row.select_item(ids[0], true);
        assert_eq!(
            row.sel_range(),
            Some((0, 2)),
            "shift takes everything between the anchor and the frame clicked, \
             whichever way round they are"
        );

        // The ends are ids, so a frame deleted in front of the block does not
        // slide the block onto other notes.
        row.select_item(ids[1], false);
        row.select_item(ids[1], false);
        row.select_item(ids[2], true);
        row.delete_item(0);
        assert_eq!(row.sel_range(), Some((0, 1)), "the same two frames, one place earlier");

        // …and a frame deleted *from* the block takes the block with it, rather
        // than leaving a range pointing at nothing.
        row.delete_item(1);
        assert_eq!(row.sel, None);
    }

    /// "Span Rows" is what makes a natural block one press: mark a phrase on
    /// the row you can hear it in, and the same *stretch of time* comes out of
    /// every other row — however many frames each of them spends on it.
    #[test]
    fn spanning_the_rows_takes_the_same_stretch_of_time_out_of_each() {
        let quarter = (60, frac(Fraction::Quarter), frac(Fraction::None));
        let eighth = (48, frac(Fraction::Eighth), frac(Fraction::None));
        let mut panel = panel_with_rows(&[&[quarter; 4], &[eighth; 8]]);

        // The second and third quarter on the top row: 1/4 up to 3/4.
        panel.rows[0].select_range(1, 2);
        panel.span_rows();

        assert_eq!(panel.rows[0].sel_range(), Some((1, 2)), "the row it was marked on is unchanged");
        assert_eq!(
            panel.rows[1].sel_range(),
            Some((2, 5)),
            "four eighths cover the same stretch as two quarters"
        );
        let block = panel.block().expect("a block");
        assert_eq!((block.from, block.to), (QUARTER, 3 * QUARTER));
        assert_eq!(block.frames(), 6);
    }

    /// The point of a block: cloning it repeats every row by **the same amount
    /// of time**, so what sounded together on the first pass sounds together on
    /// the fourth. Rows spend a window on different numbers of frames and leave
    /// different amounts of silence at either end of it; the copy occupies the
    /// window, not the frames.
    #[test]
    fn a_clone_moves_every_row_on_by_the_same_stretch_of_time() {
        // A row that comes in an eighth late, against one that starts at zero
        // and stops an eighth early — so both ends of the window are silence on
        // one row or the other.
        let mut panel = panel_with_rows(&[
            &[(60, frac(Fraction::Quarter), frac(Fraction::None))],
            &[(36, frac(Fraction::Eighth), frac(Fraction::Eighth))],
        ]);
        panel.rows[0].lead = frac(Fraction::Eighth);
        panel.select_all();

        let block = panel.block().expect("a block");
        assert_eq!((block.from, block.to), (0, 3 * EIGHTH), "the window is what the rows span");

        let before = note_starts(&panel);
        assert_eq!(before, vec![vec![EIGHTH], vec![0]]);

        panel.clone_block(2);

        let window = 3 * EIGHTH;
        assert_eq!(
            note_starts(&panel),
            vec![
                vec![EIGHTH, EIGHTH + window, EIGHTH + 2 * window],
                vec![0, window, 2 * window],
            ],
            "each copy lands exactly one window on, on both rows"
        );
        // Nothing was invented: the copies are the frames that were selected.
        assert_eq!(panel.rows[0].items.len(), 3);
        assert!(panel.rows[0].items.iter().all(|i| i.pitch == 60));
        assert!(panel.rows[1].items.iter().all(|i| i.pitch == 36));
    }

    /// Clone is pressed twice as often as it is pressed once, so the second
    /// press has to mean the same as the first. It does because the selection
    /// moves onto the copy: the *original's* trailing space now carries the gap
    /// to that copy, and measuring the block from there would grow the window
    /// every time.
    #[test]
    fn cloning_the_clone_repeats_the_same_stretch_again() {
        let mut panel = panel_with_rows(&[
            &[(60, frac(Fraction::Quarter), frac(Fraction::None))],
            &[(36, frac(Fraction::Eighth), frac(Fraction::Eighth))],
        ]);
        panel.rows[0].lead = frac(Fraction::Eighth);
        panel.select_all();
        let window = 3 * EIGHTH;

        panel.clone_block(1);
        assert_eq!(panel.block().expect("a block").window(), window, "the copy measures the same");
        panel.clone_block(1);

        assert_eq!(
            note_starts(&panel),
            vec![
                vec![EIGHTH, EIGHTH + window, EIGHTH + 2 * window],
                vec![0, window, 2 * window],
            ],
        );
    }

    /// A gap between copies is arithmetic, not a length anyone picked, so it is
    /// not always a length the three select boxes can name — `255/256` needs
    /// 255 of the smallest fraction there is. It still has to land *exactly*,
    /// because a rounded gap is a copy that drifts further out of step with
    /// every repeat, so what will not fit in one frame is laid out over frames
    /// of no length: silence, which is what a gap is.
    #[test]
    fn a_gap_no_length_box_can_name_is_still_exact() {
        assert_eq!(Duration::exact(255), None, "no fraction divides 255 units few enough times");
        assert_eq!(Duration::split(255).iter().map(|d| d.units()).sum::<i64>(), 255);
        assert_eq!(
            Duration::exact(UNITS_PER_WHOLE + 3 * EIGHTH),
            Some(Duration::with_num(1, 3, Fraction::Eighth)),
            "and a length that can be named is named the plainest way"
        );

        // A whole note against a single 1/256: the short row has 255 units of
        // window to make up between one copy and the next.
        let mut panel = panel_with_rows(&[
            &[(60, Duration::new(1, Fraction::None), frac(Fraction::None))],
            &[(36, frac(Fraction::TwoHundredFiftySixth), frac(Fraction::None))],
        ]);
        panel.select_all();
        panel.clone_block(2);

        let w = UNITS_PER_WHOLE;
        assert_eq!(
            note_starts(&panel),
            vec![vec![0, w, 2 * w], vec![0, w, 2 * w]],
            "the odd gap is spread over placeholder frames rather than rounded away"
        );
        // The frames it was spread over are silences: they play nothing.
        let spu = panel.secs_per_unit();
        assert_eq!(panel.rows[1].planned_notes(spu).len(), 3);
    }

    /// Paste puts the block back at the end of the **longest** row, not at the
    /// end of each row: rows are played from zero together, so a block pasted
    /// row by row would be dealt out as an echo of itself.
    #[test]
    fn a_pasted_block_lands_at_the_end_of_the_composition_on_every_row_at_once() {
        let mut panel = panel_with_rows(&[
            &[(60, frac(Fraction::Quarter), frac(Fraction::None))],
            &[(36, frac(Fraction::Eighth), frac(Fraction::None))],
        ]);
        panel.select_all();
        panel.copy_block();
        panel.clear_block();
        panel.paste_block();

        assert_eq!(
            note_starts(&panel),
            vec![vec![0, QUARTER], vec![0, QUARTER]],
            "the short row is padded up to the paste point rather than closing the gap"
        );
        // The block follows what was pasted, so a second paste carries on.
        assert_eq!(panel.rows[0].sel_range(), Some((1, 1)));
        assert_eq!(panel.rows[1].sel_range(), Some((1, 1)));

        panel.paste_block();
        assert_eq!(note_starts(&panel), vec![vec![0, QUARTER, 2 * QUARTER], vec![0, QUARTER, 2 * QUARTER]]);
    }

    /// A block copied off a row that has since been removed is not quietly
    /// pasted somewhere else: the frames name a pitch, a length and a place in
    /// a file that mean what they mean on the track they were written for.
    #[test]
    fn a_paste_says_when_the_row_it_was_copied_from_is_gone() {
        let mut panel = panel_with_rows(&[
            &[(60, frac(Fraction::Quarter), frac(Fraction::None))],
            &[(36, frac(Fraction::Eighth), frac(Fraction::None))],
        ]);
        panel.select_all();
        panel.copy_block();
        panel.rows.remove(1);
        panel.paste_block();

        assert_eq!(panel.rows.len(), 1);
        assert_eq!(note_starts(&panel), vec![vec![0, QUARTER]]);
        assert!(panel.status.contains("gone"), "the status says so: {}", panel.status);
    }

    /// The block is visible: the frames in it are lit and ringed, the note
    /// frames carry the box that put them there, and the bar above the lanes
    /// says what is selected — a block spans rows, so it cannot be read off any
    /// one row's head.
    #[test]
    fn the_block_bar_and_the_lit_frames_say_what_is_selected() {
        let mut panel = panel_with_rows(&[&[
            (60, frac(Fraction::Quarter), frac(Fraction::Eighth)),
            (62, frac(Fraction::Quarter), frac(Fraction::Eighth)),
        ]]);
        panel.rows[0].select_range(0, 0);

        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
        });
        let (mut texts, mut fills) = (Vec::new(), Vec::new());
        fn walk(sh: &egui::Shape, texts: &mut Vec<String>, fills: &mut Vec<egui::Color32>) {
            match sh {
                egui::Shape::Text(t) => texts.push(t.galley.text().to_string()),
                egui::Shape::Rect(r) if CARD_FILLS.contains(&r.fill) => fills.push(r.fill),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, texts, fills)),
                _ => {}
            }
        }
        for cs in &out.shapes {
            walk(&cs.shape, &mut texts, &mut fills);
        }

        // One note and its space are one item, so both of the selected item's
        // frames are lit — and nothing else is.
        assert_eq!(
            fills.iter().filter(|f| **f == NOTE_SEL_FILL).count(),
            1,
            "the selected note is not lit: {fills:?}"
        );
        assert_eq!(
            fills.iter().filter(|f| **f == SPACE_SEL_FILL).count(),
            1,
            "the space tied to it did not follow it into the block: {fills:?}"
        );
        // A select box on each note frame, filled on the one in the block.
        assert_eq!(texts.iter().filter(|t| *t == "☑").count(), 1);
        assert_eq!(texts.iter().filter(|t| *t == "☐").count(), 1);
        // And the bar, which is the only place a block that spans rows can be
        // read at all.
        assert!(
            texts.iter().any(|t| t.contains("1 frame(s) on 1 row(s)")),
            "the block bar does not say what is selected: {texts:?}"
        );
        assert!(texts.iter().any(|t| t == "↔ Span Rows"), "{texts:?}");
    }

    /// A row switched off is silent, not gone: the transport and an export skip
    /// it, and everything on it — its notes, its place, the length it gives the
    /// composition — stays exactly where the user left it. Muting a part to
    /// hear the one under it must cost nothing to undo.
    #[test]
    fn a_row_switched_off_plays_nothing_and_loses_nothing() {
        let mut panel = panel_with_rows(&[
            &[
                (60, frac(Fraction::Quarter), frac(Fraction::Eighth)),
                (62, frac(Fraction::Quarter), frac(Fraction::Eighth)),
            ],
            &[(48, Duration::new(1, Fraction::None), frac(Fraction::None))],
        ]);
        // A track behind each row, which is the other half of "has something to
        // play"; the registry is not asked anything here.
        for row in &mut panel.rows {
            row.track_id = Some(7);
            assert!(row.enabled, "a new row plays");
        }
        let length = panel.end_units();
        assert_eq!(panel.live_snapshot().rows.len(), 2);

        panel.rows[1].enabled = false;
        assert!(panel.has_playable_rows(), "the row still switched on must still play");
        assert_eq!(
            panel.live_snapshot().rows.len(),
            1,
            "a switched-off row must be out of the transport, not in it at some level"
        );

        panel.rows[0].enabled = false;
        assert!(!panel.has_playable_rows(), "nothing is switched on, so nothing plays");

        // Nothing was taken from the rows themselves.
        assert_eq!(panel.rows[0].items.len(), 2);
        assert_eq!(panel.rows[1].items.len(), 1);
        assert_eq!(panel.end_units(), length, "a silent lane is still part of the timeline");

        // And back on, whole.
        for row in &mut panel.rows {
            row.enabled = true;
        }
        assert_eq!(panel.live_snapshot().rows.len(), 2);
    }

    /// The whole of a row in one press, on as many rows as you like: what the
    /// select box on the row head is for. A hundred-frame part is a scroll to
    /// its end and back to block it by hand, and a whole part is the commonest
    /// block there is — it is what a repeat repeats.
    #[test]
    fn selecting_a_whole_row_blocks_every_frame_on_it_across_rows() {
        let mut panel = panel_with_rows(&[
            &[
                (60, frac(Fraction::Quarter), frac(Fraction::Eighth)),
                (62, frac(Fraction::Quarter), frac(Fraction::Eighth)),
                (64, frac(Fraction::Quarter), frac(Fraction::None)),
            ],
            &[
                (48, frac(Fraction::Half), frac(Fraction::None)),
                (48, frac(Fraction::Half), frac(Fraction::None)),
            ],
        ]);
        assert!(!panel.rows[0].all_selected(), "nothing is selected to begin with");

        panel.rows[0].select_all();
        assert_eq!(panel.rows[0].sel_range(), Some((0, 2)));
        assert!(panel.rows[0].all_selected());
        // Rows are independent: one row's box says nothing about another's.
        assert_eq!(panel.rows[1].sel_range(), None);
        assert!(!panel.rows[1].all_selected());

        // Ticked on the second row too, the block spans both, whole.
        panel.rows[1].select_all();
        let block = panel.block().expect("a block across both rows");
        assert_eq!(block.rows.len(), 2);
        assert_eq!((block.rows[0].first, block.rows[0].last), (0, 2));
        assert_eq!((block.rows[1].first, block.rows[1].last), (0, 1));

        // Which is what cloning it repeats: every frame of every row, once more.
        panel.clone_block(1);
        assert_eq!(panel.rows[0].items.len(), 6, "the whole row, twice");
        assert_eq!(panel.rows[1].items.len(), 4);

        // The box follows the block rather than remembering a press: the clone
        // left the copy selected, which is still the tail and not the whole.
        assert!(!panel.rows[0].all_selected(), "half a row is not a whole row");

        // And an empty row has nothing to select, so its box cannot tick.
        let mut empty = row_with(&[]);
        empty.select_all();
        assert_eq!(empty.sel_range(), None);
        assert!(!empty.all_selected());
    }

    /// The row's select box, through the real widget: it is derived from the
    /// block, which is exactly the sort of switch that draws correctly and does
    /// nothing. One press blocks the row; a second lets it go.
    #[test]
    fn the_row_select_box_blocks_the_row_and_releases_it() {
        let mut panel = panel_with_rows(&[
            &[
                (60, frac(Fraction::Quarter), frac(Fraction::Eighth)),
                (62, frac(Fraction::Quarter), frac(Fraction::Eighth)),
            ],
            &[(48, frac(Fraction::Half), frac(Fraction::None))],
        ]);
        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let run = |panel: &mut ComposerPanel, input: egui::RawInput| -> Vec<(String, egui::Pos2)> {
            let out = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
            });
            let mut texts = Vec::new();
            fn walk(sh: &egui::Shape, out: &mut Vec<(String, egui::Pos2)>) {
                match sh {
                    egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.pos)),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            for cs in &out.shapes {
                walk(&cs.shape, &mut texts);
            }
            texts
        };
        let click = |at: egui::Pos2| {
            let mut i = egui::RawInput::default();
            i.events = vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ];
            i
        };

        let laid = run(&mut panel, egui::RawInput::default());
        let boxes: Vec<egui::Pos2> = laid
            .iter()
            .filter(|(t, _)| t == "select all")
            .map(|(_, p)| egui::pos2(p.x + 4.0, p.y + 6.0))
            .collect();
        assert_eq!(boxes.len(), 2, "one select box per row: {boxes:?}");

        run(&mut panel, click(boxes[0]));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(panel.rows[0].sel_range(), Some((0, 1)), "the box did not reach the row");
        assert_eq!(panel.rows[1].sel_range(), None, "and it took no other row with it");

        // The second row's box adds it to the block rather than replacing it:
        // the block spans rows, which is the point of ticking two.
        run(&mut panel, click(boxes[1]));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(panel.rows[0].sel_range(), Some((0, 1)));
        assert_eq!(panel.rows[1].sel_range(), Some((0, 0)));

        // Ticked, it unticks: the block on that row goes.
        run(&mut panel, click(boxes[0]));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(panel.rows[0].sel_range(), None, "the box did not let the block go");
        assert_eq!(panel.rows[1].sel_range(), Some((0, 0)), "and again, only its own row");
    }

    /// A row that is switched off is not playing, so nothing in its lane may
    /// behave as though it were: no frame lit as sounding, and no scrolling the
    /// lane out from under the user — whatever “⏵ follow” is set to, which is
    /// why the box is greyed while the row is off.
    #[test]
    fn a_row_switched_off_is_left_alone_by_the_transport() {
        let mut row = row_with(&[(60, frac(Fraction::Quarter), frac(Fraction::Eighth))]);
        row.autoscroll = true;

        assert_eq!(row.lane_playhead(Some(3.5)), Some(3.5), "a playing row follows the transport");
        assert_eq!(row.lane_playhead(None), None, "and there is nothing to follow when stopped");

        row.enabled = false;
        assert_eq!(
            row.lane_playhead(Some(3.5)),
            None,
            "a switched-off lane must not be lit, or scrolled, by a transport it is not in"
        );
        // Switching it back on restores it: the follow setting was kept, not
        // cleared, so the lane goes back to behaving as the user left it.
        row.enabled = true;
        assert!(row.autoscroll);
        assert_eq!(row.lane_playhead(Some(3.5)), Some(3.5));
    }

    /// “Enable all” beside Add Track Row, through the real widget: it reports
    /// what is true of the rows, and one click carries every row with it. The
    /// state is derived rather than stored, which is exactly the sort of switch
    /// that can be drawn correctly and do nothing.
    #[test]
    fn enable_all_carries_every_row_and_reports_what_is_true() {
        let mut panel = panel_with_rows(&[
            &[(60, frac(Fraction::Quarter), frac(Fraction::Eighth))],
            &[(62, frac(Fraction::Quarter), frac(Fraction::Eighth))],
        ]);
        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let run = |panel: &mut ComposerPanel, input: egui::RawInput| -> Vec<(String, egui::Pos2)> {
            let out = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
            });
            let mut texts = Vec::new();
            fn walk(sh: &egui::Shape, out: &mut Vec<(String, egui::Pos2)>) {
                match sh {
                    egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.pos)),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            for cs in &out.shapes {
                walk(&cs.shape, &mut texts);
            }
            texts
        };
        let click = |at: egui::Pos2| {
            let mut i = egui::RawInput::default();
            i.events = vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ];
            i
        };
        let at = |texts: &[(String, egui::Pos2)], label: &str| -> egui::Pos2 {
            let (_, p) = texts
                .iter()
                .find(|(t, _)| t == label)
                .unwrap_or_else(|| panic!("no {label:?} in {texts:?}"));
            egui::pos2(p.x + 4.0, p.y + 6.0)
        };

        let laid = run(&mut panel, egui::RawInput::default());
        let all = at(&laid, "Enable all");

        // Every row is on, so one click switches the composition off.
        run(&mut panel, click(all));
        run(&mut panel, egui::RawInput::default());
        assert!(
            panel.rows.iter().all(|r| !r.enabled),
            "the switch did not reach the rows"
        );

        // It now reports that, so the next click turns them back on rather than
        // switching them off again.
        run(&mut panel, click(all));
        run(&mut panel, egui::RawInput::default());
        assert!(
            panel.rows.iter().all(|r| r.enabled),
            "the switch did not come back on with the rows"
        );

        // And it follows the rows, not the other way round: one row off leaves
        // it unticked, so a click puts that row back with the rest.
        panel.rows[1].enabled = false;
        run(&mut panel, egui::RawInput::default());
        run(&mut panel, click(all));
        run(&mut panel, egui::RawInput::default());
        assert!(
            panel.rows.iter().all(|r| r.enabled),
            "unticked while a row was off, it must switch every row on"
        );
    }

    /// Two frames must be selectable on one row, through the real widget: a
    /// second box clicked plainly joins the block rather than unticking the
    /// first, and shift then drags that end back. The rule is [`Row`]'s, but
    /// the modifier has to survive the button, which is where it would break
    /// without anything else failing.
    #[test]
    fn a_second_box_joins_the_block_instead_of_unticking_the_first() {
        let mut panel = panel_with_rows(&[&[
            (60, frac(Fraction::Quarter), frac(Fraction::Eighth)),
            (62, frac(Fraction::Quarter), frac(Fraction::Eighth)),
            (64, frac(Fraction::Quarter), frac(Fraction::Eighth)),
        ]]);
        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let run = |panel: &mut ComposerPanel, input: egui::RawInput| -> Vec<(String, egui::Pos2)> {
            let out = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
            });
            let mut texts = Vec::new();
            fn walk(sh: &egui::Shape, out: &mut Vec<(String, egui::Pos2)>) {
                match sh {
                    egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.pos)),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            for cs in &out.shapes {
                walk(&cs.shape, &mut texts);
            }
            texts
        };
        let click = |at: egui::Pos2, shift: bool| {
            let mods = egui::Modifiers { shift, ..Default::default() };
            let mut i = egui::RawInput::default();
            i.modifiers = mods;
            i.events = vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: mods,
                },
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: mods,
                },
            ];
            i
        };
        let boxes = |texts: &[(String, egui::Pos2)]| -> Vec<egui::Pos2> {
            texts
                .iter()
                .filter(|(t, _)| t == "☐" || t == "☑")
                .map(|(_, p)| egui::pos2(p.x + 3.0, p.y + 6.0))
                .collect()
        };

        let laid = run(&mut panel, egui::RawInput::default());
        let bx = boxes(&laid);
        assert_eq!(bx.len(), 3, "three note frames, three select boxes: {bx:?}");

        run(&mut panel, click(bx[0], false));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(panel.rows[0].sel_range(), Some((0, 0)), "a plain click did not select");

        run(&mut panel, click(bx[2], false));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(
            panel.rows[0].sel_range(),
            Some((0, 2)),
            "a second box unticked the first instead of joining the block"
        );

        // The block grew forwards, so it is anchored on frame 0 and shift drags
        // the end just clicked — frame 2 — back onto frame 1.
        run(&mut panel, click(bx[1], true));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(
            panel.rows[0].sel_range(),
            Some((0, 1)),
            "shift did not reach the button — the block would only ever grow"
        );
    }

    /// Every row has a name from the moment it exists — a UUID unless the box
    /// beside the button says otherwise — and that box empties into the row it
    /// names, so the next row is not quietly a second row of the same name.
    #[test]
    fn a_new_row_is_named_by_the_box_or_by_a_uuid() {
        let (registry, _) = registry_with(&["one"]);
        let mut panel = ComposerPanel::new(registry, crate::midi::new_midi_taps());

        panel.add_row();
        let auto = panel.rows[0].name.clone();
        assert!(
            uuid::Uuid::parse_str(&auto).is_ok(),
            "an unnamed row should be named after a UUID, got {auto:?}"
        );
        assert_eq!(panel.rows[0].name_edit, auto, "the row's box shows its name");

        // Two rows made the same way are not called the same thing — which is
        // the whole reason the default is a UUID and not "Row 1".
        panel.add_row();
        assert_ne!(panel.rows[0].name, panel.rows[1].name);

        // A name in the box goes on the row that button makes…
        panel.new_row_name = "  bass line  ".to_string();
        panel.add_row_at(0);
        assert_eq!(panel.rows[0].name, "bass line", "the name is taken, trimmed");
        assert_eq!(panel.rows[0].name_edit, "bass line");
        assert!(
            panel.status.contains("bass line"),
            "the status should say what was added: {}",
            panel.status
        );

        // …and only on that row: the box is empty behind it.
        assert_eq!(panel.new_row_name, "");
        panel.add_row();
        assert!(
            uuid::Uuid::parse_str(&panel.rows.last().unwrap().name).is_ok(),
            "the row after a named one should go back to a UUID"
        );

        // A status line names the row without reciting thirty-six characters of
        // hex at the user, while the row itself keeps the whole of it.
        let row = &panel.rows[3];
        assert_eq!(row.label(), format!("{}…", &row.name[..8]));
        assert_eq!(panel.rows[0].label(), "bass line", "a name the user typed is quoted whole");
    }

    /// A repeat's window is two *lengths*, both counted from the beginning of
    /// the composition — so the pair can be read against each other, and a stop
    /// left at zero means the end rather than the impossible.
    #[test]
    fn a_repeat_window_is_two_lengths_from_the_beginning() {
        // Eight quarter notes, back to back: two whole notes of composition.
        let items = [(60u8, frac(Fraction::Quarter), frac(Fraction::None)); 8];
        let mut panel = panel_with_rows(&[&items]);
        let end = panel.end_units();
        assert_eq!(end, 2 * UNITS_PER_WHOLE, "two whole notes of quarters");

        // Untouched, the window is the whole composition — what a repeat looped
        // over before there were boxes to trim it with. A stop of zero is not
        // "stop where you started"; it is the box nobody has touched, and it
        // means the end.
        assert_eq!(panel.repeat_to_units(), end);
        assert_eq!(panel.repeat_span_units(), (0, end));
        assert!(!panel.repeat_span_ignored(), "zero boxes are not a window being ignored");

        // From a half note in, to a whole note and three quarters in.
        panel.repeat_from = Duration::new(0, Fraction::Half);
        panel.repeat_to = Duration::with_num(0, 7, Fraction::Quarter);
        assert_eq!(
            panel.repeat_span_units(),
            (UNITS_PER_WHOLE / 2, 7 * UNITS_PER_WHOLE / 4)
        );
        // …in seconds, which is what the transport is handed. 120 BPM, so a
        // whole note is two seconds.
        let span = panel.repeat_span();
        assert!(
            (span.start_secs - 1.0).abs() < 1e-9 && (span.end_secs - 3.5).abs() < 1e-9,
            "{span:?}"
        );

        // Both ends are positions in the composition now, so writing more music
        // leaves a stop that was asked for exactly where it was asked for. Only
        // a stop of zero follows the music.
        panel.rows[0].add_note(60);
        let grown = panel.end_units();
        assert_eq!(grown, 2 * UNITS_PER_WHOLE + UNITS_PER_WHOLE / 4, "nine quarters");
        assert_eq!(panel.repeat_span_units(), (UNITS_PER_WHOLE / 2, 7 * UNITS_PER_WHOLE / 4));
        panel.repeat_to = Duration::new(0, Fraction::None);
        assert_eq!(panel.repeat_span_units(), (UNITS_PER_WHOLE / 2, grown));

        // A stop past the end is held at the end: out there is silence, not
        // music to repeat.
        panel.repeat_to = Duration::new(MAX_WHOLES, Fraction::None);
        assert_eq!(panel.repeat_to_units(), grown);
        assert_eq!(panel.repeat_span_units(), (UNITS_PER_WHOLE / 2, grown));
        assert!(!panel.repeat_span_ignored(), "reaching the end is not a collapsed window");

        // A start on or past the stop asks for a window with nothing in it. That
        // is not a short loop — it is a transport with no samples to play — so
        // the whole composition is looped and the panel says the boxes are being
        // overruled.
        panel.repeat_to = Duration::new(0, Fraction::Half);
        panel.repeat_from = Duration::new(0, Fraction::Half);
        assert_eq!(panel.repeat_span_units(), (0, grown), "a window has to have something in it");
        assert!(panel.repeat_span_ignored());

        panel.repeat_from = Duration::new(MAX_WHOLES, Fraction::None);
        assert_eq!(panel.repeat_span_units(), (0, grown));
        assert!(panel.repeat_span_ignored(), "a start past the stop is the same collapse");
    }

    /// The window travels to the transport the way a note edit does — in the
    /// live snapshot — so moving a box while a repeat is playing takes effect at
    /// the next wrap instead of needing Play pressed again.
    #[test]
    fn moving_a_repeat_window_is_a_live_edit_not_a_restart() {
        let items = [(60u8, frac(Fraction::Quarter), frac(Fraction::None)); 8];
        let mut panel = panel_with_rows(&[&items]);
        panel.rows[0].track_id = Some(7);
        let before = panel.live_snapshot();

        panel.repeat_from = Duration::new(0, Fraction::Half);
        let after = panel.live_snapshot();
        assert_eq!(after.shape, before.shape, "a window is not a row or a track");
        assert_eq!(after.rows, before.rows, "…and it moves no notes");
        assert_ne!(after.span, before.span, "the window did not reach the transport");
    }

    /// The window is on screen as two sets of the same three length boxes the
    /// rest of the panel is written with, and beside them what the two lengths
    /// actually came to — which is the whole reason they are next to each other.
    #[test]
    fn the_repeat_window_reads_back_what_the_two_lengths_come_to() {
        let items = [(60u8, frac(Fraction::Quarter), frac(Fraction::None)); 8];
        let mut panel = panel_with_rows(&[&items]);

        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let mut draw = |panel: &mut ComposerPanel| -> Vec<String> {
            let out = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
            });
            let mut texts = Vec::new();
            fn walk(sh: &egui::Shape, out: &mut Vec<String>) {
                match sh {
                    egui::Shape::Text(t) => out.push(t.galley.text().to_string()),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            for cs in &out.shapes {
                walk(&cs.shape, &mut texts);
            }
            texts
        };

        // Untouched, the stop box says nothing on its own, so the reading says
        // what it came to mean — the one thing about the line that has to be
        // learned rather than read.
        let texts = draw(&mut panel);
        // "from … to …" and nothing else: a word saying the pair are counted
        // off the beginning would sit behind the second set of boxes and read
        // as if it qualified only those. The seconds beside them do that job —
        // they are the composition's own clock — and the hovers say it outright.
        for label in ["🔁 Repeat from", "to"] {
            assert!(texts.iter().any(|t| t == label), "{label:?} is not drawn: {texts:?}");
        }
        assert!(
            texts.iter().any(|t| t == "loops 0.0 – 4.0 s (4.0 s, to the end)"),
            "an untouched stop must read as the end: {texts:?}"
        );

        // Two whole notes at 120 BPM is four seconds; from a half note in to
        // seven quarters in is 1.0 s to 3.5 s.
        panel.repeat_from = Duration::new(0, Fraction::Half);
        panel.repeat_to = Duration::with_num(0, 7, Fraction::Quarter);
        let texts = draw(&mut panel);
        assert!(
            texts.iter().any(|t| t == "loops 1.0 – 3.5 s (2.5 s)"),
            "the window is not read back: {texts:?}"
        );

        // Overruled, the line says so rather than showing a window that is not
        // the one playing.
        panel.repeat_from = Duration::new(MAX_WHOLES, Fraction::None);
        let texts = draw(&mut panel);
        assert!(
            texts.iter().any(|t| t.starts_with("⚠ nothing between them")),
            "a collapsed window must say so: {texts:?}"
        );
    }

    /// The rename form on a row, through the real widgets: a name typed into
    /// the box is a *draft* until the box is finished with — the button, or
    /// clicking away, which is how most people finish typing — and a rename in
    /// a project that has already been saved goes straight back into its file.
    #[test]
    fn a_row_is_renamed_by_its_own_box_and_button() {
        let mut panel = panel_with_rows(&[&[(60, frac(Fraction::Quarter), frac(Fraction::None))]]);
        panel.rows[0].set_name("old name".to_string());

        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let run = |panel: &mut ComposerPanel, input: egui::RawInput| -> Vec<(String, egui::Pos2)> {
            let out = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
            });
            let mut texts = Vec::new();
            fn walk(sh: &egui::Shape, out: &mut Vec<(String, egui::Pos2)>) {
                match sh {
                    egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.pos)),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            for cs in &out.shapes {
                walk(&cs.shape, &mut texts);
            }
            texts
        };
        let at = |texts: &[(String, egui::Pos2)], want: &str| -> egui::Pos2 {
            let (_, p) = texts
                .iter()
                .find(|(t, _)| t == want)
                .unwrap_or_else(|| panic!("{want:?} is not drawn: {texts:?}"));
            egui::pos2(p.x + 4.0, p.y + 6.0)
        };
        let click = |at: egui::Pos2| {
            let mut i = egui::RawInput::default();
            i.events = vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ];
            i
        };

        // The row draws its name in its box, and a rename button beside it.
        let laid = run(&mut panel, egui::RawInput::default());
        let box_at = at(&laid, "old name");
        let button_at = at(&laid, "✏");

        // Type into the box: select what is there, then replace it.
        run(&mut panel, click(box_at));
        let mut typing = egui::RawInput::default();
        typing.events = vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            },
            egui::Event::Text("new name".to_string()),
        ];
        run(&mut panel, typing);
        assert_eq!(panel.rows[0].name_edit, "new name", "the box did not take the typing");
        assert_eq!(
            panel.rows[0].name, "old name",
            "typing alone must not rename the row — the box has to be finished with"
        );

        // Now press it.
        run(&mut panel, click(button_at));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(panel.rows[0].name, "new name", "the button did not reach the row");
        assert!(
            panel.status.contains("new name"),
            "the status should say what happened: {}",
            panel.status
        );

        // The other way out of the box, and the one people actually take: type,
        // then click somewhere else entirely. The row is renamed all the same —
        // a name that only counted if you found the button beside it is a name
        // the project would go on without.
        let laid = run(&mut panel, egui::RawInput::default());
        let box_at = at(&laid, "new name");
        let away = at(&laid, "Project");
        run(&mut panel, click(box_at));
        let mut typing = egui::RawInput::default();
        typing.events = vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            },
            egui::Event::Text("clicked away".to_string()),
        ];
        run(&mut panel, typing);
        assert_eq!(panel.rows[0].name, "new name", "still a draft while the box has focus");

        // Escape is the way out that keeps the old name: it ends the box like
        // the others, and puts the draft back rather than over the row.
        let mut escape = egui::RawInput::default();
        escape.events = vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }];
        run(&mut panel, escape);
        run(&mut panel, egui::RawInput::default());
        assert_eq!(panel.rows[0].name, "new name", "Escape must not rename the row");
        assert_eq!(panel.rows[0].name_edit, "new name", "Escape must put the box back");

        // Type it again, now that Escape has thrown the first attempt away.
        run(&mut panel, click(box_at));
        let mut typing = egui::RawInput::default();
        typing.events = vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            },
            egui::Event::Text("clicked away".to_string()),
        ];
        run(&mut panel, typing);

        // The project has been saved before, so it knows where it lives…
        panel.set_project_dir(PathBuf::from("/nowhere/Song"), "Song".to_string());
        panel.take_request();
        run(&mut panel, click(away));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(
            panel.rows[0].name, "clicked away",
            "leaving the box must rename the row — otherwise the name is only ever on screen"
        );
        // …and the rename is written there, rather than waiting for a press of
        // Save the user has no reason to think is owed.
        match panel.take_request() {
            Some(ProjectRequest::Save { dir, name }) => {
                assert_eq!(dir, PathBuf::from("/nowhere/Song"));
                assert_eq!(name, "Song");
            }
            other => panic!("a rename should save the project in place, got {other:?}"),
        }
    }

    /// Saving a rename in place is for projects that already have a place. One
    /// that has never been saved is left alone: a folder dialog nobody opened is
    /// not what "I renamed a row" should mean.
    #[test]
    fn a_rename_in_an_unsaved_project_asks_for_nothing() {
        let mut panel = panel_with_rows(&[&[(60, frac(Fraction::Quarter), frac(Fraction::None))]]);
        panel.save_in_place();
        assert!(panel.take_request().is_none(), "an unsaved project must not be given a folder");

        // Nor does it write to the old folder once the name in the box has moved
        // on: that name belongs to a folder that does not exist yet, and only
        // Save gets to decide where it goes.
        panel.set_project_dir(PathBuf::from("/nowhere/Song"), "Song".to_string());
        panel.project_name = "Song Two".to_string();
        panel.save_in_place();
        assert!(
            panel.take_request().is_none(),
            "a renamed project must wait for Save, which is what picks the folder"
        );

        panel.project_name = "Song".to_string();
        panel.save_in_place();
        assert!(matches!(panel.take_request(), Some(ProjectRequest::Save { .. })));
    }

    /// The rename form refuses the two edits that are not renames: a box holding
    /// the name the row already has, and a box holding nothing. A row without a
    /// name could not be spoken about at all, so the blank is put back rather
    /// than taken.
    #[test]
    fn a_rename_to_nothing_leaves_the_row_named() {
        let mut row = row_with(&[]);
        row.set_name("kick".to_string());

        row.name_edit = "   ".to_string();
        assert!(!row.commit_name(), "whitespace is not a name");
        assert_eq!(row.name, "kick");
        assert_eq!(row.name_edit, "kick", "the box is put back to what the row is called");

        row.name_edit = "kick".to_string();
        assert!(!row.commit_name(), "renaming a row to its own name changes nothing");

        row.name_edit = "  kick 2  ".to_string();
        assert!(row.commit_name());
        assert_eq!(row.name, "kick 2", "the name is taken trimmed");
    }

    /// The two Add Track Row buttons, through the real widgets: which end of the
    /// list a new lane lands on is the whole difference between them, and a
    /// button that draws in the right place and inserts in the wrong one looks
    /// identical until the composition is read back.
    #[test]
    fn the_top_add_row_button_puts_the_lane_first_and_the_bottom_one_last() {
        let mut panel = panel_with_rows(&[
            &[(60, frac(Fraction::Quarter), frac(Fraction::None))],
            &[(48, frac(Fraction::Half), frac(Fraction::None))],
        ]);
        let (first, second) = (panel.rows[0].id, panel.rows[1].id);

        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let run = |panel: &mut ComposerPanel, input: egui::RawInput| -> Vec<(String, egui::Pos2)> {
            let out = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
            });
            let mut texts = Vec::new();
            fn walk(sh: &egui::Shape, out: &mut Vec<(String, egui::Pos2)>) {
                match sh {
                    egui::Shape::Text(t) => out.push((t.galley.text().to_string(), t.pos)),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            for cs in &out.shapes {
                walk(&cs.shape, &mut texts);
            }
            texts
        };
        let click = |at: egui::Pos2| {
            let mut i = egui::RawInput::default();
            i.events = vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ];
            i
        };
        // Both buttons, top one first — they are told apart by where they are,
        // which is the only thing that distinguishes them on screen either.
        let buttons = |texts: &[(String, egui::Pos2)]| -> Vec<egui::Pos2> {
            let mut ps: Vec<egui::Pos2> = texts
                .iter()
                .filter(|(t, _)| t == "➕ Add Track Row")
                .map(|(_, p)| egui::pos2(p.x + 4.0, p.y + 6.0))
                .collect();
            ps.sort_by(|a, b| a.y.total_cmp(&b.y));
            ps
        };

        let laid = buttons(&run(&mut panel, egui::RawInput::default()));
        assert_eq!(laid.len(), 2, "one button above the lanes and one below");

        run(&mut panel, click(laid[0]));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(panel.rows.len(), 3, "the top button did not add a row");
        assert_eq!(
            panel.rows.iter().map(|r| r.id).skip(1).collect::<Vec<_>>(),
            vec![first, second],
            "the new lane did not go in front — the rows it pushed down are out of order"
        );
        assert!(
            panel.rows[0].items.is_empty(),
            "the first row is not the new empty one"
        );

        // Laid out again: a row has been added, so the bottom button has moved.
        let laid = buttons(&run(&mut panel, egui::RawInput::default()));
        run(&mut panel, click(laid[1]));
        run(&mut panel, egui::RawInput::default());
        assert_eq!(panel.rows.len(), 4, "the bottom button did not add a row");
        assert_eq!(
            panel.rows.iter().map(|r| r.id).collect::<Vec<_>>()[1..3],
            [first, second],
            "the bottom button inserted somewhere other than the end"
        );
        assert!(
            panel.rows[3].items.is_empty(),
            "the last row is not the new empty one"
        );
    }

    /// TEMPORARY: render a project folder to a wav, offline. Not part of the
    /// suite (`#[ignore]`); it exists to hear a hand-written manifest.
    ///
    ///   RESYNTH_PROJECT=/path/to/X.rsyn RESYNTH_OUT=/tmp/x.wav \
    ///     cargo test --lib render_named_project -- --ignored --nocapture
    #[test]
    #[ignore]
    fn render_named_project() {
        let file = std::path::PathBuf::from(std::env::var("RESYNTH_PROJECT").unwrap());
        let out = std::env::var("RESYNTH_OUT").unwrap();
        let dir = file.parent().unwrap().to_path_buf();
        let proj = project::Project::read(&file).unwrap();
        let spu = 60.0 / (proj.tempo_bpm.max(1.0) as f64) / UNITS_PER_BEAT as f64;
        let internal = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("internal_plugins")
            .join("liblesynth_fourier.so");

        let mut plans = Vec::new();
        for (i, prow) in proj.rows.iter().enumerate() {
            if !prow.enabled {
                continue;
            }
            if let Ok(solo) = std::env::var("RESYNTH_SOLO") {
                if solo.parse::<usize>() != Ok(i) {
                    continue;
                }
            }
            let read_state = |f: &Option<String>| {
                f.as_ref().and_then(|f| std::fs::read(dir.join(f)).ok())
            };
            let source = match &prow.source {
                project::TrackSource::LeSynth { file, state } => {
                    crate::gui::registry::PlaybackSource {
                        name: prow.track_name.clone(),
                        plugin_path: internal.clone(),
                        class_id: Some(crate::vst::class_ids::FOURIER_SYNTH),
                        plugin_id: None,
                        is_lesynth: true,
                        state: Some(
                            crate::track_format::TrackState::read(&dir.join(file)).unwrap(),
                        ),
                        vst_state: read_state(state),
                        wav: None,
                    }
                }
                project::TrackSource::Vst { path, class_id, plugin_id, state } => {
                    crate::gui::registry::PlaybackSource {
                        name: prow.track_name.clone(),
                        plugin_path: path.clone(),
                        class_id: *class_id,
                        plugin_id: plugin_id.clone(),
                        is_lesynth: false,
                        state: None,
                        vst_state: read_state(state),
                        wav: None,
                    }
                }
                other => panic!("unsupported source {other:?}"),
            };
            let mut at = prow.lead.units();
            let mut notes = Vec::new();
            for item in &prow.items {
                let units = item.dur.units();
                if units > 0 {
                    notes.push(player::PlannedNote {
                        at_secs: at as f64 * spu,
                        dur_secs: units as f64 * spu,
                        pitch: item.pitch,
                        start_secs: item.start as f64,
                    });
                }
                at += item.total_units();
            }
            println!("row {i}: {} — {} note(s), gain {}", prow.track_name, notes.len(), prow.gain);
            plans.push(player::RowPlan {
                row_id: i as u64,
                source,
                gain: prow.gain,
                notes,
            });
        }

        let (samples, rows, total) = player::render_offline(plans, 44100.0, 2).unwrap();
        println!("rendered {rows}/{total} rows, {:.2}s", samples.len() as f64 / 2.0 / 44100.0);
        crate::audio::write_wav_i16(std::path::Path::new(&out), &samples, 2, 44100).unwrap();
    }

    /// The align buttons line a space up on the one note selected on another
    /// row: on the lead, the row's first note then starts exactly where that
    /// note does.
    #[test]
    fn align_lead_starts_the_row_on_the_selected_note() {
        let q = frac(Fraction::Quarter);
        let e = frac(Fraction::Eighth);
        let mut panel = panel_with_rows(&[&[(60, q, e), (62, q, e), (64, q, e)], &[(48, q, e)]]);
        panel.rows[0].lead = Duration::new(1, Fraction::None);
        let cue = panel.rows[0].items[2].id;
        panel.rows[0].select_item(cue, false);

        panel.align_space(1, None, AlignTo::Start);

        let want = UNITS_PER_WHOLE + 2 * (QUARTER + EIGHTH);
        assert_eq!(panel.rows[1].lead.units(), want);
        assert_eq!(note_starts(&panel)[1][0], note_starts(&panel)[0][2]);
        assert_eq!(panel.rows[0].sel, None, "the cue is unticked once it has been used");
    }

    /// A space behind a note works the same, measured from where the note in
    /// front of it ends; aligned to the end, the next note lands where the
    /// cue's own space runs out.
    #[test]
    fn align_space_to_the_end_of_the_selected_note() {
        let q = frac(Fraction::Quarter);
        let e = frac(Fraction::Eighth);
        let mut panel = panel_with_rows(&[&[(60, q, e), (62, q, e), (64, q, e)], &[(48, q, e), (50, q, e)]]);
        let cue = panel.rows[0].items[1].id;
        panel.rows[0].select_item(cue, false);

        panel.align_space(1, Some(0), AlignTo::End);

        assert_eq!(note_starts(&panel)[1][1], note_starts(&panel)[0][2]);
        assert_eq!(panel.rows[1].items[0].space.units(), QUARTER + 2 * EIGHTH);
        assert_eq!(panel.rows[0].sel, None);

        panel.rows[0].select_item(cue, false);
        panel.align_space(1, Some(0), AlignTo::Start);
        assert_eq!(note_starts(&panel)[1][1], note_starts(&panel)[0][1]);
    }

    /// A space that already starts past the cue cannot reach back to it: it is
    /// left alone, and so is the selection, so another space can be tried.
    #[test]
    fn align_space_past_the_cue_changes_nothing() {
        let q = frac(Fraction::Quarter);
        let e = frac(Fraction::Eighth);
        let mut panel = panel_with_rows(&[&[(60, q, e), (62, q, e)], &[(48, q, e), (50, q, e)]]);
        panel.rows[1].lead = Duration::new(1, Fraction::None);
        let cue = panel.rows[0].items[1].id;
        panel.rows[0].select_item(cue, false);

        panel.align_space(1, Some(0), AlignTo::End);

        assert_eq!(panel.rows[1].items[0].space, e, "the space was changed");
        assert_eq!(panel.rows[0].sel, Some((cue, cue)), "the cue was unticked");
    }

    /// Anything but exactly one note on another row leaves the space alone: no
    /// selection, a block of two, or the note being on the row itself.
    #[test]
    fn align_space_needs_exactly_one_note_on_another_row() {
        let q = frac(Fraction::Quarter);
        let e = frac(Fraction::Eighth);
        let mut panel = panel_with_rows(&[&[(60, q, e), (62, q, e)], &[(48, q, e)]]);
        let keep = frac(Fraction::Sixteenth);
        panel.rows[1].lead = keep;

        panel.align_space(1, None, AlignTo::Start);
        assert_eq!(panel.rows[1].lead, keep, "nothing selected");

        panel.rows[0].select_all();
        panel.align_space(1, None, AlignTo::Start);
        assert_eq!(panel.rows[1].lead, keep, "two notes selected");

        panel.rows[0].sel = None;
        let own = panel.rows[1].items[0].id;
        panel.rows[1].select_item(own, false);
        panel.align_space(1, None, AlignTo::Start);
        assert_eq!(panel.rows[1].lead, keep, "the note is on this row");

        let other = panel.rows[0].items[1].id;
        panel.rows[0].select_item(other, false);
        panel.align_space(1, None, AlignTo::End);
        assert_eq!(panel.rows[1].lead, keep, "one note on each row is two notes");
    }

    /// The align buttons' icons are real glyphs in egui's fonts, not tofu.
    #[test]
    fn align_icons_have_glyphs() {
        let ctx = egui::Context::default();
        let _ = ctx.run(Default::default(), |_| {});
        let id = egui::FontId::proportional(14.0);
        for to in [AlignTo::Start, AlignTo::End] {
            assert!(ctx.fonts(|f| f.has_glyphs(&id, to.icon())), "{} tofus", to.icon());
        }
    }
}
