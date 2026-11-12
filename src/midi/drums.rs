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
//! What a note means on a drum instrument.
//!
//! A pitch is only ever a MIDI note number — the host sends the number and the
//! plugin decides what it hits. For a melodic instrument the number *is* the
//! answer; for a drum kit it is a lookup, and "C2" tells the user nothing about
//! whether they just wrote a kick or a hi-hat.
//!
//! **General MIDI percussion is the map nearly every kit follows.** GM level 1
//! fixes notes 35–81, GM level 2 extends it to 27–87, and drum plugins built for
//! DAW work overwhelmingly ship that layout — jdrummer, for one, maps its
//! sixteen pads onto exactly the GM notes (kick 36, snare 38, closed hat 42,
//! toms 41/43/45/47/48, crash 49, ride 51).
//!
//! **Naming, not numbering, is where this goes wrong.** Drum charts almost
//! always call note 36 "C1" — they name middle C as C3, where this host names it
//! C4 — so the very same note reads "C2" in the Sequencer. Go by the numbers.
//!
//! So the hint here is *General MIDI's* answer, not the loaded plugin's: a
//! sampler with sixteen chromatic pads will still sound whatever is on the pad,
//! whatever this calls it. It is shown beside the note name, never instead of
//! it, so a hint that does not match the kit costs the user nothing.

/// The General MIDI percussion name for a note, if it has one (GM2 range,
/// 27–87). `None` for notes outside it.
pub fn gm_percussion_name(note: u8) -> Option<&'static str> {
    Some(match note {
        27 => "High Q",
        28 => "Slap",
        29 => "Scratch Push",
        30 => "Scratch Pull",
        31 => "Sticks",
        32 => "Square Click",
        33 => "Metronome Click",
        34 => "Metronome Bell",
        35 => "Acoustic Bass Drum",
        36 => "Bass Drum (Kick)",
        37 => "Side Stick / Rim",
        38 => "Acoustic Snare",
        39 => "Hand Clap",
        40 => "Electric Snare",
        41 => "Low Floor Tom",
        42 => "Closed Hi-Hat",
        43 => "High Floor Tom",
        44 => "Pedal Hi-Hat",
        45 => "Low Tom",
        46 => "Open Hi-Hat",
        47 => "Low-Mid Tom",
        48 => "Hi-Mid Tom",
        49 => "Crash Cymbal 1",
        50 => "High Tom",
        51 => "Ride Cymbal 1",
        52 => "Chinese Cymbal",
        53 => "Ride Bell",
        54 => "Tambourine",
        55 => "Splash Cymbal",
        56 => "Cowbell",
        57 => "Crash Cymbal 2",
        58 => "Vibraslap",
        59 => "Ride Cymbal 2",
        60 => "Hi Bongo",
        61 => "Low Bongo",
        62 => "Mute Hi Conga",
        63 => "Open Hi Conga",
        64 => "Low Conga",
        65 => "High Timbale",
        66 => "Low Timbale",
        67 => "High Agogo",
        68 => "Low Agogo",
        69 => "Cabasa",
        70 => "Maracas",
        71 => "Short Whistle",
        72 => "Long Whistle",
        73 => "Short Guiro",
        74 => "Long Guiro",
        75 => "Claves",
        76 => "Hi Wood Block",
        77 => "Low Wood Block",
        78 => "Mute Cuica",
        79 => "Open Cuica",
        80 => "Mute Triangle",
        81 => "Open Triangle",
        82 => "Shaker",
        83 => "Jingle Bell",
        84 => "Bell Tree",
        85 => "Castanets",
        86 => "Mute Surdo",
        87 => "Open Surdo",
        _ => return None,
    })
}

/// Product names that are drum kits without saying so. Everything whose name
/// already contains "drum" (jdrummer included) is caught by the substrings
/// below and does not belong here.
const KNOWN_DRUM_PLUGINS: &[&str] = &[
    "sitala", "battery", "maschine", "geist", "hydrogen", "808", "909", "hats",
];

/// Whether a plugin plays a drum kit, and so should have its notes named.
///
/// `subcategories` is the VST3 class's own declaration (`Instrument|Drum` — the
/// precise answer, when a plugin bothers to make it). Plenty do not: jdrummer
/// calls itself `Instrument|Synth`, so the name is the fallback.
///
/// Being wrong here is cheap in both directions — the hint is additive, and a
/// missing one leaves the plain note name — so the name test is deliberately
/// generous rather than a curated list of products.
pub fn plays_a_drum_kit(name: &str, subcategories: &str) -> bool {
    if subcategories
        .split('|')
        .any(|c| c.trim().eq_ignore_ascii_case("Drum"))
    {
        return true;
    }
    let name = name.to_ascii_lowercase();
    ["drum", "percussion", "kit", "beats"]
        .iter()
        .chain(KNOWN_DRUM_PLUGINS)
        .any(|needle| name.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The notes a beat is actually written from, which are the ones a wrong
    /// table would be noticed on first.
    #[test]
    fn the_core_kit_is_the_general_midi_one() {
        assert_eq!(gm_percussion_name(36), Some("Bass Drum (Kick)"));
        assert_eq!(gm_percussion_name(38), Some("Acoustic Snare"));
        assert_eq!(gm_percussion_name(42), Some("Closed Hi-Hat"));
        assert_eq!(gm_percussion_name(46), Some("Open Hi-Hat"));
        assert_eq!(gm_percussion_name(49), Some("Crash Cymbal 1"));
        assert_eq!(gm_percussion_name(51), Some("Ride Cymbal 1"));
        // The GM2 percussion range and nothing outside it.
        assert_eq!(gm_percussion_name(26), None);
        assert_eq!(gm_percussion_name(88), None);
        assert!((27..=87).all(|n| gm_percussion_name(n).is_some()));
    }

    #[test]
    fn a_drum_plugin_is_recognised_by_its_category_or_its_name() {
        // The precise signal, when a plugin makes it.
        assert!(plays_a_drum_kit("Anything", "Instrument|Drum"));
        // jdrummer does not: it calls itself a synth, so the name answers.
        assert!(plays_a_drum_kit("jdrummer", "Instrument|Synth"));
        assert!(plays_a_drum_kit("MT Power Drum Kit 2", ""));
        assert!(plays_a_drum_kit("Sitala", ""));
        // And a synth is not a drum kit however it is spelled.
        assert!(!plays_a_drum_kit("Vital", "Instrument|Synth"));
        assert!(!plays_a_drum_kit("Surge XT", ""));
        assert!(!plays_a_drum_kit("Dexed", "Instrument|Synth"));
        assert!(!plays_a_drum_kit("LeSynth", ""));
    }
}
