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
pub mod drums;
pub mod input;

pub use drums::{gm_percussion_name, plays_a_drum_kit};
pub use input::{
    add_midi_tap, list_midi_ports, list_usb_midi_keyboards, new_midi_queue, new_midi_taps,
    new_octave_shift, KeyboardTarget, MidiEventQueue, MidiFeed, MidiRouter, MidiTap, MidiTaps, OctaveShift,
    MAX_OCTAVE_SHIFT,
};