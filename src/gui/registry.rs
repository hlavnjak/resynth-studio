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
//! The one list of **Instruments** every panel agrees on.
//!
//! An Instrument is a playable sound source — a plugin instrument from the Instruments
//! panel or a subtrack published from Resynthesis. Sequencer rows map onto one by
//! id, so the registry, not any single panel, is what "all Instruments" means.
//!
//! Entries are *recipes*, not live plugins: a library path, a class id, and the
//! state to load into a fresh instance — LeSynth's harmonic grid, or any other
//! plugin's own `IComponent` state — so a sequence plays whether or not the
//! instrument's editor is open. When it is, [`InstrumentEntry::live`] points at that
//! instance, so the Sequencer can snapshot what is being edited right now.
//!
//! One kind of instrument is not a plugin at all: a **wav instrument** ([`InstrumentEntry::wav`])
//! is an audio file, played back whole as one note. Its recipe is the path, and
//! nothing else — there is no instance to load, no state to import and no editor
//! to snapshot.
//!
//! GUI-thread only (`Rc<RefCell<_>>`) — every user of it is an egui panel.

use std::cell::{Ref, RefCell};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Weak;

use crate::track_format::TrackState;
use crate::plugin::PluginInstance;

/// One registered Instrument.
pub struct InstrumentEntry {
    /// Stable id. Sequencer rows store this, so a row survives instruments being added
    /// and removed around it.
    pub id: u64,
    pub name: String,
    /// Library to load for playback.
    pub plugin_path: PathBuf,
    /// Class to select from the factory; `None` takes the first.
    pub class_id: Option<[i8; 16]>,
    /// Which plugin inside the file, for formats where one file holds several:
    /// a CLAP plugin id, an LV2 plugin URI, or an Audio Unit's codes. `None`
    /// takes the first. See [`crate::plugin`].
    pub plugin_id: Option<String>,
    /// LeSynth instruments can carry a grid and be tagged for state import/export.
    pub is_lesynth: bool,
    /// This instrument plays a drum kit, so its notes are named after what they hit
    /// rather than left as pitches. See [`crate::midi::plays_a_drum_kit`].
    pub percussion: bool,
    /// The audio file this instrument *is*, played whole as one note — an instrument with
    /// no plugin behind it at all. `Some` makes every plugin field above inert:
    /// nothing is loaded, no state is imported, and the note's pitch is ignored
    /// (a file has the pitch it was recorded at). Published from the
    /// Resynthesis panel, and saved in a project as this path.
    pub wav: Option<PathBuf>,
    /// How long that file runs. Not playback's business — it decodes the file
    /// and knows — but the Sequencer's: it is the range of the slider that says
    /// where in the file a note starts, and a range short of the recording puts
    /// part of it out of reach.
    pub wav_secs: f32,
    /// Grid to import into a freshly loaded instance. `None` = the plugin's own
    /// default state (a plain synth-mode LeSynth, or any custom VST).
    pub state: Option<TrackState>,
    /// The same idea for every other plugin: an opaque `IComponent` state, which
    /// is where a third-party VST3 keeps the knob positions the user set in its
    /// editor. `None` = whatever the plugin starts up with.
    pub vst_state: Option<Vec<u8>>,
    /// The instance behind this instrument's open editor, if any. Weak so a closed
    /// editor is simply gone; the entry itself outlives it.
    pub live: Option<Weak<PluginInstance>>,
}

/// Everything the Sequencer needs to load and configure its own instance of a
/// instrument — a value, so it can be built while the registry is borrowed and used
/// after the borrow ends.
#[derive(Clone)]
pub struct PlaybackSource {
    pub name: String,
    pub plugin_path: PathBuf,
    pub class_id: Option<[i8; 16]>,
    pub plugin_id: Option<String>,
    pub is_lesynth: bool,
    pub state: Option<TrackState>,
    pub vst_state: Option<Vec<u8>>,
    /// See [`InstrumentEntry::wav`]. `Some` means the row plays this file rather than
    /// any of the fields above.
    pub wav: Option<PathBuf>,
}

struct Inner {
    entries: Vec<InstrumentEntry>,
    next_id: u64,
}

/// Shared handle to the instrument list. Cloning shares, it does not copy.
#[derive(Clone)]
pub struct InstrumentRegistry(Rc<RefCell<Inner>>);

impl Default for InstrumentRegistry {
    fn default() -> Self {
        Self(Rc::new(RefCell::new(Inner {
            entries: Vec::new(),
            next_id: 0,
        })))
    }
}

impl InstrumentRegistry {
    /// Register an instrument and return its id.
    pub fn add(
        &self,
        name: impl Into<String>,
        plugin_path: PathBuf,
        class_id: Option<[i8; 16]>,
        is_lesynth: bool,
        state: Option<TrackState>,
    ) -> u64 {
        let mut inner = self.0.borrow_mut();
        let id = inner.next_id;
        inner.next_id += 1;
        inner.entries.push(InstrumentEntry {
            id,
            name: name.into(),
            plugin_path,
            class_id,
            plugin_id: None,
            is_lesynth,
            percussion: false,
            wav: None,
            wav_secs: 0.0,
            state,
            vst_state: None,
            live: None,
        });
        id
    }

    /// Register an audio file as an instrument that plays it whole, as one note, and
    /// return its id. The path is the whole recipe — see [`InstrumentEntry::wav`];
    /// `secs` is how long the file runs, which only the Sequencer's start slider
    /// needs.
    pub fn add_wav(&self, name: impl Into<String>, path: PathBuf, secs: f32) -> u64 {
        let mut inner = self.0.borrow_mut();
        let id = inner.next_id;
        inner.next_id += 1;
        inner.entries.push(InstrumentEntry {
            id,
            name: name.into(),
            // A wav instrument has no library; the file it plays is what it is, and
            // naming it here is what lets a message about the instrument say so.
            plugin_path: path.clone(),
            class_id: None,
            plugin_id: None,
            is_lesynth: false,
            percussion: false,
            wav: Some(path),
            wav_secs: secs.max(0.0),
            state: None,
            vst_state: None,
            live: None,
        });
        id
    }

    /// The wav instrument playing `path`, if one is already registered. A file is one
    /// instrument however many times it is asked for: opening it again in Resynthesis,
    /// or loading a project that names it, finds the entry rather than adding a
    /// second one whose rows would be indistinguishable from the first's.
    pub fn find_wav(&self, path: &Path) -> Option<u64> {
        self.0
            .borrow()
            .entries
            .iter()
            .find(|e| e.wav.as_deref() == Some(path))
            .map(|e| e.id)
    }

    /// How long the file this instrument plays runs, or `None` for an instrument that is
    /// not a wav instrument (or is gone). This is also the question "is this a wav
    /// instrument?": what the Sequencer does with the answer — no pitch box, and a
    /// start slider over the file — needs the length in the same breath.
    pub fn wav_secs(&self, id: u64) -> Option<f32> {
        let inner = self.0.borrow();
        let entry = inner.entries.iter().find(|e| e.id == id)?;
        entry.wav.as_ref().map(|_| entry.wav_secs)
    }

    /// Whether this instrument plays an audio file rather than a plugin. `false` for
    /// an instrument that is gone.
    pub fn is_wav(&self, id: u64) -> bool {
        self.wav_secs(id).is_some()
    }

    /// Record that this instrument plays a drum kit, which is what puts drum names in
    /// the Sequencer's note pickers.
    pub fn set_percussion(&self, id: u64, percussion: bool) {
        if let Some(e) = self.0.borrow_mut().entries.iter_mut().find(|e| e.id == id) {
            e.percussion = percussion;
        }
    }

    /// Whether this instrument plays a drum kit. `false` for an instrument that is gone.
    pub fn is_percussion(&self, id: u64) -> bool {
        self.0
            .borrow()
            .entries
            .iter()
            .find(|e| e.id == id)
            .is_some_and(|e| e.percussion)
    }

    /// Name the plugin inside the instrument's file — see [`InstrumentEntry::plugin_id`].
    pub fn set_plugin_id(&self, id: u64, plugin_id: Option<String>) {
        if let Some(e) = self.0.borrow_mut().entries.iter_mut().find(|e| e.id == id) {
            e.plugin_id = plugin_id;
        }
    }

    /// Remember a plugin's own state for this instrument — what a freshly loaded
    /// instance is given, and what a project save writes out. Set when a custom
    /// plugin's editor closes, and when a project is loaded.
    pub fn set_vst_state(&self, id: u64, state: Option<Vec<u8>>) {
        if let Some(e) = self.0.borrow_mut().entries.iter_mut().find(|e| e.id == id) {
            e.vst_state = state;
        }
    }

    /// Remember a LeSynth grid for this instrument, the same way — set when its
    /// editor closes, so the Sequencer keeps playing what was edited.
    pub fn set_state(&self, id: u64, state: Option<TrackState>) {
        if let Some(e) = self.0.borrow_mut().entries.iter_mut().find(|e| e.id == id) {
            if state.is_some() {
                e.state = state;
            }
        }
    }

    pub fn remove(&self, id: u64) {
        self.0.borrow_mut().entries.retain(|e| e.id != id);
    }

    /// Point this instrument at the live instance behind its open editor, or clear it
    /// (`None`) when the editor closes.
    pub fn set_live(&self, id: u64, live: Option<Weak<PluginInstance>>) {
        if let Some(e) = self.0.borrow_mut().entries.iter_mut().find(|e| e.id == id) {
            e.live = live;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.borrow().entries.is_empty()
    }

    pub fn contains(&self, id: u64) -> bool {
        self.0.borrow().entries.iter().any(|e| e.id == id)
    }

    pub fn first_id(&self) -> Option<u64> {
        self.0.borrow().entries.first().map(|e| e.id)
    }

    pub fn name_of(&self, id: u64) -> Option<String> {
        self.0
            .borrow()
            .entries
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.name.clone())
    }

    /// `(id, name)` for every instrument, for the row select box.
    pub fn list(&self) -> Vec<(u64, String)> {
        self.0
            .borrow()
            .entries
            .iter()
            .map(|e| (e.id, e.name.clone()))
            .collect()
    }

    pub fn entries(&self) -> Ref<'_, [InstrumentEntry]> {
        Ref::map(self.0.borrow(), |i| i.entries.as_slice())
    }

    /// What to load to play instrument `id`. The grid comes from the live editor when
    /// one is open — the user is composing with what they can hear there, not
    /// with the state the instrument was registered in — and from the registered
    /// state otherwise.
    pub fn playback_source(&self, id: u64) -> Option<PlaybackSource> {
        let inner = self.0.borrow();
        let entry = inner.entries.iter().find(|e| e.id == id)?;
        // A wav instrument is its file. There is no instance to ask what it is
        // playing, and nothing about it changes while a sequence is open.
        if let Some(path) = &entry.wav {
            return Some(PlaybackSource {
                name: entry.name.clone(),
                plugin_path: entry.plugin_path.clone(),
                class_id: None,
                plugin_id: None,
                is_lesynth: false,
                state: None,
                vst_state: None,
                wav: Some(path.clone()),
            });
        }
        let live = entry.live.as_ref().and_then(Weak::upgrade);

        // Whichever kind of state this instrument carries, prefer the open editor's:
        // the user is composing with what they can hear there.
        let (mut live_grid, mut live_vst) = (None, None);
        if let Some(plugin) = live {
            if entry.is_lesynth {
                match plugin.export_state() {
                    Ok(s) => live_grid = Some(s),
                    Err(e) => log::debug!("live grid unavailable for '{}': {e}", entry.name),
                }
            }
            // Every plugin keeps its knobs in its own opaque state — LeSynth too,
            // where that is the whole Synth editor: the curve type, offset and
            // granularity of each harmonic and the nested-Fourier sliders under
            // them, none of which is in the grid.
            match plugin.save_state() {
                Ok(bytes) if !bytes.is_empty() => live_vst = Some(bytes),
                Ok(_) => {}
                Err(e) => log::debug!("live state unavailable for '{}': {e:#}", entry.name),
            }
        }
        Some(PlaybackSource {
            name: entry.name.clone(),
            plugin_path: entry.plugin_path.clone(),
            class_id: entry.class_id,
            plugin_id: entry.plugin_id.clone(),
            is_lesynth: entry.is_lesynth,
            state: live_grid.or_else(|| entry.state.clone()),
            vst_state: live_vst.or_else(|| entry.vst_state.clone()),
            wav: None,
        })
    }
}
