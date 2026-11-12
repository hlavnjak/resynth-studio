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
use std::sync::atomic::Ordering;

use anyhow::Context;
use eframe::egui;

use crate::midi::{self, MidiRouter, OctaveShift, MAX_OCTAVE_SHIFT};

use super::sequencer::project::{self, Project, InstrumentSource};
use super::sequencer::{SequencerPanel, ProjectRequest};
use super::registry::InstrumentRegistry;
use super::resynth::ResynthPanel;
use super::instrument::InstrumentsPanel;

pub struct DawApp {
    midi_status: String,
    midi_ports: Vec<String>,
    selected_midi_port: Option<String>,
    usb_keyboards: Vec<String>,
    selected_usb_keyboard: Option<String>,

    // Runtime state
    /// One keyboard per plugin instance: which port feeds what, and the
    /// connections behind them. Replaces the single shared queue every editor
    /// used to drain — two editors on it each heard half of what was played.
    midi_router: MidiRouter,
    /// How far the keyboard is transposed on the way in. Shared with the input
    /// thread, so the picker moves a connection that is already open.
    octave_shift: OctaveShift,

    // Plugin instruments (LeSynth Fourier / custom VST), each with its own editor.
    instruments: InstrumentsPanel,
    // Arrange the registered instruments on a timeline and play them.
    sequencer: SequencerPanel,
    // Resynthesis (.wav/.mp3/.m4a → LeSynth Fourier analysis)
    resynth: ResynthPanel,
    // The shared instrument list, kept here too so saving a project can read every
    // instrument — including subtracks published straight from Resynthesis, which
    // never pass through the Instruments panel.
    registry: InstrumentRegistry,
    /// Wav instruments a loaded project asked for. They play a file, so they have no
    /// plugin, no editor and no place in the Instruments panel — but something has to
    /// take them back out of the registry when the next project replaces them,
    /// the way [`InstrumentsPanel::clear`] does for its own. A wav instrument published in
    /// this session from Resynthesis is *not* in here: that panel owns it, and
    /// loading a project leaves it alone.
    adopted_wavs: Vec<u64>,
}

impl Default for DawApp {
    fn default() -> Self {
        let midi_taps = midi::new_midi_taps();
        let octave_shift = midi::new_octave_shift();
        let midi_router = MidiRouter::new(octave_shift.clone(), midi_taps.clone());
        // The one instrument list the panels share: Instruments and Resynthesis publish
        // into it, the Sequencer builds its rows from it.
        let registry = InstrumentRegistry::default();
        Self {
            midi_status: "Disconnected".to_string(),
            midi_ports: Vec::new(),
            selected_midi_port: None,
            usb_keyboards: Vec::new(),
            selected_usb_keyboard: None,
            instruments: InstrumentsPanel::new(midi_router.clone(), registry.clone()),
            sequencer: SequencerPanel::new(registry.clone(), midi_taps.clone()),
            midi_router,
            octave_shift,
            resynth: ResynthPanel::new(registry.clone()),
            registry,
            adopted_wavs: Vec::new(),
        }
    }
}

impl DawApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // eframe has the main window's GL context current while it builds the app.
        #[cfg(target_os = "macos")]
        super::editor_window::remember_main_gl_context();
        Self::configure_style(&cc.egui_ctx);
        super::editor_window::set_host_context(&cc.egui_ctx);
        let mut app = Self::default();
        app.refresh_midi_ports();
        app
    }

    /// Apply a consistent, slightly roomier look across the whole app.
    ///
    /// `pub(crate)` so a test that measures a panel's layout can lay it out
    /// under the text sizes and spacing the user actually sees: egui's defaults
    /// are smaller, and a card that fits under them can still overflow here.
    pub(crate) fn configure_style(ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(10.0, 6.0);
        // A line starts this tall, and a button padded as above is 28 px: start
        // it any shorter and each thing on it is centred on a line still
        // growing, so select boxes drift down below the buttons beside them.
        style.spacing.interact_size.y = 28.0;
        style.spacing.indent = 16.0;
        // Bump the default body/heading text a touch for legibility.
        use egui::{FontFamily::Proportional, FontId, TextStyle};
        style.text_styles = [
            (TextStyle::Heading, FontId::new(18.0, Proportional)),
            (TextStyle::Body, FontId::new(14.0, Proportional)),
            (TextStyle::Button, FontId::new(14.0, Proportional)),
            (TextStyle::Monospace, FontId::new(13.0, egui::FontFamily::Monospace)),
            (TextStyle::Small, FontId::new(11.0, Proportional)),
        ]
        .into();
        ctx.set_style(style);
    }

    /// Draw a titled "card": a bordered group with a heading and a separator,
    /// used to give every top-level section the same framed look.
    fn section<R>(
        ui: &mut egui::Ui,
        title: &str,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> R {
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::same(12))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.heading(title);
                });
                ui.separator();
                ui.add_space(2.0);
                add_contents(ui)
            })
            .inner
    }

    /// Colour a status line by its apparent sentiment (error / success / neutral).
    fn status_color(status: &str) -> egui::Color32 {
        let lower = status.to_ascii_lowercase();
        if ["fail", "error", "unavailable", "could not", "no "]
            .iter()
            .any(|k| lower.contains(k))
        {
            egui::Color32::from_rgb(230, 120, 110)
        } else if ["loaded", "playing", "connected", "opened", "decoded", "removed"]
            .iter()
            .any(|k| lower.contains(k))
        {
            egui::Color32::from_rgb(130, 210, 150)
        } else {
            egui::Color32::from_gray(170)
        }
    }

    /// A status line rendered with sentiment colouring.
    fn status_label(ui: &mut egui::Ui, status: &str) {
        ui.label(egui::RichText::new(status).color(Self::status_color(status)));
    }

    fn refresh_midi_ports(&mut self) {
        self.midi_ports = midi::input::list_midi_ports().unwrap_or_default();
        self.usb_keyboards = midi::list_usb_midi_keyboards().unwrap_or_default();
    }

    fn connect_midi(&mut self) {
        // Prefer USB keyboard selection, fall back to general MIDI port
        let port_filter = self
            .selected_usb_keyboard
            .clone()
            .or_else(|| self.selected_midi_port.clone());
        // The picker offers the ports as the driver names them, so a chosen one
        // is a name to connect by; with nothing chosen, the first port there is.
        let port = match port_filter.or_else(|| self.midi_ports.first().cloned()) {
            Some(p) => p,
            None => {
                self.midi_status = "No MIDI input ports found.".to_string();
                return;
            }
        };
        match self.midi_router.set_default_port(Some(port.clone())) {
            Ok(()) => self.midi_status = format!("Connected: {port}"),
            Err(e) => self.midi_status = format!("MIDI error: {e:#}"),
        }
    }

    fn midi_section(&mut self, ui: &mut egui::Ui) {
        Self::section(ui, "MIDI", |ui| {
            // Lay the two device pickers out in a grid so their labels and
            // combo boxes share aligned columns instead of drifting out of line
            // when placed side by side in a wrapping row.
            egui::Grid::new("midi_devices")
                .num_columns(2)
                .spacing([8.0, 6.0])
                .show(ui, |ui| {
                    // USB keyboard picker
                    ui.label("USB keyboard:");
                    let usb_label = self
                        .selected_usb_keyboard
                        .clone()
                        .unwrap_or_else(|| "Select USB keyboard…".to_string());
                    // The list length is part of the id: a `ComboBox` popup is
                    // an `egui::Area`, and an `Area` measures itself only on the
                    // first pass it is shown for a given id, then reuses that
                    // size forever — so a picker first opened with two entries
                    // clips away everything a later rescan finds. See the
                    // Sequencer's instrument select box for the long version.
                    egui::ComboBox::from_id_salt(("usb_keyboard", self.usb_keyboards.len()))
                        .width(260.0)
                        .selected_text(usb_label)
                        .show_ui(ui, |ui| {
                            if self.usb_keyboards.is_empty() {
                                ui.label("No USB keyboards detected");
                            }
                            for kb in self.usb_keyboards.clone() {
                                ui.selectable_value(
                                    &mut self.selected_usb_keyboard,
                                    Some(kb.clone()),
                                    kb,
                                );
                            }
                        });
                    ui.end_row();

                    // General MIDI port picker
                    ui.label("MIDI port:");
                    let port_label = self
                        .selected_midi_port
                        .clone()
                        .unwrap_or_else(|| "Select MIDI port…".to_string());
                    egui::ComboBox::from_id_salt(("midi_port", self.midi_ports.len()))
                        .width(260.0)
                        .selected_text(port_label)
                        .show_ui(ui, |ui| {
                            for port in self.midi_ports.clone() {
                                ui.selectable_value(
                                    &mut self.selected_midi_port,
                                    Some(port.clone()),
                                    port,
                                );
                            }
                        });
                    ui.end_row();

                    // Which instrument the panel's keyboard plays. Every open
                    // editor following the panel's port hears it otherwise,
                    // so three open instruments are three sounds per key.
                    ui.label("Plays instrument:");
                    let choices = self.instruments.keyboard_choices();
                    let current = self.instruments.keyboard_target();
                    let mut target = current;
                    let target_label = match current {
                        None => "All open instruments".to_string(),
                        Some(id) => choices
                            .iter()
                            .find(|(i, _)| *i == id)
                            .map(|(_, name)| name.clone())
                            .unwrap_or_else(|| "All open instruments".to_string()),
                    };
                    egui::ComboBox::from_id_salt(("midi_instrument", choices.len()))
                        .width(260.0)
                        .selected_text(target_label)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut target, None, "All open instruments");
                            for (id, name) in &choices {
                                ui.selectable_value(&mut target, Some(*id), name);
                            }
                        })
                        .response
                        .on_hover_text(
                            "The instrument this keyboard plays.\n\n“All open \
                             instruments” plays every instrument whose editor is \
                             open. Picking one plays only that instrument, and opens \
                             its editor if it is closed — an instrument sounds \
                             through its open editor.\n\nAn instrument given a MIDI \
                             port of its own under Instruments keeps listening to \
                             that port and is not affected.",
                        );
                    if target != current {
                        self.instruments.set_keyboard_target(target);
                    }
                    ui.end_row();

                    // Where the keyboard's keys land. A small controller — a
                    // Keystation Mini 32 starts at C3 — has no low notes on it
                    // at all, so a bass part has to be played up here and moved
                    // down. Applied to the incoming MIDI itself, so what the
                    // editors play and what the Sequencer records are the same
                    // note.
                    ui.label("Octave shift:");
                    let mut shift = self.octave_shift.load(Ordering::Relaxed);
                    egui::ComboBox::from_id_salt("midi_octave")
                        .width(260.0)
                        // Tall enough for all nine steps at this style's row
                        // height: the default caps a popup at 200 px and scrolls
                        // the rest, which hides the very octaves a small
                        // controller is here to reach.
                        .height(9.0 * 44.0)
                        .selected_text(octave_shift_label(shift))
                        .show_ui(ui, |ui| {
                            for step in (-MAX_OCTAVE_SHIFT..=MAX_OCTAVE_SHIFT).rev() {
                                ui.selectable_value(
                                    &mut shift,
                                    step,
                                    octave_shift_label(step),
                                );
                            }
                        })
                        .response
                        .on_hover_text(
                            "Move every note the keyboard sends by whole octaves \
                             before anything plays it.\n\nA key already held keeps \
                             the shift it was pressed with, so moving this mid-note \
                             cannot leave one sounding for good; a note that would \
                             fall off either end of MIDI is not sent.\n\nThe wheel, \
                             the pedal and everything else pass through untouched.",
                        );
                    self.octave_shift.store(shift, Ordering::Relaxed);
                    ui.end_row();
                });
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                if ui.button("Connect").clicked() {
                    self.connect_midi();
                }
                if ui.button("Refresh").clicked() {
                    self.refresh_midi_ports();
                }
            });
            ui.add_space(2.0);
            Self::status_label(ui, &self.midi_status);
        });
    }
    /// Save or load a project — see [`ProjectRequest`].
    fn perform_project_request(&mut self, request: ProjectRequest) {
        let status = match request {
            ProjectRequest::Save { dir, name } => match self.save_project(&dir, &name) {
                Ok(n) => {
                    self.sequencer.set_project_dir(dir.clone(), name);
                    format!("Saved to {} ({n} grid file(s)).", crate::file_label(&dir))
                }
                Err(e) => format!("Save failed: {e:#}"),
            },
            ProjectRequest::Load { file } => match self.load_project(&file) {
                Ok(()) => return, // `apply_project` sets its own status.
                Err(e) => format!("Load failed: {e:#}"),
            },
        };
        self.sequencer.set_status(status);
    }

    /// Write the project folder: the manifest, one `.lsft` for every LeSynth
    /// Fourier instrument a row plays, and one `.vststate` for every plugin instrument —
    /// the plugin's own state, which is where its knobs live. A LeSynth instrument
    /// gets both: the `.lsft` is its harmonic grid, the `.vststate` is every
    /// control that drew it. Returns how many sound files were written.
    ///
    /// A file is re-written when any row playing that instrument has autosave on;
    /// otherwise an existing one is left as it is, which is what pins a sound. A
    /// instrument with nothing to save (a plain synth-mode LeSynth, or a plugin that
    /// gives back no state) writes no file — the manifest records what it is
    /// instead.
    fn save_project(&mut self, dir: &std::path::Path, name: &str) -> anyhow::Result<usize> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create the folder {}", crate::file_label(dir)))?;

        let autosave: Vec<u64> = self.sequencer.autosave_instrument_ids();
        let mut sources: std::collections::HashMap<u64, InstrumentSource> = Default::default();
        let mut taken: Vec<String> = Vec::new();
        let mut taken_audio: Vec<std::path::PathBuf> = Vec::new();
        let mut written = 0usize;

        // One file per distinct instrument, however many rows share it. Read through
        // the registry rather than the Instruments panel: a subtrack published from
        // Resynthesis is a real playable instrument that never appears there, and
        // asking the panel would silently drop its sound from the project.
        for (id, instrument_name) in self.registry.list() {
            // `playback_source` snapshots the live editor when one is open, so
            // what gets written is what the user can currently hear.
            let Some(src) = self.registry.playback_source(id) else { continue };
            // A wav instrument's file is copied into the folder, so the project
            // keeps its recordings when the folder they were made in is gone.
            // The instrument itself keeps playing the file it was opened from.
            if let Some(path) = src.wav {
                let path = project::bundle_audio(dir, &path, &mut taken_audio)?;
                sources.insert(id, InstrumentSource::Wav { path });
                continue;
            }
            // The plugin's own state, whatever kind of plugin it is. For a
            // third-party VST3 it is the only thing the project can keep — its
            // binary stays where it is. For LeSynth it is everything *beside*
            // the grid: each harmonic's curve type, offset and granularity, and
            // the nested-Fourier sliders under them. Saving only the grid
            // reloads the curves with every control that drew them at zero.
            let mut state_file = None;
            if let Some(bytes) = &src.vst_state {
                let file = project::unique_file_name(&instrument_name, "vststate", &taken);
                let full = dir.join(&file);
                // Same rule as a grid: autosave off with a file already there
                // means "keep what is pinned".
                if autosave.contains(&id) || !full.exists() {
                    std::fs::write(&full, bytes)
                        .with_context(|| format!("write {}", crate::file_label(&full)))?;
                    written += 1;
                }
                taken.push(file.clone());
                state_file = Some(file);
            }
            if !src.is_lesynth {
                sources.insert(
                    id,
                    InstrumentSource::Vst {
                        path: src.plugin_path,
                        class_id: src.class_id,
                        plugin_id: src.plugin_id,
                        state: state_file,
                    },
                );
                continue;
            }
            let Some(state) = src.state else {
                // A LeSynth instrument in plain synth mode: nothing to save, and the
                // manifest says so rather than pointing at a file that is not there.
                sources.insert(id, InstrumentSource::LeSynthDefault);
                continue;
            };
            let file = project::grid_file_name(&instrument_name, &taken);
            let full = dir.join(&file);
            // Autosave off with a file already there means "keep what is pinned".
            if autosave.contains(&id) || !full.exists() {
                state
                    .write(&full)
                    .with_context(|| format!("write {}", crate::file_label(&full)))?;
                written += 1;
            }
            taken.push(file.clone());
            sources.insert(id, InstrumentSource::LeSynth { file, state: state_file });
        }

        let project = self.sequencer.to_project(name, |instrument_id| {
            instrument_id
                .and_then(|id| sources.get(&id).cloned())
                .unwrap_or(InstrumentSource::None)
        });
        project.write(&dir.join(format!("{name}.{}", project::EXTENSION)))?;
        Ok(written)
    }

    /// Read a project folder back: adopt every instrument it names into the Instruments
    /// panel, then hand the sequence to the Sequencer with one resolved
    /// registry id per row (`None` where the source could not be found).
    fn load_project(&mut self, file: &std::path::Path) -> anyhow::Result<()> {
        let mut project = Project::read(file)?;
        let dir = file
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| std::path::PathBuf::from("."));

        // A wav saved inside the folder is named relative to it. Older
        // projects name wavs, and every project names plugins, by absolute
        // path; one made on another machine finds them by name in the folder
        // itself, the recordings folder (and the one from before the rename)
        // and the standard plugin folders.
        project.resolve_audio(&dir);
        use crate::plugin::{search_paths, PluginFormat};
        let wav_dirs: Vec<_> = std::iter::once(dir.clone())
            .chain(crate::audio::recordings_dir().ok())
            .chain(crate::audio::legacy_recordings_dir())
            .collect();
        let plugin_dirs: Vec<_> = std::iter::once(dir.clone())
            .chain(
                [PluginFormat::Vst3, PluginFormat::Clap, PluginFormat::Vst2, PluginFormat::Lv2]
                    .into_iter()
                    .flat_map(search_paths),
            )
            .collect();
        for (instrument, saved, found) in project.relocate_missing(&wav_dirs, &plugin_dirs) {
            log::info!("'{instrument}': {} not found, using {}", saved.display(), found.display());
        }

        // The loaded project replaces what is open, so the instruments it brings are
        // the only ones left — otherwise the previous project's instruments would sit
        // in the list looking like part of this one. The wav tracks the *last*
        // load brought go with them; the ones Resynthesis is holding open stay,
        // because that panel is still showing the file they play.
        self.instruments.clear();
        for id in self.adopted_wavs.drain(..) {
            self.registry.remove(id);
        }

        // Distinct sources first, so rows sharing an instrument share one instance.
        let mut adopted: Vec<(InstrumentSource, Option<u64>)> = Vec::new();
        let mut resolved = Vec::with_capacity(project.rows.len());
        for row in &project.rows {
            if let Some((_, id)) = adopted.iter().find(|(src, _)| *src == row.source) {
                resolved.push(*id);
                continue;
            }
            let id = self.adopt_source(&row.source, &dir, &row.instrument_name);
            adopted.push((row.source.clone(), id));
            resolved.push(id);
        }
        self.sequencer.apply_project(&project, dir, &resolved);
        Ok(())
    }
}

/// Read a plugin's saved `.vststate` from the project folder. A state file that
/// has gone missing is not fatal: the instrument loads, and the plugin comes up on
/// its own defaults.
fn read_state_file(
    dir: &std::path::Path,
    file: Option<&str>,
    instrument_name: &str,
) -> Option<Vec<u8>> {
    let full = dir.join(file?);
    match std::fs::read(&full) {
        Ok(b) => Some(b),
        Err(e) => {
            log::warn!("'{instrument_name}': cannot read {} ({e})", full.display());
            None
        }
    }
}

impl DawApp {
    /// Bind one project source to an instrument, or `None` when it cannot be found —
    /// a deleted `.lsft`, or a VST that has moved. The row then shows what is
    /// missing and the user picks a replacement.
    fn adopt_source(
        &mut self,
        source: &InstrumentSource,
        dir: &std::path::Path,
        name: &str,
    ) -> Option<u64> {
        let name = if name.is_empty() { "Instrument" } else { name };
        match source {
            InstrumentSource::None => None,
            InstrumentSource::LeSynthDefault => self.instruments.adopt_lesynth(name, None, None).ok(),
            InstrumentSource::LeSynth { file, state } => {
                let grid = crate::track_format::TrackState::read(&dir.join(file)).ok()?;
                // The controls that drew the grid. A project saved before these
                // were kept — or one whose file has gone missing — still loads:
                // the curves come back, the controls under them read as new.
                let bytes = read_state_file(dir, state.as_deref(), name);
                self.instruments.adopt_lesynth(name, Some(grid), bytes).ok()
            }
            InstrumentSource::Wav { path } => {
                if !path.exists() {
                    log::warn!("'{name}': audio file {} is missing", path.display());
                    return None;
                }
                // One instrument per file: a file already open in Resynthesis, or
                // named by two rows, is the instrument that is already registered
                // rather than a second one indistinguishable from it.
                if let Some(id) = self.registry.find_wav(path) {
                    return Some(id);
                }
                // How long it runs, for the start slider on its notes. Read
                // from the container rather than by decoding: a project of long
                // recordings would otherwise take a decode each to open.
                let secs = crate::audio::probe_duration_secs(path).unwrap_or_else(|e| {
                    log::warn!("'{name}': cannot measure {} ({e:#})", path.display());
                    0.0
                });
                let id = self.registry.add_wav(name, path.clone(), secs);
                self.adopted_wavs.push(id);
                Some(id)
            }
            InstrumentSource::Vst { path, class_id, plugin_id, state } => {
                let bytes = read_state_file(dir, state.as_deref(), name);
                self.instruments
                    .adopt_vst(name, path.clone(), *class_id, plugin_id.clone(), bytes)
                    .ok()
            }
        }
    }
}

impl eframe::App for DawApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // A plugin editor's OpenGL can leave another context current, which
        // eframe would paint into without noticing (see the function).
        #[cfg(target_os = "macos")]
        super::editor_window::restore_main_gl_context();
        // Reactive repaint: nothing in this window changes without user input, so
        // let eframe/winit idle and wake on real events (egui still self-requests
        // repaints for its own hover/tooltip/scroll animations). The Instruments and
        // Resynthesis panels each request a low-frequency repaint while they hold
        // an open editor, so windows the user closes directly are reaped promptly.

        // Space plays and stops, as in any DAW — unless a text box is being
        // typed in, where a space is a space. Consumed before anything is drawn,
        // so a button that happens to have keyboard focus is not pressed by it
        // as well.
        if !ctx.wants_keyboard_input()
            && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space))
        {
            self.sequencer.toggle_transport(ctx);
        }
        // Ctrl+S (Cmd+S on macOS) is the Save Project button — even from a text
        // box, since saving is never what a typed character means.
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::S)) {
            self.sequencer.save_project();
        }

        // App title bar.
        egui::TopBottomPanel::top("title_bar")
            .frame(
                egui::Frame::new()
                    .fill(ctx.style().visuals.panel_fill)
                    .inner_margin(egui::Margin::symmetric(14, 10)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🌊 ReSynth Studio")
                            .heading()
                            .strong()
                            .color(egui::Color32::from_rgb(150, 200, 255)),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("additive resynthesis workstation")
                            .italics()
                            .color(egui::Color32::from_gray(150)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(concat!("v", env!("CARGO_PKG_VERSION")))
                                .color(egui::Color32::from_gray(130)),
                        );
                    });
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(6.0);
                    self.instruments.sync_keyboard_target();
                    Self::section(ui, "Instruments", |ui| {
                        self.instruments.ui(ui);
                    });
                    ui.add_space(14.0);
                    Self::section(ui, "Sequencer", |ui| {
                        self.sequencer.ui(ui);
                    });
                    // Saving reads the grids out of the live plugin instances and
                    // loading puts instruments into the Instruments panel, so the request is
                    // performed here rather than inside the Sequencer, which owns
                    // neither. Deferred past the draw so the panel is not mutated
                    // mid-frame.
                    if let Some(request) = self.sequencer.take_request() {
                        self.perform_project_request(request);
                    }
                    ui.add_space(14.0);
                    self.midi_section(ui);
                    ui.add_space(14.0);
                    Self::section(ui, "Resynthesis", |ui| {
                        self.resynth.ui(ui);
                    });
                    ui.add_space(10.0);
                });
        });
    }
}

/// How an octave shift reads in its select box: signed, and saying what it does
/// rather than only how far.
fn octave_shift_label(shift: i32) -> String {
    match shift {
        0 => "none (as played)".to_string(),
        1 => "+1 octave up".to_string(),
        -1 => "−1 octave down".to_string(),
        n if n > 0 => format!("+{n} octaves up"),
        n => format!("−{} octaves down", -n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn d5() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("D5.wav")
    }

    /// A wav instrument survives a project the way every other instrument does. Loading
    /// a project that names the file by absolute path must bind the row to a
    /// instrument that plays it; saving must copy the file into the folder and name
    /// it relative, so the project still plays once the original is gone.
    #[test]
    fn a_wav_row_is_copied_into_the_folder_and_loads_back_onto_the_copy() {
        let dir = std::env::temp_dir().join("resynth-wav-project");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let manifest = dir.join(format!("Take.{}", project::EXTENSION));
        std::fs::write(
            &manifest,
            format!(
                "resynth-project 1\nname = Take\ntempo = 120\n\n\
                 [row]\ntrack = take.wav\nsource = wav {}\ngain = 1\n\
                 lead = 0 none\nnote = 60 0 1/4 0 1/8\n",
                d5().display()
            ),
        )
        .expect("write the manifest");

        let mut app = DawApp::default();
        app.load_project(&manifest).expect("loads");

        // One instrument, playing the file, and the row is on it — not left showing a
        // missing source.
        let instruments = app.registry.list();
        assert_eq!(instruments.len(), 1, "{instruments:?}");
        let id = app.registry.find_wav(&d5()).expect("the file is an instrument");
        assert!(app.registry.is_wav(id));
        // How long the file runs came back with it: that is the range of the
        // start slider on every note of the row, and a zero would put the whole
        // recording out of reach.
        let secs = app.registry.wav_secs(id).expect("a wav instrument knows its length");
        assert!(
            (secs - 5.81).abs() < 0.05,
            "D5.wav is 5.8 s; the project loaded it as {secs}"
        );

        // Saving copies the recording in and names the copy — through the app,
        // which is what decides what an instrument's source is.
        app.save_project(&dir, "Take").expect("saves");
        let copy = dir.join(project::AUDIO_DIR).join("D5.wav");
        assert_eq!(std::fs::read(&copy).expect("copied"), std::fs::read(d5()).unwrap());
        let text = std::fs::read_to_string(&manifest).expect("re-read");
        assert!(
            text.contains(&format!("source = wav {}/D5.wav", project::AUDIO_DIR)),
            "the manifest does not name the copy: {text}"
        );

        // Loading again plays the copy, and must not leave the first load's
        // instrument behind.
        app.load_project(&manifest).expect("loads again");
        assert_eq!(app.registry.list().len(), 1, "{:?}", app.registry.list());
        assert!(app.registry.find_wav(&copy).is_some(), "the row does not play the copy");

        // A save of the reloaded project names the copy where it is, rather than
        // copying it into itself again.
        app.save_project(&dir, "Take").expect("saves again");
        let files: Vec<_> = std::fs::read_dir(dir.join(project::AUDIO_DIR)).unwrap().collect();
        assert_eq!(files.len(), 1, "{files:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A project naming a file that is not there must say so rather than bind
    /// the row to nothing quietly: the row keeps what it was looking for, which
    /// is what puts "⚠ missing" in its select box.
    #[test]
    fn a_missing_audio_file_leaves_the_row_asking_for_it() {
        let dir = std::env::temp_dir().join("resynth-wav-project-missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let manifest = dir.join(format!("Gone.{}", project::EXTENSION));
        std::fs::write(
            &manifest,
            "resynth-project 1\nname = Gone\n\n\
             [row]\ntrack = gone.wav\nsource = wav /nonexistent/gone.wav\n\
             note = 60 0 1/4 0 none\n",
        )
        .expect("write the manifest");

        let mut app = DawApp::default();
        app.load_project(&manifest).expect("loads");
        assert!(app.registry.list().is_empty(), "nothing to play, so no instrument");
        let saved = app.sequencer.to_project("Gone", |_| InstrumentSource::None);
        assert_eq!(
            saved.rows[0].source,
            InstrumentSource::Wav { path: PathBuf::from("/nonexistent/gone.wav") },
            "a save after a failed load must not lose the file it was looking for"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A line of labels, buttons and select boxes reads as one line. egui
    /// centres each thing on the line as tall as it is *so far*, and a line
    /// starts at `interact_size.y`; under egui's 18 px default a padded button
    /// grows it, so every box after the first sits a little lower than the one
    /// before — the repeat window's six boxes, and the input picker beside the
    /// recording button, both read as a staircase. Starting the line at a
    /// button's full height leaves nothing to grow.
    #[test]
    fn labels_buttons_and_select_boxes_share_one_line() {
        let ctx = egui::Context::default();
        DawApp::configure_style(&ctx);
        let mut centres = Vec::new();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal(|ui| {
                    centres.push(("label", ui.label("from").rect.center().y));
                    centres.push(("button", ui.button("Record").rect.center().y));
                    for salt in ["first box", "second box"] {
                        let r = egui::ComboBox::from_id_salt(salt)
                            .selected_text(salt)
                            .show_ui(ui, |_| {})
                            .response;
                        centres.push((salt, r.rect.center().y));
                    }
                    centres.push(("last label", ui.label("to").rect.center().y));
                });
            });
        });
        let line = centres[0].1;
        for (what, y) in &centres {
            assert!((y - line).abs() < 0.5, "the {what} is off the line: {centres:?}");
        }
    }
}
