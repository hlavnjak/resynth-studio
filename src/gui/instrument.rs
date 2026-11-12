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
//! Plugin instruments — a hosted plugin with an open/close editor, but no
//! audio-file analysis. Two flavours are offered:
//!
//!   * **LeSynth Fourier** — the embedded internal plugin, opened in its plain
//!     (non-analysis, empty) synth mode. No `push_analysis`, so no bucket grid.
//!   * **Custom plugin** — any third-party VST3, CLAP, VST2 or LV2 (or, on
//!     macOS, Audio Unit), picked in [`PluginBrowser`] from the plugins
//!     installed on this machine or browsed for by hand.
//!
//! An instrument is lightweight metadata (name + plugin path); the heavy plugin
//! instance, its audio stream and its editor window live in [`EditorInstance`]
//! and exist only while the editor is open — closing the window tears them all
//! down, matching the resynth subtrack editors.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use eframe::egui;

use super::editor_window::{open_editor_in_thread, EditorHandle};
use super::registry::InstrumentRegistry;
use crate::audio::AudioEngine;
use crate::midi::{KeyboardTarget, MidiEventQueue, MidiFeed, MidiRouter};
use crate::track_format::TrackState;
use crate::midi::plays_a_drum_kit;
use crate::plugin::{scan_installed, FoundPlugin, PluginFormat, PluginInstance};
use crate::vst::{class_ids, next_instance_token};

/// A live plugin editor: the loaded instance, its editor-window thread and an
/// audio stream driving `process()` so the plugin's in-GUI piano is audible.
///
/// Dropping this asks the window thread to close and joins it (so the plugin
/// view is detached before the library unloads), then stops the audio stream.
/// Conversely, when the user closes the window directly the thread sets
/// `closed`; the owner polls [`EditorInstance::is_closed`] each frame and drops
/// this to reclaim the resources.
pub struct EditorInstance {
    // Teardown order is everything here, and `Drop` below does it explicitly
    // rather than leaning on the order these fields happen to be declared in:
    // the audio stream and the editor window both hold pointers into the plugin,
    // so both must be gone before `_plugin` terminates it and unloads its
    // library.
    handle: Option<JoinHandle<()>>,
    engine: Option<AudioEngine>,
    _plugin: Arc<PluginInstance>,
    close_flag: Arc<AtomicBool>,
    closed: Arc<AtomicBool>,
}

impl EditorInstance {
    /// Open an editor window for `plugin` and start an audio stream (fed by
    /// `midi_queue`) so it is audible. The plugin must already be initialised.
    /// Fails only if the editor window itself cannot be created; an unavailable
    /// audio device merely leaves the instance silent.
    pub fn open(plugin: Arc<PluginInstance>, midi_queue: MidiEventQueue) -> Result<Self> {
        let EditorHandle {
            handle,
            close_flag,
            closed,
        } = open_editor_in_thread(&plugin)?;

        let engine = match AudioEngine::start(plugin.clone(), midi_queue) {
            Ok(e) => Some(e),
            Err(e) => {
                log::warn!("Instrument audio start failed: {e:#}");
                None
            }
        };

        Ok(EditorInstance {
            handle: Some(handle),
            engine,
            _plugin: plugin,
            close_flag,
            closed,
        })
    }

    /// True once the editor window has gone away — whether the user closed it or
    /// we asked it to via `close_flag`.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    /// Whether an audio stream is driving the instance (false if no device).
    pub fn is_audible(&self) -> bool {
        self.engine.is_some()
    }

    /// Snapshot the live LeSynth grid into a [`TrackState`] for saving. Errors if
    /// the instance wasn't tagged (non-LeSynth) or has no grid.
    pub fn export_state(&self) -> Result<TrackState> {
        self._plugin.export_state()
    }

    /// The instance behind this editor, for the instrument registry to hold weakly —
    /// that is how the Sequencer plays the grid as it is being edited rather than
    /// the one the instrument was registered with.
    pub fn plugin(&self) -> &Arc<PluginInstance> {
        &self._plugin
    }
}

impl Drop for EditorInstance {
    fn drop(&mut self) {
        // 1) Stop the audio stream. Dropping it stops and joins the device
        //    callback, and the callback calls `process()` on this very plugin —
        //    so anything that tears the plugin down while the stream is alive is
        //    a crash on the audio thread, not a leak.
        self.engine = None;

        // 2) Ask the window thread to exit, then wait for it: it detaches the
        //    plugin view (`view.removed()`) as it unwinds, which must happen
        //    before the `_plugin` Arc unloads the library the view points into.
        //    If the user already closed the window, the thread has finished and
        //    this joins instantly.
        super::editor_window::request_close(&self.close_flag);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }

        // 3) `_plugin` now drops with nothing else pointing into it, terminating
        //    the plugin and unloading its library.
    }
}

/// Instrument flavour. Only LeSynth instruments carry the harmonic grid that can be
/// exported/imported; custom plugin instruments are create-only.
#[derive(Clone, Copy, PartialEq)]
enum InstrumentKind {
    LeSynth,
    CustomVst,
}

/// One plugin instrument: persistent metadata plus its editor while open.
struct PluginInstrument {
    /// Stable id, used to key per-instrument egui widget state across frames.
    id: u64,
    /// This instrument's id in the shared [`InstrumentRegistry`], where the Sequencer finds
    /// it. Distinct from `id`, which is local to this panel.
    registry_id: u64,
    /// Display name (the plugin kind, or the chosen `.so` file name).
    name: String,
    kind: InstrumentKind,
    /// Library to (re)load whenever the editor is opened.
    plugin_path: PathBuf,
    /// Class ID to select from the factory; `None` takes the first class.
    class_id: Option<[i8; 16]>,
    /// Which plugin inside the file, for a CLAP or an LV2 bundle holding
    /// several (or an Audio Unit's codes); `None` takes the first.
    plugin_id: Option<String>,
    /// A saved grid to push into the instance when its editor is opened (set for
    /// instruments created via "Load LeSynth Fourier Instrument").
    import_state: Option<TrackState>,
    /// The same for a custom plugin: its own saved state, which is where it
    /// keeps the knobs the user set. Captured when the editor closes and restored
    /// when it opens, so the sound survives the window and reaches the Sequencer.
    vst_state: Option<Vec<u8>>,
    /// Which keyboard plays this instrument, by port name. `None` follows whatever
    /// the MIDI panel is connected to, which is what every instrument does until it
    /// is pointed somewhere else.
    midi_source: Option<String>,
    /// This instrument's own feed from the router, made when its editor first opens
    /// and kept afterwards so changing the source does not mean restarting the
    /// audio engine draining it.
    feed: Option<MidiFeed>,
    editor: Option<EditorInstance>,
}

impl PluginInstrument {
    /// Load the plugin, initialise it at the output device's format and open its
    /// editor. Idempotent: does nothing if the editor is already open. LeSynth
    /// instruments are tagged with a token (so their grid can be exported), and if the
    /// instrument carries an `import_state` it is pushed in before the window opens —
    /// otherwise LeSynth stays in its plain synth mode (no `push_analysis`).
    fn open_editor(&mut self, router: &MidiRouter) -> Result<()> {
        if self.editor.is_some() {
            return Ok(());
        }
        // Only LeSynth instances support the state export/import C ABI.
        let token = (self.kind == InstrumentKind::LeSynth).then(next_instance_token);
        let inst = Arc::new(PluginInstance::load(
            &self.plugin_path,
            self.class_id.as_ref(),
            self.plugin_id.as_deref(),
            token,
        )?);

        // A custom plugin's own state goes in before activation, as the VST3
        // spec has it — and before the window opens, so the editor draws the
        // knobs the user left rather than the plugin's defaults.
        if let Some(bytes) = &self.vst_state {
            if let Err(e) = inst.restore_state(bytes) {
                log::warn!("Instrument state restore failed: {e:#}");
            }
        }

        let (sr, block) = AudioEngine::query_device_config()
            .map(|c| (c.sample_rate, c.max_buffer_size as i32))
            .unwrap_or((44_100.0, 512));
        let _ = inst.initialize_audio(sr, block);

        // Push a loaded grid (Analysis mode) before the window renders it.
        if let Some(state) = &self.import_state {
            if let Err(e) = inst.import_state(state) {
                log::warn!("Instrument import failed: {e:#}");
            }
        }

        // One feed per instrument, made once and kept: the engine drains the queue it
        // was started with, so an instrument that changes keyboard changes what the
        // router puts *into* that queue rather than which queue it is.
        let feed = match &self.feed {
            Some(feed) => feed.queue.clone(),
            None => {
                let feed = router.subscribe(self.midi_source.clone())?;
                let queue = feed.queue.clone();
                self.feed = Some(feed);
                queue
            }
        };
        self.editor = Some(EditorInstance::open(inst, feed)?);
        Ok(())
    }

    /// Whether this instrument's live grid can be exported (LeSynth, editor open).
    fn can_export(&self) -> bool {
        self.kind == InstrumentKind::LeSynth && self.editor.is_some()
    }

    /// Take what the open editor is playing and keep it on the instrument, so closing
    /// the window does not throw the user's edits away.
    ///
    /// Without this the Sequencer would fall back to the state the instrument was
    /// registered with the moment an editor closed — the plugin's defaults, for
    /// a custom plugin whose knobs had just been set by hand.
    fn capture_editor_state(&mut self, registry: &InstrumentRegistry) {
        let Some(editor) = &self.editor else { return };
        if let InstrumentKind::LeSynth = self.kind {
            // An instance in plain synth mode has no grid to export; that is
            // not a failure, it is an instrument with nothing to remember.
            if let Ok(state) = editor.export_state() {
                registry.set_state(self.registry_id, Some(state.clone()));
                self.import_state = Some(state);
            }
        }
        // Every plugin, LeSynth included, also keeps its own state — and for
        // LeSynth that is where the whole Synth editor lives: the curve type,
        // offset and granularity of each harmonic, and the nested-Fourier
        // sliders under them. The grid says what the curves *are*; this says
        // what drew them, and without it reopening the instrument shows a correct
        // picture over controls that all read zero.
        match editor.plugin().save_state() {
            Ok(bytes) if !bytes.is_empty() => {
                registry.set_vst_state(self.registry_id, Some(bytes.clone()));
                self.vst_state = Some(bytes);
            }
            Ok(_) => {}
            Err(e) => log::warn!("'{}' state capture failed: {e:#}", self.name),
        }
    }

    /// Drop the editor if its window was closed directly, freeing the instance —
    /// keeping what it was playing first.
    fn reap_editor(&mut self, registry: &InstrumentRegistry) {
        if self.editor.as_ref().is_some_and(EditorInstance::is_closed) {
            self.capture_editor_state(registry);
            self.editor = None;
        }
    }
}

/// The "add a custom plugin" picker.
///
/// A VST3 is a *bundle* — a `Foo.vst3` directory with the real library buried at
/// `Contents/x86_64-linux/Foo.so` — and so is an LV2; a file dialog cannot
/// select a directory, so "pick the plugin file" alone is a dead end for most
/// plugins a user actually has installed. This lists what is installed, in
/// every format, and keeps both browse buttons for anything outside the
/// standard locations.
pub struct PluginBrowser {
    pub found: Vec<FoundPlugin>,
    /// Where the scan looked, shown when it found nothing.
    pub searched: Vec<PathBuf>,
    /// Only plugins of this format are listed; `None` lists every format.
    filter: Option<PluginFormat>,
    error: Option<String>,
}

impl PluginBrowser {
    pub fn scan() -> Self {
        let (found, searched) = scan_installed();
        PluginBrowser {
            found,
            searched,
            filter: None,
            error: None,
        }
    }
}

/// Whether this plugin plays a drum kit, so the Sequencer can name its notes.
///
/// Asks the plugin first — its VST3 subcategories are the precise answer — and
/// falls back to the name, which is what catches the many that declare
/// themselves a plain instrument. The scan opens the module and closes it again;
/// if that fails the instrument is still perfectly usable, so the name alone decides.
fn is_a_drum_kit(path: &std::path::Path, plugin_id: Option<&str>, name: &str) -> bool {
    match crate::plugin::describe(path, plugin_id) {
        Some((own_name, categories)) => {
            plays_a_drum_kit(&own_name, &categories) || plays_a_drum_kit(name, "")
        }
        None => plays_a_drum_kit(name, ""),
    }
}

/// The directories a VST3 is installed into, most specific first.
pub fn vst3_search_paths() -> Vec<PathBuf> {
    crate::plugin::search_paths(PluginFormat::Vst3)
}

/// The Instruments panel: the "add" buttons and the list of plugin instruments.
pub struct InstrumentsPanel {
    instruments: Vec<PluginInstrument>,
    next_id: u64,
    status: String,
    /// The custom-VST picker, while it is open.
    browser: Option<PluginBrowser>,
    /// Where an instrument's keyboard feed comes from — one queue per instance, so two
    /// open editors both hear everything instead of splitting the stream.
    midi_router: MidiRouter,
    /// The one instrument the MIDI panel's keyboard plays, by instrument id, or
    /// `None` for every instrument following the panel's port.
    keyboard_target: Option<u64>,
    /// The app-wide instrument list the Sequencer draws its rows from. Every instrument
    /// created here is registered, and removed from it when it goes.
    registry: InstrumentRegistry,
}

impl InstrumentsPanel {
    pub fn new(midi_router: MidiRouter, registry: InstrumentRegistry) -> Self {
        Self {
            instruments: Vec::new(),
            next_id: 0,
            status: "Add a LeSynth Fourier or custom plugin instrument.".to_string(),
            browser: None,
            midi_router,
            keyboard_target: None,
            registry,
        }
    }

    /// The instruments the MIDI panel's keyboard can be given to: id and name.
    pub fn keyboard_choices(&self) -> Vec<(u64, String)> {
        self.instruments.iter().map(|i| (i.id, i.name.clone())).collect()
    }

    /// The instrument the MIDI panel's keyboard plays; `None` is all of them.
    pub fn keyboard_target(&self) -> Option<u64> {
        self.keyboard_target
    }

    /// Give the MIDI panel's keyboard to one instrument, or back to all of them.
    ///
    /// An instrument only sounds through its open editor, so picking one opens
    /// it: a keyboard pointed at a closed instrument would play nothing.
    pub fn set_keyboard_target(&mut self, target: Option<u64>) {
        self.keyboard_target = target;
        let router = self.midi_router.clone();
        if let Some(instrument) =
            self.instruments.iter_mut().find(|i| Some(i.id) == target)
        {
            self.status = match instrument.open_editor(&router) {
                Ok(()) => format!("The MIDI keyboard now plays {}.", instrument.name),
                Err(e) => format!("Open editor failed: {e:#}"),
            };
        } else if target.is_none() {
            self.status = "The MIDI keyboard plays every open instrument.".to_string();
        }
        self.sync_keyboard_target();
    }

    /// Tell the router who the choice is. Every frame: the chosen instrument's
    /// feed appears when its editor first opens, and the instrument itself may
    /// be removed — the keyboard then goes back to everything rather than
    /// staying on something that no longer exists.
    pub fn sync_keyboard_target(&mut self) {
        let target = match self.keyboard_target {
            None => KeyboardTarget::Everyone,
            Some(id) => match self.instruments.iter().find(|i| i.id == id) {
                Some(instrument) => match &instrument.feed {
                    Some(feed) => KeyboardTarget::Feed(feed.id),
                    None => KeyboardTarget::Nobody,
                },
                None => {
                    self.keyboard_target = None;
                    KeyboardTarget::Everyone
                }
            },
        };
        self.midi_router.set_default_target(target);
    }

    fn internal_plugin_path() -> Option<PathBuf> {
        crate::internal_plugin_path()
    }

    /// Add a LeSynth Fourier instrument (internal plugin, plain synth mode).
    fn add_lesynth_instrument(&mut self) {
        let Some(path) = Self::internal_plugin_path() else {
            self.status = "Could not locate the internal plugin.".to_string();
            return;
        };
        if !path.exists() {
            self.status = format!("Internal plugin {} not found.", crate::file_label(&path));
            return;
        }
        let id = self.take_id();
        let name = unique_instrument_name(&format!("LeSynth Fourier {}", id + 1));
        let registry_id =
            self.registry
                .add(&name, path.clone(), Some(class_ids::FOURIER_SYNTH), true, None);
        self.instruments.push(PluginInstrument {
            id,
            registry_id,
            name,
            kind: InstrumentKind::LeSynth,
            plugin_path: path,
            class_id: Some(class_ids::FOURIER_SYNTH),
            plugin_id: None,
            import_state: None,
            vst_state: None,
            midi_source: None,
            feed: None,
            editor: None,
        });
        self.status = "Created LeSynth Fourier instrument.".to_string();
    }

    /// Add a custom plugin instrument for `path` — a bundle or a bare library, of
    /// any format — and `plugin_id` inside it, named `name` if the picker knew
    /// what the plugin is called.
    ///
    /// The plugin is checked here rather than when its editor is opened: a file
    /// that is not a plugin, or one whose dependencies the loader cannot
    /// satisfy, should say so while the user is still looking at the picker.
    fn add_custom_plugin_instrument(
        &mut self,
        path: PathBuf,
        plugin_id: Option<String>,
        name: Option<String>,
    ) -> Result<()> {
        let format = crate::plugin::validate(&path, plugin_id.as_deref())?;
        // Name the instrument after the bundle, not the library inside it: "Dexed"
        // rather than "Dexed.so", and never "libsomething.so".
        let name = name.unwrap_or_else(|| crate::plugin::display_stem(&path));
        // With no class id the first audio-module class in a VST3 factory is
        // taken — a bundle may hold several, and only the plugin knows its own.
        let registry_id = self.registry.add(&name, path.clone(), None, false, None);
        self.registry.set_plugin_id(registry_id, plugin_id.clone());
        self.registry.set_percussion(
            registry_id,
            is_a_drum_kit(&path, plugin_id.as_deref(), &name),
        );
        let id = self.take_id();
        self.instruments.push(PluginInstrument {
            id,
            registry_id,
            name: name.clone(),
            kind: InstrumentKind::CustomVst,
            plugin_path: path,
            class_id: None,
            plugin_id,
            import_state: None,
            vst_state: None,
            midi_source: None,
            feed: None,
            editor: None,
        });
        self.status = format!("Created {} instrument '{name}'.", format.label());
        Ok(())
    }

    /// The picker window: installed plugins, plus the two browse buttons for
    /// anything that lives elsewhere.
    fn browser_ui(&mut self, ctx: &egui::Context) {
        let Some(browser) = &mut self.browser else {
            return;
        };
        let mut open = true;
        // `(path, plugin id, name)` of the plugin picked.
        let mut chosen: Option<(PathBuf, Option<String>, Option<String>)> = None;
        let mut close = false;

        egui::Window::new("Add a custom plugin")
            .open(&mut open)
            .collapsible(false)
            .default_width(460.0)
            .show(ctx, |ui| {
                // Which formats to list. Only the ones this scan found anything
                // in are offered, so the row says what is installed at a glance.
                ui.horizontal_wrapped(|ui| {
                    ui.label("Show:");
                    ui.selectable_value(&mut browser.filter, None, "All");
                    for format in PluginFormat::ALL {
                        let count = browser.found.iter().filter(|p| p.format == format).count();
                        if count > 0 {
                            ui.selectable_value(
                                &mut browser.filter,
                                Some(format),
                                format!("{} ({count})", format.label()),
                            );
                        }
                    }
                });
                ui.add_space(4.0);

                if browser.found.is_empty() {
                    ui.label("No plugins found in:");
                    for dir in &browser.searched {
                        ui.label(
                            egui::RichText::new(format!("  {}", crate::file_label(dir)))
                                .color(egui::Color32::from_gray(160)),
                        )
                        .on_hover_text(dir.display().to_string());
                    }
                    if browser.searched.is_empty() {
                        ui.label(
                            egui::RichText::new("  (none of the standard plugin directories exist)")
                                .color(egui::Color32::from_gray(160)),
                        );
                    }
                    ui.add_space(4.0);
                    ui.label("Use the browse buttons below.");
                } else {
                    egui::ScrollArea::vertical()
                        .max_height(320.0)
                        .show(ui, |ui| {
                            let shown = browser
                                .found
                                .iter()
                                .filter(|p| browser.filter.is_none_or(|f| f == p.format));
                            for plugin in shown {
                                ui.horizontal(|ui| {
                                    if ui.button("Add").clicked() {
                                        chosen = Some((
                                            plugin.path.clone(),
                                            plugin.plugin_id.clone(),
                                            Some(plugin.name.clone()),
                                        ));
                                    }
                                    ui.label(
                                        egui::RichText::new(format!("{:<4}", plugin.format.label()))
                                            .monospace()
                                            .color(egui::Color32::from_gray(150)),
                                    );
                                    let hover = match &plugin.plugin_id {
                                        Some(id) => format!("{}\n{id}", crate::file_label(&plugin.path)),
                                        None => crate::file_label(&plugin.path),
                                    };
                                    ui.label(egui::RichText::new(&plugin.name).strong())
                                        .on_hover_text(hover);
                                });
                            }
                        });
                }

                ui.add_space(6.0);
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("📁 Browse bundle…")
                        .on_hover_text("Pick a .vst3 or .lv2 bundle directory")
                        .clicked()
                    {
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            chosen = Some((dir, None, None));
                        }
                    }
                    if ui
                        .button("📄 Browse library…")
                        .on_hover_text(
                            "Pick a plugin file directly: a .clap, a VST2 library, \
                             or a VST3 library",
                        )
                        .clicked()
                    {
                        if let Some(file) = rfd::FileDialog::new()
                            .add_filter("Plugin", &["clap", "vst3", "so", "dll", "dylib", "vst"])
                            .add_filter("All files", &["*"])
                            .pick_file()
                        {
                            chosen = Some((file, None, None));
                        }
                    }
                    if ui.button("↻ Rescan").clicked() {
                        let refreshed = PluginBrowser::scan();
                        browser.found = refreshed.found;
                        browser.searched = refreshed.searched;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });

                if let Some(err) = &browser.error {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(err).color(egui::Color32::from_rgb(240, 140, 140)),
                    );
                }
            });

        if let Some((path, plugin_id, name)) = chosen {
            match self.add_custom_plugin_instrument(path, plugin_id, name) {
                Ok(()) => close = true,
                // Stay open with the reason on screen — the user is most likely
                // to want to pick something else straight away.
                Err(e) => {
                    if let Some(browser) = &mut self.browser {
                        browser.error = Some(format!("{e:#}"));
                    }
                }
            }
        }
        if close || !open {
            self.browser = None;
        }
    }

    /// Adopt a LeSynth Fourier instrument from an already-parsed grid — what loading a
    /// project does for every row that plays one. Returns its registry id.
    ///
    /// `state` is `None` for a plain synth-mode instrument (no grid to import).
    /// `vst_state` is the plugin's own saved state, which is where everything
    /// *beside* the grid lives — the curve type, offset and granularity of every
    /// harmonic, and the nested-Fourier sliders under them. The grid alone
    /// reloads the picture and none of the controls that drew it.
    pub fn adopt_lesynth(
        &mut self,
        name: &str,
        state: Option<TrackState>,
        vst_state: Option<Vec<u8>>,
    ) -> Result<u64> {
        let path = Self::internal_plugin_path()
            .filter(|p| p.exists())
            .context("internal LeSynth Fourier plugin not found")?;
        let id = self.take_id();
        let registry_id = self.registry.add(
            name,
            path.clone(),
            Some(class_ids::FOURIER_SYNTH),
            true,
            state.clone(),
        );
        self.registry.set_vst_state(registry_id, vst_state.clone());
        self.instruments.push(PluginInstrument {
            id,
            registry_id,
            name: name.to_string(),
            kind: InstrumentKind::LeSynth,
            plugin_path: path,
            class_id: Some(class_ids::FOURIER_SYNTH),
            plugin_id: None,
            import_state: state,
            vst_state,
            midi_source: None,
            feed: None,
            editor: None,
        });
        Ok(registry_id)
    }

    /// Adopt a custom plugin instrument by path — the other half of loading a
    /// project. `vst_state` is the plugin's own saved state from the project,
    /// restored into every instance the instrument loads.
    pub fn adopt_vst(
        &mut self,
        name: &str,
        path: PathBuf,
        class_id: Option<[i8; 16]>,
        plugin_id: Option<String>,
        vst_state: Option<Vec<u8>>,
    ) -> Result<u64> {
        if crate::plugin::has_file(&path, plugin_id.as_deref()) && !path.exists() {
            anyhow::bail!("plugin {} not found", crate::file_label(&path));
        }
        let id = self.take_id();
        let registry_id = self.registry.add(name, path.clone(), class_id, false, None);
        self.registry.set_plugin_id(registry_id, plugin_id.clone());
        self.registry.set_vst_state(registry_id, vst_state.clone());
        self.registry.set_percussion(
            registry_id,
            is_a_drum_kit(&path, plugin_id.as_deref(), name),
        );
        self.instruments.push(PluginInstrument {
            id,
            registry_id,
            name: name.to_string(),
            kind: InstrumentKind::CustomVst,
            plugin_path: path,
            class_id,
            plugin_id,
            import_state: None,
            vst_state,
            midi_source: None,
            feed: None,
            editor: None,
        });
        Ok(registry_id)
    }

    /// Drop every instrument and its editor — what loading a project does before it
    /// adopts the project's own.
    pub fn clear(&mut self) {
        for instrument in self.instruments.drain(..) {
            self.registry.remove(instrument.registry_id);
        }
        self.status = "Cleared for a loaded project.".to_string();
    }

    /// Load a saved `.lsft` LeSynth instrument: pick the file, parse it, and add a
    /// LeSynth instrument whose grid is pushed into the instance when its editor opens.
    fn load_lesynth_instrument(&mut self) {
        let Some(file) = rfd::FileDialog::new()
            .add_filter("LeSynth Fourier instrument (.lsft)", &["lsft"])
            .add_filter("All files", &["*"])
            .pick_file()
        else {
            return;
        };
        let state = match TrackState::read(&file) {
            Ok(s) => s,
            Err(e) => {
                self.status = format!("Load failed: {e:#}");
                return;
            }
        };
        let Some(plugin_path) = Self::internal_plugin_path() else {
            self.status = "Could not locate the internal plugin.".to_string();
            return;
        };
        if !plugin_path.exists() {
            self.status = format!("Internal plugin {} not found.", crate::file_label(&plugin_path));
            return;
        }
        let name = unique_instrument_name(
            &file
                .file_stem()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "LeSynth instrument".to_string()),
        );
        let id = self.take_id();
        // The saved grid is registered with the instrument, so the Sequencer can play
        // it without the editor ever being opened.
        let registry_id = self.registry.add(
            &name,
            plugin_path.clone(),
            Some(class_ids::FOURIER_SYNTH),
            true,
            Some(state.clone()),
        );
        self.instruments.push(PluginInstrument {
            id,
            registry_id,
            name,
            kind: InstrumentKind::LeSynth,
            plugin_path,
            class_id: Some(class_ids::FOURIER_SYNTH),
            plugin_id: None,
            import_state: Some(state),
            vst_state: None,
            midi_source: None,
            feed: None,
            editor: None,
        });
        self.status = "Loaded LeSynth instrument — open its editor to view.".to_string();
    }

    /// Save an instrument's live grid to a chosen `.lsft` file.
    fn export_instrument(&mut self, idx: usize) {
        // Snapshot first (ends the borrow of `self.instruments` before the dialog).
        let snapshot = self
            .instruments
            .get(idx)
            .filter(|t| t.can_export())
            .map(|t| (t.name.clone(), t.editor.as_ref().unwrap().export_state()));
        let Some((name, result)) = snapshot else {
            return;
        };
        let state = match result {
            Ok(s) => s,
            Err(e) => {
                self.status = format!("Export failed: {e:#}");
                return;
            }
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("LeSynth Fourier instrument (.lsft)", &["lsft"])
            .set_file_name(format!("{name}.lsft"))
            .save_file()
        else {
            return;
        };
        self.status = match state.write(&path) {
            Ok(()) => format!("Exported {name}."),
            Err(e) => format!("Export failed: {e:#}"),
        };
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("➕ Create LeSynth Fourier Instrument").clicked() {
                self.add_lesynth_instrument();
            }
            if ui
                .button("➕ Create Custom Plugin Instrument")
                .on_hover_text(if cfg!(target_os = "macos") {
                    "Pick an installed VST3, CLAP, VST2, LV2 or Audio Unit, or browse for one"
                } else {
                    "Pick an installed VST3, CLAP, VST2 or LV2, or browse for one"
                })
                .clicked()
            {
                self.browser = Some(PluginBrowser::scan());
            }
            if ui
                .button("➕ Load LeSynth Fourier Instrument")
                .on_hover_text("Open a saved .lsft instrument")
                .clicked()
            {
                self.load_lesynth_instrument();
            }
        });
        // Before the early return below: the picker is a window of its own and
        // has to be drawn whether or not there are any instruments yet.
        self.browser_ui(ui.ctx());

        ui.add_space(4.0);
        ui.label(egui::RichText::new(&self.status).color(egui::Color32::from_gray(170)));

        if self.instruments.is_empty() {
            return;
        }
        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        // Reap editors the user closed directly, so button state and resources
        // stay honest, and keep the registry's view of which instances are live
        // in step with them — that is what lets the Sequencer play a grid the
        // user is editing right now.
        for instrument in &mut self.instruments {
            instrument.reap_editor(&self.registry);
            self.registry.set_live(
                instrument.registry_id,
                instrument.editor.as_ref().map(|e| Arc::downgrade(e.plugin())),
            );
        }

        // Deferred actions, so we don't mutate an instrument while iterating.
        enum Action {
            Open(usize),
            Close(usize),
            Export(usize),
            Remove(usize),
        }
        let mut action: Option<Action> = None;

        // The ports on offer, read once for the whole list rather than per row:
        // asking the driver is not free, and every row offers the same answer.
        let ports = crate::midi::list_midi_ports().unwrap_or_default();
        let default_port = self.midi_router.default_port();
        // Mutable: a row's keyboard picker writes straight into its instrument.
        let mut retarget: Option<(usize, Option<String>)> = None;
        for (idx, instrument) in self.instruments.iter_mut().enumerate() {
            // Scope widget ids by the stable instrument id, so buttons keep their
            // identity when instruments above them are removed and indices shift.
            ui.push_id(instrument.id, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_gray(34))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(&instrument.name).strong());
                            if instrument.editor.is_some() {
                                ui.label(
                                    egui::RichText::new("● editor open")
                                        .color(egui::Color32::from_rgb(130, 230, 150)),
                                );
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .button("🗑 Remove")
                                        .on_hover_text("Remove this instrument (closes its editor)")
                                        .clicked()
                                    {
                                        action = Some(Action::Remove(idx));
                                    }
                                    if instrument.editor.is_some() {
                                        if ui.button("✖ Close editor").clicked() {
                                            action = Some(Action::Close(idx));
                                        }
                                    } else if ui.button("Open editor").clicked() {
                                        action = Some(Action::Open(idx));
                                    }
                                    // Export the live grid (LeSynth only, editor open).
                                    if instrument.can_export()
                                        && ui
                                            .button("💾 Export…")
                                            .on_hover_text("Save this instrument's grid to a .lsft file")
                                            .clicked()
                                    {
                                        action = Some(Action::Export(idx));
                                    }
                                },
                            );
                        });

                        // Which keyboard plays *this* instance. Each instrument has a
                        // feed of its own, so two open editors both hear
                        // everything their keyboard sends instead of taking
                        // turns at one queue — and a machine with two keyboards
                        // on it can play them into two instruments at once.
                        ui.horizontal(|ui| {
                            ui.label("MIDI in:");
                            let mut want = instrument.midi_source.clone();
                            let label = match &want {
                                Some(port) => port.clone(),
                                None => match &default_port {
                                    Some(p) => format!("default ({p})"),
                                    None => "default (not connected)".to_string(),
                                },
                            };
                            egui::ComboBox::from_id_salt(("midi_in", instrument.id, ports.len()))
                                .width(300.0)
                                .height(9.0 * 44.0)
                                .selected_text(label)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut want,
                                        None,
                                        "— default (the MIDI panel's port) —",
                                    );
                                    for port in &ports {
                                        ui.selectable_value(
                                            &mut want,
                                            Some(port.clone()),
                                            port,
                                        );
                                    }
                                })
                                .response
                                .on_hover_text(
                                    "The keyboard that plays this instrument's instance.\n\n\
                                     Every instance has a feed of its own, so two open \
                                     editors both hear everything rather than splitting \
                                     one stream between them, and two keyboards can play \
                                     two instruments at once.\n\n\
                                     “Default” follows whatever the MIDI panel is \
                                     connected to. Takes effect on the next note — \
                                     nothing is reopened, and the audio keeps running.",
                                );
                            if want != instrument.midi_source {
                                retarget = Some((idx, want));
                            }
                        });
                    });
            });
            ui.add_space(6.0);
        }

        // Applied after the loop: pointing a feed somewhere else talks to the
        // router, which the rows above are still borrowing.
        if let Some((idx, want)) = retarget {
            if let Some(instrument) = self.instruments.get_mut(idx) {
                let feed_id = instrument.feed.as_ref().map(|f| f.id);
                match feed_id.map_or(Ok(()), |id| self.midi_router.set_source(id, want.clone())) {
                    Ok(()) => {
                        instrument.midi_source = want;
                        // An instrument whose editor has never been opened has no feed
                        // yet; the choice is remembered and used when it is.
                        self.status = match &instrument.midi_source {
                            Some(port) => format!("{} now plays from {port}.", instrument.name),
                            None => format!("{} follows the MIDI panel's port.", instrument.name),
                        };
                    }
                    Err(e) => self.status = format!("Could not switch MIDI source: {e:#}"),
                }
            }
        }
        // An editor that has been closed since the last frame may have been the
        // last listener on its keyboard; this is what hands the device back.
        self.midi_router.release_unused();

        match action {
            Some(Action::Open(idx)) => {
                let router = self.midi_router.clone();
                if let Some(instrument) = self.instruments.get_mut(idx) {
                    if let Err(e) = instrument.open_editor(&router) {
                        self.status = format!("Open editor failed: {e:#}");
                    } else if instrument.editor.as_ref().is_some_and(|e| !e.is_audible()) {
                        self.status =
                            format!("Opened {} — audio output unavailable.", instrument.name);
                    } else {
                        self.status = String::new();
                    }
                }
            }
            Some(Action::Close(idx)) => {
                if let Some(instrument) = self.instruments.get_mut(idx) {
                    instrument.capture_editor_state(&self.registry);
                    instrument.editor = None;
                }
            }
            Some(Action::Export(idx)) => self.export_instrument(idx),
            Some(Action::Remove(idx)) => {
                if idx < self.instruments.len() {
                    // Nothing to capture: the instrument is going, and with it the
                    // registry entry any state would have been kept in.
                    let removed = self.instruments.remove(idx);
                    // Sequencer rows pointing here fall back to another instrument on
                    // their next frame.
                    self.registry.remove(removed.registry_id);
                    self.status = format!("Removed {}.", removed.name);
                }
            }
            None => {}
        }
    }
}

/// An instrument name nothing else can already be called: the readable part, then a
/// fresh UUID.
///
/// Two LeSynth instruments with the same name are not a cosmetic problem. A saved
/// project writes one `.lsft` per instrument named after the instrument, a row records
/// which instrument it plays by that name, and the Sequencer's select boxes are read
/// by it — so a duplicate means a grid file quietly overwritten by another
/// instrument's, and a row that cannot say which of the two it wanted. The counter in
/// front of the UUID is what a person reads; the UUID is what makes the
/// guarantee, including across projects and across machines.
pub fn unique_instrument_name(stem: &str) -> String {
    format!("{stem} · {}", uuid::Uuid::new_v4())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two instruments made the same way must never be called the same thing: a
    /// saved project names each grid file after its instrument, so a duplicate is a
    /// grid overwritten by another instrument's.
    #[test]
    fn a_fresh_instrument_name_is_never_a_repeat() {
        let names: std::collections::HashSet<String> =
            (0..64).map(|_| unique_instrument_name("LeSynth Fourier 1")).collect();
        assert_eq!(names.len(), 64);
        // The readable part still leads, so a select box shows what the instrument is
        // before it shows which one it is.
        assert!(
            unique_instrument_name("LeSynth Fourier 1").starts_with("LeSynth Fourier 1 · "),
            "the name no longer reads as a name"
        );
        // And it survives being turned into a file name.
        let name = unique_instrument_name("Voice");
        assert_eq!(
            crate::gui::sequencer::project::sanitize_name(&name).matches('-').count(),
            4,
            "the UUID did not survive sanitising: {name}"
        );
    }
}
