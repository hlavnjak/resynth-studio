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
//! Resynthesis panel — the headline feature.
//!
//! Flow:
//!   1. Pick an arbitrary `.wav` / `.mp3` / `.m4a` file.
//!   2. Decode it and segment it into pitch-stable *subtracks*
//!      (host-side, see [`crate::analysis`]).
//!   3. For each *reasonable* subtrack, hand it to LeSynth Fourier, which
//!      subdivides it into per-period *buckets* and extracts an
//!      amplitude/phase value per harmonic per bucket (the FFT step).
//!
//! The analysed grid is previewed inline here (a compact amplitude chart), and
//! can also be opened in a full LeSynth Fourier editor instance running in
//! Analysis mode, where individual harmonics can be toggled.
//!
//! A file can also skip all of that: **"Add whole file as Instrument"** publishes the
//! recording itself as a *wav instrument* — an Instrument a Sequencer row plays from its
//! start for the length of one note, with no pitch to pick and nothing
//! resynthesised. It is what a recording no additive analysis does justice to (a
//! drum loop, a spoken line, a whole take) is for, and what a project saves is
//! the path to the file.

use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;

use super::registry::InstrumentRegistry;
use super::instrument::EditorInstance;
use crate::analysis::{self, build_contour, Subtrack};
use crate::audio::capture::{self, Recorder, DEFAULT_INPUT};
use crate::audio::{decode_audio_file, write_wav_i16, AudioEngine, DecodedAudio};
use crate::midi::new_midi_queue;
use crate::plugin::PluginInstance;
use crate::vst::{class_ids, Vst3Instance};

/// Number of harmonics we extract / preview per subtrack.
const PREVIEW_HARMONICS: usize = 16;
/// Bucket count requested for the inline host-side preview.
const PREVIEW_BUCKETS: usize = 128;
/// Harmonics kept when a subtrack is turned into an Instrument — the plugin's own grid
/// height, so the registered grid is the full analysis rather than a preview of
/// it.
const INSTRUMENT_HARMONICS: usize = 256;

struct SubtrackView {
    sub: Subtrack,
    /// `[harmonic][bucket]` amplitude grid for the inline preview.
    preview_amp: Option<Vec<Vec<f32>>>,
    /// Set once this subtrack has been registered as a Track the Sequencer can
    /// play; the id it was registered under.
    registry_id: Option<u64>,
    /// The dedicated LeSynth editor for this subtrack, present only while its
    /// window is open. While this is `Some`, "Open in LeSynth" is replaced by a
    /// "Close" control, so repeat clicks never spawn duplicate instances.
    editor: Option<EditorInstance>,
}

/// One opened audio file: its path, decode result, and segmented subtracks. Each
/// subtrack may own an editor window; removing the file drops this whole struct,
/// which closes those editors via [`EditorInstance`]'s `Drop`.
struct AudioFile {
    /// Stable id, used to key per-file egui widget state across frames.
    id: u64,
    file_path: String,
    status: String,
    decoded: Option<DecodedAudio>,
    subtracks: Vec<SubtrackView>,
    /// Set once the *whole file* has been published as a Track the Sequencer can
    /// play as one note — the id it was registered under. Not an analysis: a wav
    /// instrument is the recording itself, played back, which is what a part no
    /// resynthesis does justice to (a drum loop, a spoken line, a field
    /// recording) needs.
    wav_instrument: Option<u64>,
}

impl AudioFile {
    fn new(id: u64, path: PathBuf) -> Self {
        let mut file = Self {
            id,
            file_path: path.display().to_string(),
            status: String::new(),
            decoded: None,
            subtracks: Vec::new(),
            wav_instrument: None,
        };
        file.decode_and_segment();
        file
    }

    /// The path this file was opened from.
    fn path(&self) -> PathBuf {
        PathBuf::from(self.file_path.trim())
    }

    /// How long the file runs, for the hint on the wav-instrument button.
    fn duration_secs(&self) -> f32 {
        self.decoded.as_ref().map(DecodedAudio::duration_secs).unwrap_or(0.0)
    }

    /// Number of subtracks whose editor window is currently open.
    fn open_editor_count(&self) -> usize {
        self.subtracks.iter().filter(|s| s.editor.is_some()).count()
    }

    /// Drop any editor whose window the user has closed, reclaiming its audio
    /// stream and plugin. Called every frame so the UI count and the running
    /// resources stay in sync with the actual windows on screen.
    fn reap_closed_editors(&mut self) {
        for s in &mut self.subtracks {
            if s.editor.as_ref().is_some_and(EditorInstance::is_closed) {
                s.editor = None;
            }
        }
    }

    /// Close every open editor for this file.
    fn close_all_editors(&mut self) {
        for s in &mut self.subtracks {
            s.editor = None;
        }
    }

    /// The trailing file name (for compact headers), falling back to the full path.
    fn display_name(&self) -> &str {
        let path = self.file_path.trim();
        path.rsplit(['/', '\\']).next().filter(|s| !s.is_empty()).unwrap_or(path)
    }

    fn sample_rate(&self) -> f32 {
        self.decoded.as_ref().map(|d| d.sample_rate).unwrap_or(44_100.0)
    }

    fn decode_and_segment(&mut self) {
        let path = PathBuf::from(self.file_path.trim());
        match decode_audio_file(&path) {
            Ok(audio) => {
                let subs = analysis::segment(&audio.samples, audio.sample_rate);
                let reasonable = subs
                    .iter()
                    .filter(|s| s.is_reasonable(audio.sample_rate))
                    .count();
                self.status = format!(
                    "Decoded {:.1}s @ {} Hz » {} subtrack(s), {} reasonable to analyse.",
                    audio.duration_secs(),
                    audio.sample_rate as u32,
                    subs.len(),
                    reasonable
                );
                self.subtracks = subs
                    .into_iter()
                    .map(|sub| SubtrackView {
                        sub,
                        preview_amp: None,
                        registry_id: None,
                        editor: None,
                    })
                    .collect();
                self.decoded = Some(audio);
            }
            Err(e) => {
                self.status = format!("Decode failed: {}", e);
                self.decoded = None;
                self.subtracks.clear();
            }
        }
    }

    /// Decoded samples, sample rate, base frequency and pitch contour for
    /// subtrack `idx` — the inputs the analysis FFI needs.
    fn analysis_inputs(&self, idx: usize) -> Option<(Vec<f32>, f32, f32, Vec<f32>)> {
        let audio = self.decoded.as_ref()?;
        let s = &self.subtracks.get(idx)?.sub;
        Some((
            audio.samples[s.start..s.end.min(audio.samples.len())].to_vec(),
            audio.sample_rate,
            s.base_freq,
            build_contour(s),
        ))
    }
}

pub struct ResynthPanel {
    /// All currently-open audio files.
    files: Vec<AudioFile>,
    /// Monotonic source of stable per-file ids.
    next_id: u64,
    /// Panel-wide status line (add/analysis errors, hints).
    status: String,
    /// Shared library handle used for the stateless analysis FFI calls, reused
    /// across every open file.
    ffi_plugin: Option<Arc<Vst3Instance>>,
    /// The app-wide instrument list. A subtrack analysed here can be published to it
    /// ("Add as Instrument"), which is what makes resynthesised material available to
    /// the Sequencer.
    registry: InstrumentRegistry,
    /// What the source select box offers. Empty until the panel is first drawn:
    /// asking the host for its devices wakes the sound system, which is not
    /// something to do at startup for a panel the user may never open.
    input_devices: Vec<String>,
    /// The source the next take records from, by the name in `input_devices`.
    input_device: String,
    /// The take in progress. `Some` is the whole recording state there is: the
    /// button, the clock and the disabled select box all read it.
    recorder: Option<Recorder>,
}

impl ResynthPanel {
    pub fn new(registry: InstrumentRegistry) -> Self {
        Self {
            files: Vec::new(),
            next_id: 0,
            status: "Add a .wav, .mp3 or .m4a file to begin.".to_string(),
            ffi_plugin: None,
            registry,
            input_devices: Vec::new(),
            input_device: DEFAULT_INPUT.to_string(),
            recorder: None,
        }
    }
}

impl ResynthPanel {
    fn internal_plugin_path() -> Option<PathBuf> {
        crate::internal_plugin_path()
    }

    /// Load (once) a plugin instance whose shared object backs the analysis
    /// FFI. Returns a clone of the loaded instance.
    fn ensure_ffi_plugin(&mut self) -> Option<Arc<Vst3Instance>> {
        if let Some(p) = &self.ffi_plugin {
            return Some(p.clone());
        }
        let path = Self::internal_plugin_path()?;
        // Stateless analysis FFI only — no editor, so no token needed.
        match Vst3Instance::load(&path, Some(&class_ids::FOURIER_SYNTH), None) {
            Ok(inst) => {
                let arc = Arc::new(inst);
                self.ffi_plugin = Some(arc.clone());
                Some(arc)
            }
            Err(e) => {
                self.status = format!("Could not load internal plugin for analysis: {}", e);
                None
            }
        }
    }

    /// Open a native file picker and add the chosen file as a new open file,
    /// decoding and segmenting it immediately.
    fn add_audio_file(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .add_filter("Audio (.wav, .mp3, .m4a)", &["wav", "mp3", "m4a"])
            .add_filter("All files", &["*"]);
        // Start from the directory of the most recently added file, if any.
        if let Some(last) = self.files.last() {
            let current = PathBuf::from(last.file_path.trim());
            if let Some(parent) = current.parent().filter(|p| p.is_dir()) {
                dialog = dialog.set_directory(parent);
            }
        }
        if let Some(path) = dialog.pick_file() {
            self.open_file(path);
        }
    }

    /// Start a take from the selected source. Nothing is written yet — a
    /// recording becomes a file when it is stopped, and is named after the
    /// moment it began.
    fn start_recording(&mut self) {
        match Recorder::start(Some(&self.input_device)) {
            Ok(rec) => {
                self.status = format!(
                    "Recording from {} — press Stop to save the take and analyse it.",
                    rec.device()
                );
                self.recorder = Some(rec);
            }
            Err(e) => self.status = format!("Cannot record: {e:#}"),
        }
    }

    /// Stop the take, write it, and put it through the same door a file opened
    /// by hand comes through — decode, segment, and a card per subtrack.
    fn stop_recording(&mut self) {
        let Some(rec) = self.recorder.take() else {
            return;
        };
        let take = rec.finish();
        if take.samples.is_empty() {
            self.status = format!(
                "Nothing came from {} — the take is empty, so no file was written.",
                take.device
            );
            return;
        }
        let dir = match capture::recordings_dir() {
            Ok(dir) => dir,
            Err(e) => {
                self.status = format!(
                    "Recorded {:.1}s but cannot save it: {e:#}",
                    take.duration_secs()
                );
                return;
            }
        };
        let path = dir.join(take.file_name());
        if let Err(e) = write_wav_i16(&path, &take.samples, take.channels, take.sample_rate) {
            self.status = format!(
                "Recorded {:.1}s but cannot write it: {e:#}",
                take.duration_secs()
            );
            return;
        }
        let secs = take.duration_secs();
        self.open_file(path.clone());
        // `open_file` sets the status to the analysis result, which is the more
        // useful half; the take itself is reported in front of it so the file
        // that was just written can be found.
        self.status = format!(
            "Recorded {secs:.1}s from {} » {}{} · {}",
            take.device,
            crate::file_label(&path),
            if take.truncated {
                " (cut at the length limit)"
            } else {
                ""
            },
            self.status
        );
    }

    /// Open `path` as an audio file of this panel: decode it, segment it, and
    /// show it. The other half of [`Self::add_audio_file`], with the dialog left
    /// out — which is also what lets a test lay this panel out with a file in it.
    fn open_file(&mut self, path: PathBuf) {
        let id = self.next_id;
        self.next_id += 1;
        let mut file = AudioFile::new(id, path);
        // The same file may already be a wav instrument — published before this panel
        // was cleared, or brought in by a loaded project. One file is one instrument,
        // so this view takes over the entry rather than offering to add a second
        // one just like it.
        file.wav_instrument = self.registry.find_wav(&file.path());
        self.status = file.status.clone();
        self.files.push(file);
    }

    /// Compute the inline preview grid for subtrack `sub_idx` of file `file_idx`
    /// via the plugin's stateless analysis FFI.
    fn preview_subtrack(&mut self, file_idx: usize, sub_idx: usize) {
        let Some((samples, sr, freq, contour)) =
            self.files.get(file_idx).and_then(|f| f.analysis_inputs(sub_idx))
        else {
            return;
        };
        let Some(plugin) = self.ensure_ffi_plugin() else { return };
        match plugin.analyze(&samples, sr, freq, &contour, PREVIEW_BUCKETS, PREVIEW_HARMONICS) {
            Ok((amp, _phase)) => {
                if let Some(view) =
                    self.files.get_mut(file_idx).and_then(|f| f.subtracks.get_mut(sub_idx))
                {
                    view.preview_amp = Some(amp);
                }
            }
            Err(e) => self.status = format!("Analyze failed: {}", e),
        }
    }

    /// Open a dedicated LeSynth Fourier instance for subtrack `sub_idx` of file
    /// `file_idx`, push the audio for analysis, and show its editor (Analysis mode).
    fn open_in_lesynth(&mut self, file_idx: usize, sub_idx: usize) {
        // Idempotent: if this subtrack already has an open editor, do nothing
        // rather than spawn a duplicate instance.
        if self
            .files
            .get(file_idx)
            .and_then(|f| f.subtracks.get(sub_idx))
            .is_some_and(|v| v.editor.is_some())
        {
            return;
        }

        let Some((samples, sr, freq, contour)) =
            self.files.get(file_idx).and_then(|f| f.analysis_inputs(sub_idx))
        else {
            return;
        };

        let Some(path) = Self::internal_plugin_path() else {
            self.status = "Could not locate internal plugin.".to_string();
            return;
        };
        // Tag the instance so its edited grid can be exported to a .lsft later.
        let inst = match Vst3Instance::load(
            &path,
            Some(&class_ids::FOURIER_SYNTH),
            Some(crate::vst::next_instance_token()),
        ) {
            Ok(i) => Arc::new(PluginInstance::from_vst3(i)),
            Err(e) => {
                self.status = format!("Plugin load failed: {e:#}");
                return;
            }
        };
        // Initialise the instance at the *output device* sample rate (not the
        // file's) so its piano renders at the correct pitch for playback. The
        // analysis itself carries the file's `sr` independently via
        // `push_analysis`, so this only affects synthesis/playback.
        let device_cfg = AudioEngine::query_device_config().ok();
        let (init_sr, init_block) = device_cfg
            .as_ref()
            .map(|c| (c.sample_rate, c.max_buffer_size as i32))
            .unwrap_or((sr as f64, 512));
        let _ = inst.initialize_audio(init_sr, init_block);

        // Queue the subtrack, then open the editor which will claim it.
        if let Err(e) = inst.push_analysis(&samples, sr, freq, &contour) {
            self.status = format!("Push analysis failed: {e:#}");
            return;
        }

        // Drive the dedicated instance's `process()` with its own audio stream so
        // the in-editor piano is audible. The piano writes voices directly into
        // the instance's shared state, so an empty MIDI queue is fine here.
        match EditorInstance::open(inst, new_midi_queue()) {
            Ok(editor) => {
                let audible = editor.is_audible();
                if let Some(view) =
                    self.files.get_mut(file_idx).and_then(|f| f.subtracks.get_mut(sub_idx))
                {
                    view.editor = Some(editor);
                }
                self.status = if audible {
                    format!("Opened subtrack {} in LeSynth (Analysis mode).", sub_idx + 1)
                } else {
                    format!(
                        "Opened subtrack {} in LeSynth (Analysis mode) — audio output unavailable.",
                        sub_idx + 1
                    )
                };
            }
            Err(e) => self.status = format!("Editor failed: {e:#}"),
        }
    }

    /// Publish subtrack `sub_idx` to the shared instrument list, so the Sequencer can
    /// put it on a row.
    ///
    /// The grid is analysed here and registered *with* the instrument, rather than
    /// registering a reference to an open editor: a resynthesised instrument is then
    /// playable in a sequence without its editor ever being opened, and the
    /// analysis is the plugin's own (`analyze_full`, full harmonic height), not
    /// the reduced inline preview.
    fn add_as_instrument(&mut self, file_idx: usize, sub_idx: usize) {
        let already = self
            .files
            .get(file_idx)
            .and_then(|f| f.subtracks.get(sub_idx))
            .and_then(|v| v.registry_id)
            .is_some_and(|id| self.registry.contains(id));
        if already {
            self.status = "That subtrack is already in the instrument list.".to_string();
            return;
        }
        let Some((samples, sr, freq, contour)) =
            self.files.get(file_idx).and_then(|f| f.analysis_inputs(sub_idx))
        else {
            return;
        };
        let Some(path) = Self::internal_plugin_path() else {
            self.status = "Could not locate the internal plugin.".to_string();
            return;
        };
        let Some(plugin) = self.ensure_ffi_plugin() else { return };
        // `0` buckets = the plugin's own period-synchronous bucketing, the only
        // one whose grid inverts back to the source exactly.
        let grid = match plugin.analyze_full(&samples, sr, freq, &contour, 0, INSTRUMENT_HARMONICS) {
            Ok(g) => g,
            Err(e) => {
                self.status = format!("Analysis failed: {e}");
                return;
            }
        };
        // A fresh UUID, as for every other LeSynth instrument: analysing the same
        // file twice must not produce two instruments called the same thing, one of
        // whose grids would overwrite the other's on the next save.
        let name = match self.files.get(file_idx) {
            Some(f) => crate::gui::instrument::unique_instrument_name(&format!(
                "{} · subtrack {}",
                f.display_name(),
                sub_idx + 1
            )),
            None => return,
        };
        let id = self.registry.add(
            &name,
            path,
            Some(class_ids::FOURIER_SYNTH),
            true,
            Some(grid.to_track_state()),
        );
        if let Some(view) = self.files.get_mut(file_idx).and_then(|f| f.subtracks.get_mut(sub_idx)) {
            view.registry_id = Some(id);
        }
        self.status = format!("Added “{name}” to the instrument list — pick it in the Sequencer.");
    }

    /// Publish the **whole file** to the shared instrument list as a wav instrument: a
    /// Instrument that plays the recording itself, from its start, for as long as the
    /// note a Sequencer row gives it.
    ///
    /// Nothing is analysed and nothing is decoded here — the recipe is the path,
    /// and the transport decodes it when it loads the sequence (once per
    /// file, however many rows play it). That is also what a project saves, so a
    /// wav instrument survives on any machine the file does.
    fn add_whole_file_as_instrument(&mut self, file_idx: usize) {
        let Some(file) = self.files.get(file_idx) else { return };
        let path = file.path();
        if !path.exists() {
            self.status = format!("{} is no longer there.", crate::file_label(&path));
            return;
        }
        // One instrument per file, as when a file is opened: two entries playing the
        // same recording would be indistinguishable in every select box.
        let name = match self.registry.find_wav(&path) {
            Some(id) => {
                self.files[file_idx].wav_instrument = Some(id);
                self.registry.name_of(id).unwrap_or_default()
            }
            None => {
                let name = crate::gui::instrument::unique_instrument_name(file.display_name());
                // The file is already decoded here, so its length is known
                // exactly — no need to go back to the container for it.
                let id = self.registry.add_wav(&name, path, file.duration_secs());
                self.files[file_idx].wav_instrument = Some(id);
                name
            }
        };
        self.status = format!("Added “{name}” to the instrument list as a wav instrument.");
    }

    /// Take the whole-file instrument back out of the instrument list.
    fn remove_whole_file_instrument(&mut self, file_idx: usize) {
        let Some(file) = self.files.get_mut(file_idx) else { return };
        if let Some(id) = file.wav_instrument.take() {
            self.registry.remove(id);
            self.status = format!("Removed {} from the instrument list.", file.display_name());
        }
    }

    /// Take a published subtrack back out of the instrument list. Sequencer rows using
    /// it fall back to another instrument on their next frame.
    fn remove_from_instruments(&mut self, file_idx: usize, sub_idx: usize) {
        let Some(view) = self.files.get_mut(file_idx).and_then(|f| f.subtracks.get_mut(sub_idx))
        else {
            return;
        };
        if let Some(id) = view.registry_id.take() {
            self.registry.remove(id);
            self.status = format!("Removed subtrack {} from the instrument list.", sub_idx + 1);
        }
    }

    /// Save subtrack `sub_idx`'s live (edited) LeSynth grid to a `.lsft` file.
    /// Resynthesis is export-only; loading happens in the Instruments panel.
    fn export_subtrack(&mut self, file_idx: usize, sub_idx: usize) {
        // Snapshot the grid first, ending the borrow before the file dialog.
        let snapshot = self
            .files
            .get(file_idx)
            .and_then(|f| f.subtracks.get(sub_idx))
            .and_then(|v| v.editor.as_ref())
            .map(|e| e.export_state());
        let Some(result) = snapshot else {
            return;
        };
        let state = match result {
            Ok(s) => s,
            Err(e) => {
                self.status = format!("Export failed: {}", e);
                return;
            }
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("LeSynth Fourier instrument (.lsft)", &["lsft"])
            .set_file_name(format!("subtrack_{}.lsft", sub_idx + 1))
            .save_file()
        else {
            return;
        };
        self.status = match state.write(&path) {
            Ok(()) => format!("Exported subtrack {} to {}.", sub_idx + 1, crate::file_label(&path)),
            Err(e) => format!("Export failed: {}", e),
        };
    }

    fn draw_amp_preview(ui: &mut egui::Ui, grid: &[Vec<f32>]) {
        let (rect, _resp) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 90.0), egui::Sense::hover());
        let painter = ui.painter().with_clip_rect(rect);
        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(20));
        let buckets = grid.first().map(|r| r.len()).unwrap_or(0);
        if buckets < 2 {
            return;
        }
        let palette = [
            egui::Color32::from_rgb(120, 200, 255),
            egui::Color32::from_rgb(255, 170, 120),
            egui::Color32::from_rgb(150, 255, 150),
            egui::Color32::from_rgb(255, 130, 200),
            egui::Color32::from_rgb(230, 230, 130),
        ];
        for (h, row) in grid.iter().enumerate().take(5) {
            let color = palette[h % palette.len()];
            let pts: Vec<egui::Pos2> = row
                .iter()
                .enumerate()
                .map(|(b, &v)| {
                    let x = rect.left() + rect.width() * b as f32 / (buckets - 1) as f32;
                    let y = rect.bottom() - rect.height() * v.clamp(0.0, 1.0);
                    egui::pos2(x, y)
                })
                .collect();
            painter.add(egui::Shape::line(pts, egui::Stroke::new(1.0, color)));
        }
    }

    /// Draw one subtrack card. Returns the action the user clicked, if any, so
    /// the caller can run it after the borrow of `self` ends.
    fn draw_subtrack(ui: &mut egui::Ui, view: &SubtrackView, sr: f32, idx: usize) -> Option<SubtrackAction> {
        let sub = &view.sub;
        let reasonable = sub.is_reasonable(sr);
        let mut action = None;
        let fill = if reasonable {
            egui::Color32::from_rgb(28, 42, 38)
        } else {
            egui::Color32::from_gray(34)
        };
        egui::Frame::new()
            .fill(fill)
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("Subtrack {}", idx + 1)).strong());
                    ui.label(format!(
                        "{:.0} Hz · {:.2}s · conf {:.2} · @{:.2}s",
                        sub.base_freq,
                        sub.duration_secs(sr),
                        sub.confidence,
                        sub.start as f32 / sr
                    ));
                    if !reasonable {
                        ui.label(
                            egui::RichText::new("(skipped — not pitched enough)")
                                .italics()
                                .color(egui::Color32::from_gray(160)),
                        );
                    } else if view.editor.is_some() {
                        ui.label(
                            egui::RichText::new("● in LeSynth")
                                .color(egui::Color32::from_rgb(130, 230, 150)),
                        );
                    }
                });
                if reasonable {
                    ui.horizontal(|ui| {
                        if ui.button("Preview FFT").clicked() {
                            action = Some(SubtrackAction::Preview);
                        }
                        // Publish to the shared instrument list, so a Sequencer row can
                        // play this subtrack.
                        if view.registry_id.is_some() {
                            if ui
                                .button("🗑 Remove from instruments")
                                .on_hover_text("Take this subtrack back out of the instrument list")
                                .clicked()
                            {
                                action = Some(SubtrackAction::RemoveInstrument);
                            }
                        } else if ui
                            .button("🎼 Add as Instrument")
                            .on_hover_text(
                                "Analyse this subtrack and add it to the instrument list, \
                                 so the Sequencer can play it",
                            )
                            .clicked()
                        {
                            action = Some(SubtrackAction::AddInstrument);
                        }
                        if view.editor.is_some() {
                            // Editor already open: offer to close it rather than
                            // spawn a duplicate instance.
                            if ui
                                .button("✖ Close editor")
                                .on_hover_text("Close this subtrack's LeSynth editor")
                                .clicked()
                            {
                                action = Some(SubtrackAction::Close);
                            }
                            // Save the current (edited) grid to a .lsft file.
                            if ui
                                .button("💾 Export…")
                                .on_hover_text("Save this subtrack's grid to a .lsft file")
                                .clicked()
                            {
                                action = Some(SubtrackAction::Export);
                            }
                        } else if ui.button("Open in LeSynth").clicked() {
                            action = Some(SubtrackAction::Open);
                        }
                    });
                    if let Some(grid) = &view.preview_amp {
                        Self::draw_amp_preview(ui, grid);
                    }
                }
            });
        ui.add_space(4.0);
        action
    }

    /// The record button and the source it records from.
    ///
    /// One button with two states rather than a Record and a Stop sitting side
    /// by side: there is one take at a time, so the control that starts it is
    /// the control that ends it, and it carries the clock so the length of what
    /// is being captured is on the button being watched.
    fn recording_ui(&mut self, ui: &mut egui::Ui) {
        // Asked for once, on the first frame this panel is drawn, rather than at
        // startup — enumerating devices wakes the sound system.
        if self.input_devices.is_empty() {
            self.input_devices = capture::input_device_names();
        }
        // Read what the button needs out of the recorder first: stopping takes
        // `self` mutably, and the label is drawn from the same borrow.
        let running = self
            .recorder
            .as_ref()
            .map(|rec| (rec.secs(), rec.device().to_string()));
        match running {
            Some((secs, device)) => {
                let mut stop = false;
                if ui
                    .button(format!(
                        "⏹ Stop recording and resynthesise · {}:{:04.1}",
                        (secs / 60.0) as u64,
                        secs % 60.0
                    ))
                    .on_hover_text(
                        "Stop the take, save it under the time it started, and \
                         segment it into subtracks — the same thing that happens \
                         to a file added by hand.",
                    )
                    .clicked()
                {
                    stop = true;
                }
                ui.label(
                    egui::RichText::new(format!("🔴 {device}"))
                        .color(egui::Color32::from_rgb(220, 90, 90)),
                );
                if stop {
                    self.stop_recording();
                }
            }
            None => {
                if ui
                    .button("➕ Start recording from source and then resynthesise")
                    .on_hover_text(format!(
                        "Record from the selected input and put the take straight \
                         through the analysis, with no file to name: it is saved \
                         under the moment it started, to the millisecond, in \
                         {}.",
                        capture::recordings_dir()
                            .map(|d| d.display().to_string())
                            .unwrap_or_else(|_| "~/ReSynthRecordings".to_string())
                    ))
                    .clicked()
                {
                    self.start_recording();
                }
                ui.label("from");
                // The id carries the list length: an egui popup measures itself
                // once per id, so a rescan that finds more devices would open a
                // box still sized for the shorter list.
                egui::ComboBox::from_id_salt(("resynth_input", self.input_devices.len()))
                    .selected_text(&self.input_device)
                    .width(220.0)
                    .height(260.0)
                    .show_ui(ui, |ui| {
                        for name in &self.input_devices {
                            ui.selectable_value(&mut self.input_device, name.clone(), name);
                        }
                    });
                if ui
                    .button("⟳")
                    .on_hover_text(
                        "Look for input devices again — for one plugged in since the app started",
                    )
                    .clicked()
                {
                    self.input_devices = capture::input_device_names();
                    if !self.input_devices.contains(&self.input_device) {
                        self.input_device = DEFAULT_INPUT.to_string();
                    }
                }
            }
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new(".wav / .mp3 / .m4a » LeSynth Fourier")
                .italics()
                .color(egui::Color32::from_gray(150)),
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button("➕ Add audio file…").clicked() {
                self.add_audio_file();
            }
            self.recording_ui(ui);
        });
        ui.add_space(2.0);
        ui.label(egui::RichText::new(&self.status).color(egui::Color32::from_gray(170)));
        ui.add_space(6.0);
        if !self.files.is_empty() {
            ui.separator();
            ui.add_space(4.0);
        }

        // Reap editors whose windows the user closed directly, so their audio
        // streams and plugins are released and the on-screen state stays honest.
        // Published subtracks also keep the registry's live-instance pointer in
        // step, so the Sequencer plays the grid as it is being edited here.
        for file in &mut self.files {
            file.reap_closed_editors();
            // A wav instrument can leave the registry without this panel: loading a
            // project drops the ones a previous load brought in. The button then
            // has to offer to add it again rather than to remove what is gone.
            if file.wav_instrument.is_some_and(|id| !self.registry.contains(id)) {
                file.wav_instrument = None;
            }
            for view in &file.subtracks {
                if let Some(id) = view.registry_id {
                    self.registry.set_live(
                        id,
                        view.editor.as_ref().map(|e| Arc::downgrade(e.plugin())),
                    );
                }
            }
        }

        // Deferred actions, so we don't mutate `self` while iterating/borrowing it.
        let mut to_remove: Option<usize> = None;
        let mut to_close_editors: Option<usize> = None;
        let mut to_add_wav: Option<usize> = None;
        let mut to_remove_wav: Option<usize> = None;
        let mut pending: Option<(usize, usize, SubtrackAction)> = None;

        for (file_idx, file) in self.files.iter().enumerate() {
            let sr = file.sample_rate();
            let editor_count = file.open_editor_count();
            let header_id = ui.make_persistent_id(("resynth_file", file.id));
            egui::Frame::group(ui.style())
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    egui::collapsing_header::CollapsingState::load_with_default_open(
                        ui.ctx(),
                        header_id,
                        true,
                    )
                    .show_header(ui, |ui| {
                        ui.label(egui::RichText::new(file.display_name()).strong());
                        ui.label(
                            egui::RichText::new(format!("· {} subtracks", file.subtracks.len()))
                                .color(egui::Color32::from_gray(150)),
                        );
                        // Right-aligned controls: remove the whole file, put it
                        // on the instrument list as it stands, and optionally just
                        // close its open editors.
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .button("🗑 Remove")
                                .on_hover_text("Remove this file and all its subtracks")
                                .clicked()
                            {
                                to_remove = Some(file_idx);
                            }
                            // The whole file as one Instrument, played as the
                            // recording rather than as an analysis of it.
                            if file.wav_instrument.is_some() {
                                if ui
                                    .button("🗑 Remove wav instrument")
                                    .on_hover_text(
                                        "Take this file's wav instrument back out of the \
                                         instrument list",
                                    )
                                    .clicked()
                                {
                                    to_remove_wav = Some(file_idx);
                                }
                            } else if ui
                                .button("🌊 Add whole file as Instrument")
                                .on_hover_text(format!(
                                    "Add the whole file ({:.1}s) to the instrument list as \
                                     a wav instrument: a Sequencer row plays the recording \
                                     itself, from its start, for as long as the note \
                                     it is given.\n\nNothing is analysed and nothing \
                                     is resynthesised — there is no pitch to pick, \
                                     only a length. The project saves the path to \
                                     this file.",
                                    file.duration_secs()
                                ))
                                .clicked()
                            {
                                to_add_wav = Some(file_idx);
                            }
                            if editor_count > 0
                                && ui.button(format!("Close editors ({})", editor_count)).clicked()
                            {
                                to_close_editors = Some(file_idx);
                            }
                        });
                    })
                    .body(|ui| {
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new(&file.status)
                                .color(egui::Color32::from_gray(180)),
                        );
                        ui.add_space(4.0);
                        for (sub_idx, view) in file.subtracks.iter().enumerate() {
                            if let Some(action) = Self::draw_subtrack(ui, view, sr, sub_idx) {
                                pending = Some((file_idx, sub_idx, action));
                            }
                        }
                    });
                });
            ui.add_space(8.0);
        }

        if let Some(idx) = to_add_wav {
            self.add_whole_file_as_instrument(idx);
        }
        if let Some(idx) = to_remove_wav {
            self.remove_whole_file_instrument(idx);
        }
        if let Some(idx) = to_close_editors {
            // Dropping the editors closes their windows (via `EditorInstance`'s
            // `Drop`) and stops each instance's audio stream.
            if let Some(file) = self.files.get_mut(idx) {
                file.close_all_editors();
            }
        }
        if let Some(idx) = to_remove {
            if idx < self.files.len() {
                let removed = self.files.remove(idx);
                // The file's subtracks go with it, so any instrument published from
                // them leaves the registry too — and so does the wav instrument, if
                // the whole file was published as one.
                for view in &removed.subtracks {
                    if let Some(id) = view.registry_id {
                        self.registry.remove(id);
                    }
                }
                if let Some(id) = removed.wav_instrument {
                    self.registry.remove(id);
                }
                self.status = format!("Removed {}.", removed.display_name());
            }
        }
        if let Some((file_idx, sub_idx, action)) = pending {
            match action {
                SubtrackAction::Preview => self.preview_subtrack(file_idx, sub_idx),
                SubtrackAction::Open => self.open_in_lesynth(file_idx, sub_idx),
                SubtrackAction::Close => {
                    if let Some(view) =
                        self.files.get_mut(file_idx).and_then(|f| f.subtracks.get_mut(sub_idx))
                    {
                        view.editor = None;
                    }
                }
                SubtrackAction::Export => self.export_subtrack(file_idx, sub_idx),
                SubtrackAction::AddInstrument => self.add_as_instrument(file_idx, sub_idx),
                SubtrackAction::RemoveInstrument => self.remove_from_instruments(file_idx, sub_idx),
            }
        }

        // A take's clock is the one thing here that moves without the user
        // touching anything, so while one runs the panel asks to be redrawn.
        if self.recorder.is_some() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

/// A user action requested on a subtrack card, resolved after the UI borrow ends.
#[derive(Clone, Copy)]
enum SubtrackAction {
    Preview,
    Open,
    Close,
    Export,
    AddInstrument,
    RemoveInstrument,
}
#[cfg(test)]
mod tests {
    use super::*;

    fn d5() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("D5.wav")
    }

    /// The whole file as one Instrument: the button publishes it, a second press of
    /// it — or the same file opened twice — finds the instrument that is already
    /// there rather than adding one indistinguishable from it, and removing the
    /// file takes it back out of the list.
    #[test]
    fn a_file_is_one_wav_instrument_however_often_it_is_asked_for() {
        let registry = InstrumentRegistry::default();
        let mut panel = ResynthPanel::new(registry.clone());
        panel.open_file(d5());
        assert!(panel.files[0].wav_instrument.is_none(), "nothing is published until it is asked for");

        panel.add_whole_file_as_instrument(0);
        let id = panel.files[0].wav_instrument.expect("published");
        assert!(registry.is_wav(id), "the instrument plays a file, not a plugin");
        assert_eq!(registry.find_wav(&d5()), Some(id));
        assert_eq!(registry.list().len(), 1);

        // The same file again — a second view of it takes over the same instrument.
        panel.open_file(d5());
        assert_eq!(panel.files[1].wav_instrument, Some(id));
        panel.add_whole_file_as_instrument(1);
        assert_eq!(registry.list().len(), 1, "one file is one instrument: {:?}", registry.list());

        panel.remove_whole_file_instrument(1);
        assert!(registry.list().is_empty());
        assert!(panel.files[1].wav_instrument.is_none());
    }

    /// Lay the panel out for real (headless egui, no window) with a file open,
    /// published and unpublished: the header carries two different controls in
    /// those two states, and a layout test is what catches a duplicate widget id
    /// or a rect that does not fit.
    #[test]
    fn the_panel_lays_out_with_a_published_file_without_panicking() {
        let registry = InstrumentRegistry::default();
        let mut panel = ResynthPanel::new(registry);
        panel.open_file(d5());
        let ctx = egui::Context::default();
        crate::gui::app::DawApp::configure_style(&ctx);
        let frame = |panel: &mut ResynthPanel| -> Vec<String> {
            let out = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
            });
            fn walk(sh: &egui::Shape, texts: &mut Vec<String>) {
                match sh {
                    egui::Shape::Text(t) => texts.push(t.galley.text().to_string()),
                    egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, texts)),
                    _ => {}
                }
            }
            let mut texts = Vec::new();
            for cs in &out.shapes {
                walk(&cs.shape, &mut texts);
            }
            texts
        };

        let texts = frame(&mut panel);
        assert!(
            texts.iter().any(|t| t.contains("Add whole file as Instrument")),
            "the whole-file control is missing: {texts:?}"
        );
        panel.add_whole_file_as_instrument(0);
        let texts = frame(&mut panel);
        assert!(
            texts.iter().any(|t| t.contains("Remove wav instrument")),
            "a published file should offer to unpublish: {texts:?}"
        );
    }
}
