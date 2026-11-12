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
//! One plugin, whatever format it came in.
//!
//! The rest of the DAW — the Instruments panel, the Sequencer, the audio engine —
//! never asks which format a plugin is. It holds a [`PluginInstance`], sets it
//! up with [`PluginInstance::initialize_audio`], drives it through a
//! [`BlockProcessor`], saves and restores it as an opaque byte string, and asks
//! it for a [`PluginEditor`] to put in a window. Each format answers those in
//! its own module:
//!
//!   * **VST3** — [`crate::vst`], everywhere. The embedded LeSynth Fourier is one,
//!     and only a VST3 instance carries its harmonic-grid C ABI.
//!   * **CLAP** — [`clap`], everywhere.
//!   * **VST2** — [`vst2`], everywhere. Long deprecated and still what a great
//!     many installed plugins are.
//!   * **LV2** — [`lv2`], everywhere, though it is mostly a Linux format.
//!   * **Audio Unit** — [`au`], macOS only.
//!
//! What identifies a plugin is a path plus, where one file holds several
//! plugins, an id inside it: a VST3 class id, a CLAP plugin id, an LV2 plugin
//! URI. An Audio Unit has no path at all — the system registry finds it by its
//! three four-character codes, which ride in the id.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::track_format::TrackState;
use crate::vst::Vst3Instance;

#[cfg(target_os = "macos")]
pub mod au;
pub mod clap;
pub mod editor;
pub mod lv2;
pub mod processor;
pub mod scan;
pub mod vst2;

pub use editor::{ParentWindow, PluginEditor};
pub use processor::{BlockProcessor, MidiEvent};
pub use scan::{scan_installed, search_paths, FoundPlugin};

/// The audio bus layout a plugin settled on, so the engine can hand `process()`
/// buffers that match what it negotiated. Getting this wrong is not cosmetic: a
/// plugin with an audio input bus reads its inputs unconditionally, so an effect
/// handed no input buffers crashes the audio thread.
#[derive(Clone, Debug, Default)]
pub struct PluginIo {
    /// Channel count of each activated audio input bus, in bus order.
    pub inputs: Vec<usize>,
    /// Channel count of each activated audio output bus, in bus order.
    pub outputs: Vec<usize>,
    /// The largest block the plugin was set up for, in frames — a promise it
    /// sizes its own buffers to. Handing it a bigger one writes past them, and
    /// the corruption surfaces later as a crash somewhere else entirely, so
    /// whoever drives `process()` clamps to this. Zero before initialisation.
    pub max_block: usize,
}

impl PluginIo {
    /// Channels on the main (first) output bus — what actually reaches the
    /// speakers. Zero when the plugin has no audio output at all.
    pub fn main_output_channels(&self) -> usize {
        self.outputs.first().copied().unwrap_or(0)
    }
}

/// The plugin formats this host loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PluginFormat {
    Vst3,
    Clap,
    Vst2,
    Lv2,
    /// Audio Unit (version 2 API). macOS only.
    Au,
}

impl PluginFormat {
    /// Every format, in the order the plugin picker lists them.
    pub const ALL: [PluginFormat; 5] = [
        PluginFormat::Vst3,
        PluginFormat::Clap,
        PluginFormat::Vst2,
        PluginFormat::Lv2,
        PluginFormat::Au,
    ];

    /// How the format is written on screen.
    pub fn label(self) -> &'static str {
        match self {
            PluginFormat::Vst3 => "VST3",
            PluginFormat::Clap => "CLAP",
            PluginFormat::Vst2 => "VST2",
            PluginFormat::Lv2 => "LV2",
            PluginFormat::Au => "AU",
        }
    }

    /// Whether this build can load the format at all.
    pub fn is_supported(self) -> bool {
        self != PluginFormat::Au || cfg!(target_os = "macos")
    }
}

/// The `plugin_path` an Audio Unit is recorded under. It has no file of its own
/// that the host loads — the system finds it by the codes in its id — so this is
/// a name for the UI and a marker for [`detect_format`], never a path to open.
pub const AU_PATH: &str = "AudioUnit";

/// Prefix of an Audio Unit's plugin id: `au:<type>:<subtype>:<manufacturer>`.
pub const AU_ID_PREFIX: &str = "au:";

/// Whether `path` names a real file the plugin needs, as opposed to the
/// placeholder an Audio Unit is recorded under.
pub fn has_file(path: &Path, plugin_id: Option<&str>) -> bool {
    !(path == Path::new(AU_PATH) || plugin_id.is_some_and(|id| id.starts_with(AU_ID_PREFIX)))
}

/// Which format the plugin at `path` is.
///
/// The extension decides wherever there is one — that is how every format but
/// VST2 is installed. A bare library has to be opened and asked: our own
/// embedded plugin is a bare `.so` exporting both a VST3 factory and a CLAP
/// entry, and is a VST3 here because that is the instance its grid ABI lives on;
/// a VST2 is almost always a bare library.
pub fn detect_format(path: &Path, plugin_id: Option<&str>) -> Result<PluginFormat> {
    if !has_file(path, plugin_id) {
        return Ok(PluginFormat::Au);
    }
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "vst3" => return Ok(PluginFormat::Vst3),
        "clap" => return Ok(PluginFormat::Clap),
        "lv2" => return Ok(PluginFormat::Lv2),
        "component" => return Ok(PluginFormat::Au),
        // A macOS VST2 is a `.vst` bundle.
        "vst" => return Ok(PluginFormat::Vst2),
        "ttl" => return Ok(PluginFormat::Lv2),
        _ => {}
    }
    if !path.exists() {
        bail!("no such file or directory: {}", crate::file_label(path));
    }
    if path.is_dir() {
        if path.join("manifest.ttl").is_file() {
            return Ok(PluginFormat::Lv2);
        }
        // Something inside a VST3 bundle, which the VST3 loader digs out.
        return Ok(PluginFormat::Vst3);
    }
    // A library inside an LV2 bundle is that bundle's plugin.
    if path
        .parent()
        .is_some_and(|dir| dir.join("manifest.ttl").is_file())
    {
        return Ok(PluginFormat::Lv2);
    }
    library_format(path)
}

/// Open a plugin library that will run: one an instance or a UI is made from.
/// It stays mapped for the life of the process, whatever later drops it.
///
/// Unloading plugin code is not safe in general, and a Rust plugin proves it on
/// every platform. Each thread that ever ran its code — the GUI thread, a loader,
/// the audio thread — carries thread-local destructors that point into the
/// library, and runs them when it exits; unload first and that thread jumps into
/// unmapped memory. A plugin thread still running at unload does the same. On
/// Linux and macOS a lingering thread used to pin the library by accident; on
/// Windows nothing did, and closing LeSynth's editor or stopping the Sequencer
/// crashed the process. So the library is pinned on purpose: `RTLD_NODELETE`, or
/// a pinned module handle. The plugin's own exit hook still runs; only the
/// unmapping is skipped, and opening the same library again reuses the mapping.
///
/// Used for VST3 modules and CLAP libraries — the formats LeSynth ships in, so
/// the ones where this is proven — and only where the code will run: sniffing
/// a file for its entry point opens and closes it as usual. Not for VST2: Vital
/// (JUCE) hangs when a second instance is made in a mapping whose first one was
/// closed, so a VST2 still gets a fresh load each time.
pub(crate) fn open_pinned(path: &Path) -> std::result::Result<libloading::Library, libloading::Error> {
    #[cfg(unix)]
    {
        use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};
        // RTLD_LAZY | RTLD_LOCAL is what `Library::new` uses; NODELETE is the pin.
        let lib = unsafe { UnixLibrary::open(Some(path), RTLD_LAZY | RTLD_LOCAL | libc::RTLD_NODELETE)? };
        Ok(lib.into())
    }
    #[cfg(windows)]
    {
        use winapi::um::libloaderapi::{
            GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN,
        };
        let lib = unsafe { libloading::os::windows::Library::new(path)? };
        let module = lib.into_raw();
        let mut pinned = std::ptr::null_mut();
        let ok = unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                module as *const u16,
                &mut pinned,
            )
        };
        if ok == 0 {
            log::warn!("could not pin {}; it will be unloaded when released", path.display());
        }
        Ok(unsafe { libloading::os::windows::Library::from_raw(module) }.into())
    }
}

/// Ask a bare shared library which entry point it exports.
fn library_format(path: &Path) -> Result<PluginFormat> {
    let library = unsafe { libloading::Library::new(path) }
        .with_context(|| format!("Failed to open {}", crate::file_label(path)))?;
    let has = |name: &[u8]| unsafe { library.get::<*mut std::ffi::c_void>(name).is_ok() };
    if has(b"GetPluginFactory\0") {
        Ok(PluginFormat::Vst3)
    } else if has(b"clap_entry\0") {
        Ok(PluginFormat::Clap)
    } else if has(b"VSTPluginMain\0") || has(b"main_plugin\0") || has(b"main_macho\0") || has(b"main\0") {
        Ok(PluginFormat::Vst2)
    } else {
        bail!(
            "{} is not a plugin this host can load — it exports no VST3, CLAP or VST2 entry point",
            crate::file_label(path)
        )
    }
}

/// Check that `path` really is a loadable plugin, without starting it up — what
/// the Instruments panel calls the moment a plugin is picked, so a wrong choice is
/// reported while the user is still looking at the picker.
pub fn validate(path: &Path, plugin_id: Option<&str>) -> Result<PluginFormat> {
    let format = detect_format(path, plugin_id)?;
    if !format.is_supported() {
        bail!("{} plugins can only be loaded on macOS", format.label());
    }
    match format {
        PluginFormat::Vst3 => {
            crate::vst::validate_module(path)?;
        }
        PluginFormat::Clap => clap::validate(path, plugin_id)?,
        PluginFormat::Vst2 => vst2::validate(path)?,
        PluginFormat::Lv2 => lv2::validate(path, plugin_id)?,
        PluginFormat::Au => {}
    }
    Ok(format)
}

/// What a plugin calls itself and how it categorises itself, without making an
/// instance — the Instruments panel uses it to spot a drum kit. `None` when the
/// format has no such description or the plugin could not be read.
pub fn describe(path: &Path, plugin_id: Option<&str>) -> Option<(String, String)> {
    match detect_format(path, plugin_id).ok()? {
        PluginFormat::Vst3 => {
            let classes = crate::vst::scan_classes(path).ok()?;
            let class = classes
                .iter()
                .find(|c| c.category == "Audio Module Class")
                .or_else(|| classes.first())?;
            Some((class.name.clone(), class.subcategories.clone()))
        }
        PluginFormat::Clap => clap::describe(path, plugin_id),
        PluginFormat::Lv2 => lv2::describe(path, plugin_id),
        PluginFormat::Vst2 | PluginFormat::Au => None,
    }
}

enum Backend {
    Vst3(Vst3Instance),
    Clap(clap::ClapInstance),
    Vst2(vst2::Vst2Instance),
    Lv2(lv2::Lv2Instance),
    #[cfg(target_os = "macos")]
    Au(au::AuInstance),
}

/// A loaded plugin instance, of any format.
pub struct PluginInstance {
    backend: Backend,
}

impl PluginInstance {
    /// Load the plugin at `plugin_path` and initialise it.
    ///
    /// `class_id` picks a VST3 class, `plugin_id` a plugin inside a CLAP file or
    /// an LV2 bundle (or names an Audio Unit); `None` takes the first instrument
    /// or effect there is. `token` tags a LeSynth instance for its grid ABI and
    /// is ignored by everything else.
    pub fn load(
        plugin_path: &Path,
        class_id: Option<&[i8; 16]>,
        plugin_id: Option<&str>,
        token: Option<u64>,
    ) -> Result<Self> {
        let format = detect_format(plugin_path, plugin_id)?;
        let backend = match format {
            PluginFormat::Vst3 => Backend::Vst3(Vst3Instance::load(plugin_path, class_id, token)?),
            PluginFormat::Clap => Backend::Clap(clap::ClapInstance::load(plugin_path, plugin_id)?),
            PluginFormat::Vst2 => Backend::Vst2(vst2::Vst2Instance::load(plugin_path)?),
            PluginFormat::Lv2 => Backend::Lv2(lv2::Lv2Instance::load(plugin_path, plugin_id)?),
            #[cfg(target_os = "macos")]
            PluginFormat::Au => Backend::Au(au::AuInstance::load(
                plugin_id.context("an Audio Unit is named by its id, and none was given")?,
            )?),
            #[cfg(not(target_os = "macos"))]
            PluginFormat::Au => bail!("Audio Unit plugins can only be loaded on macOS"),
        };
        Ok(Self { backend })
    }

    /// Wrap a VST3 instance already made — the Sequencer opens each VST3 module
    /// once and makes all of its instances from it.
    pub fn from_vst3(instance: Vst3Instance) -> Self {
        Self { backend: Backend::Vst3(instance) }
    }

    pub fn format(&self) -> PluginFormat {
        match &self.backend {
            Backend::Vst3(_) => PluginFormat::Vst3,
            Backend::Clap(_) => PluginFormat::Clap,
            Backend::Vst2(_) => PluginFormat::Vst2,
            Backend::Lv2(_) => PluginFormat::Lv2,
            #[cfg(target_os = "macos")]
            Backend::Au(_) => PluginFormat::Au,
        }
    }

    /// The VST3 instance behind this, if it is one — the only kind that can be
    /// LeSynth Fourier and carry its grid.
    pub fn vst3(&self) -> Option<&Vst3Instance> {
        match &self.backend {
            Backend::Vst3(p) => Some(p),
            _ => None,
        }
    }

    /// The plugin's display name — the editor window's title.
    pub fn name(&self) -> &str {
        match &self.backend {
            Backend::Vst3(p) => p.name(),
            Backend::Clap(p) => p.name(),
            Backend::Vst2(p) => p.name(),
            Backend::Lv2(p) => p.name(),
            #[cfg(target_os = "macos")]
            Backend::Au(p) => p.name(),
        }
    }

    /// The bus layout negotiated by [`Self::initialize_audio`]; empty before it.
    pub fn io(&self) -> PluginIo {
        match &self.backend {
            Backend::Vst3(p) => p.io(),
            Backend::Clap(p) => p.io(),
            Backend::Vst2(p) => p.io(),
            Backend::Lv2(p) => p.io(),
            #[cfg(target_os = "macos")]
            Backend::Au(p) => p.io(),
        }
    }

    /// Set the plugin up for `sample_rate` and blocks of up to `max_block_size`
    /// frames, and switch it on. Calling it again on a running instance does
    /// nothing: a second editor for the same instrument must not reconfigure the
    /// plugin under the stream already driving it.
    pub fn initialize_audio(&self, sample_rate: f64, max_block_size: i32) -> Result<()> {
        match &self.backend {
            Backend::Vst3(p) => p.initialize_audio(sample_rate, max_block_size),
            Backend::Clap(p) => p.initialize_audio(sample_rate, max_block_size),
            Backend::Vst2(p) => p.initialize_audio(sample_rate, max_block_size),
            Backend::Lv2(p) => p.initialize_audio(sample_rate, max_block_size),
            #[cfg(target_os = "macos")]
            Backend::Au(p) => p.initialize_audio(sample_rate, max_block_size),
        }
    }

    /// The plugin's own saved state — every knob the user set in its editor —
    /// as an opaque byte string, the `.vststate` beside a project.
    ///
    /// Opaque to the host and **specific to the format**: a VST3's component
    /// state means nothing to the same plugin loaded as a CLAP. An instrument records
    /// which file it loads, and with it which format, so the two always travel
    /// together.
    pub fn save_state(&self) -> Result<Vec<u8>> {
        match &self.backend {
            Backend::Vst3(p) => p.component_state(),
            Backend::Clap(p) => p.save_state(),
            Backend::Vst2(p) => p.save_state(),
            Backend::Lv2(p) => p.save_state(),
            #[cfg(target_os = "macos")]
            Backend::Au(p) => p.save_state(),
        }
    }

    /// Put a state from [`Self::save_state`] back.
    pub fn restore_state(&self, bytes: &[u8]) -> Result<()> {
        anyhow::ensure!(!bytes.is_empty(), "empty plugin state");
        match &self.backend {
            Backend::Vst3(p) => p.set_component_state(bytes),
            Backend::Clap(p) => p.restore_state(bytes),
            Backend::Vst2(p) => p.restore_state(bytes),
            Backend::Lv2(p) => p.restore_state(bytes),
            #[cfg(target_os = "macos")]
            Backend::Au(p) => p.restore_state(bytes),
        }
    }

    /// An editor for this instance, ready to be embedded in a window by
    /// [`crate::gui::editor_window`]. Fails, saying why, when the plugin has no
    /// GUI this platform can host.
    pub fn create_editor(&self) -> Result<Box<dyn PluginEditor>> {
        match &self.backend {
            Backend::Vst3(p) => crate::vst::editor::create(p),
            Backend::Clap(p) => p.create_editor(),
            Backend::Vst2(p) => p.create_editor(),
            Backend::Lv2(p) => p.create_editor(),
            #[cfg(target_os = "macos")]
            Backend::Au(p) => p.create_editor(),
        }
    }

    /// Whether [`Self::create_editor`] has anything to offer, without making one.
    pub fn has_editor(&self) -> bool {
        match &self.backend {
            Backend::Vst3(p) => p.create_view().is_some(),
            Backend::Clap(p) => p.has_editor(),
            Backend::Vst2(p) => p.has_editor(),
            Backend::Lv2(p) => p.has_editor(),
            #[cfg(target_os = "macos")]
            Backend::Au(_) => false,
        }
    }

    /// The LeSynth instance behind this, or the reason there is none.
    fn lesynth(&self) -> Result<&Vst3Instance> {
        self.vst3()
            .context("only the embedded LeSynth Fourier (a VST3) carries a harmonic grid")
    }

    /// Export the live LeSynth grid. See [`Vst3Instance::export_state`].
    pub fn export_state(&self) -> Result<TrackState> {
        self.lesynth()?.export_state()
    }

    /// Load a LeSynth grid. See [`Vst3Instance::import_state`].
    pub fn import_state(&self, state: &TrackState) -> Result<()> {
        self.lesynth()?.import_state(state)
    }

    /// Hand a recorded subtrack to this LeSynth instance's editor for analysis.
    /// See [`Vst3Instance::push_analysis`].
    pub fn push_analysis(
        &self,
        samples: &[f32],
        sample_rate: f32,
        base_freq: f32,
        contour: &[f32],
    ) -> Result<()> {
        self.lesynth()?
            .push_analysis(samples, sample_rate, base_freq, contour)
    }

    /// The format-specific half of a [`BlockProcessor`]: whatever the plugin's
    /// `process()` call needs pointed at the scratch buffers.
    fn realtime(
        &self,
        scratch: &mut processor::AudioScratch,
        in_buses: &[usize],
        out_buses: &[usize],
        max_block: usize,
    ) -> Result<Box<dyn processor::RealtimeProcess>> {
        match &self.backend {
            Backend::Vst3(p) => crate::vst::realtime::create(p, scratch, in_buses, out_buses),
            Backend::Clap(p) => p.realtime(scratch, in_buses, out_buses),
            Backend::Vst2(p) => p.realtime(scratch, in_buses, out_buses),
            Backend::Lv2(p) => p.realtime(scratch, in_buses, out_buses, max_block),
            #[cfg(target_os = "macos")]
            Backend::Au(p) => p.realtime(scratch, in_buses, out_buses, max_block),
        }
    }
}

/// The name a plugin is listed and named under: the file or bundle, without the
/// extension or a `lib` prefix — "Dexed", never "libDexed.so".
pub fn display_stem(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| crate::file_label(path));
    match stem.strip_prefix("lib") {
        Some(rest) if !rest.is_empty() && path.extension().is_some_and(|e| e != "vst3") => {
            rest.to_string()
        }
        _ => stem,
    }
}

/// `dirs` with the ones that do not exist dropped and repeats removed, in order.
pub(crate) fn existing_dirs(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        if dir.is_dir() && !out.contains(&dir) {
            out.push(dir);
        }
    }
    out
}
