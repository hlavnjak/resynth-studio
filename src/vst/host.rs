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
use std::ffi::{c_void, CStr};
use std::mem::zeroed;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::{bail, Context, Result};
use libloading::Library;

use crate::track_format::TrackState;

use vst3::Steinberg::{
    kResultOk, FUnknown, IBStream, IPluginFactory, IPluginFactoryTrait, IPluginBaseTrait, TUID,
};
use vst3::Steinberg::Vst::{
    BusDirections_, BusInfo, IAudioProcessor, IAudioProcessorTrait, IComponent, IComponentTrait,
    IComponentHandler, IConnectionPoint, IConnectionPointTrait, IEditController,
    IEditControllerTrait, IHostApplication, MediaTypes_, ProcessSetup, SpeakerArr,
    SpeakerArrangement,
};
use vst3::{ComPtr, ComRef, ComWrapper, Interface};

use crate::plugin::PluginIo;

use super::handler::ParamChangeHandler;
use super::param_changes::ParamEdits;
use super::host_context::{HostApplication, MemoryStream};
use super::module::{classes, platform_cid, Vst3Module};

// LeSynth Fourier's host-facing analysis C ABI (see lesynth-fourier/src/lib.rs).
// `contour` (ptr,len) is the per-position fundamental in Hz, uniformly resampled
// across the subtrack; null/0 means flat. Addressed to the instance carrying
// `token`, so another open editor cannot claim the job — the plugin's untargeted
// entry point is unusable here because we open several editors at once.
type PushAnalysisToProc =
    unsafe extern "C" fn(u64, *const f32, usize, f32, f32, *const f32, usize) -> i64;
type AnalyzeProc = unsafe extern "C" fn(
    *const f32,    // samples
    usize,         // len
    f32,           // sample_rate
    f32,           // base_freq
    *const f32,    // contour
    usize,         // contour_len
    usize,         // num_buckets
    usize,         // num_harmonics
    *mut f32,      // out_amp
    *mut f32,      // out_phase
) -> i64;
// Full analysis + offline resynthesis, the pair that lets the host reproduce the
// plugin's own grid and playback (see lesynth-fourier/src/lib.rs).
type AnalyzeFullProc = unsafe extern "C" fn(
    *const f32, // samples
    usize,      // len
    f32,        // sample_rate
    f32,        // base_freq
    *const f32, // contour
    usize,      // contour_len
    usize,      // num_buckets (0 = period-synchronous)
    usize,      // num_harmonics
    usize,      // max_buckets
    usize,      // cap_buckets
    *mut f32,   // out_amp
    *mut f32,   // out_phase
    *mut f32,   // out_pitch_ratio
    *mut f32,   // out_bucket_periods
    *mut f32,   // out_display_gain
    *mut f32,   // out_dc
    *mut f32,   // out_nyquist
) -> i64;
/// The exact inverse of the analysis — see `lesynth_fourier_resynthesize_exact`.
type ResynthesizeExactProc = unsafe extern "C" fn(
    usize,      // num_harmonics
    usize,      // num_buckets
    *const f32, // amp
    *const f32, // phase
    *const u32, // bucket_lengths
    *const f32, // dc (may be null)
    *const f32, // nyquist (may be null)
    f32,        // display_gain (0 = render at the grid's own level)
    f32,        // rate_ratio (output_rate / analysis_rate; 1.0 = exact)
    *mut f32,   // out
    usize,      // out_cap
) -> i64;
type ResynthesizeProc = unsafe extern "C" fn(
    usize,      // num_harmonics
    usize,      // num_buckets
    *const f32, // amp
    *const f32, // phase
    *const f32, // pitch_ratio
    f32,        // base_period (fractional — must not be rounded)
    usize,      // max_harmonic
    usize,      // target_samples
    f32,        // display_gain (0 = render at the grid's own level)
    *mut f32,   // out
    usize,      // out_cap
) -> i64;

/// The keyboard's own render path — see `lesynth_fourier_resynthesize_key`.
type ResynthesizeKeyProc = unsafe extern "C" fn(
    usize,       // num_harmonics
    usize,       // num_buckets
    *const f32,  // amp
    *const f32,  // phase
    *const u32,  // bucket_lengths
    *const f32,  // dc
    *const f32,  // nyquist
    *const f32,  // pitch_ratio
    f32,         // base_period (output samples, fractional)
    f32,         // base_freq
    f32,         // analysis_rate
    f32,         // out_rate
    usize,       // max_harmonic
    usize,       // target_samples
    f32,         // display_gain
    *mut f32,    // out
    usize,       // out_cap
) -> i64;

/// A key rendered through a live engine — see `lesynth_fourier_render_key_live`.
type RenderKeyLiveProc = unsafe extern "C" fn(
    *const f32, // samples
    usize,      // len
    f32,        // sample_rate
    f32,        // out_sample_rate
    f32,        // base_freq
    *const f32, // contour
    usize,      // contour_len
    usize,      // num_buckets
    usize,      // num_harmonics
    usize,      // key
    *mut i32,   // out_used_playback_grid
    *mut f32,   // out
    usize,      // out_cap
) -> i64;

// LeSynth Fourier's state save/load C ABI (see lesynth-fourier/src/lib.rs). The
// host tags an instance with a token before creating it, then exports/imports
// that exact instance's grid by token.
type PrepareInstanceProc = unsafe extern "C" fn(u64);
type ExportDimsProc =
    unsafe extern "C" fn(u64, *mut u32, *mut u32, *mut f32, *mut f32, *mut f32, *mut f32) -> i64;
/// `_export_grid` / `_import_grid` gained the exact inverse's per-bucket state
/// (lengths, DC, Nyquist) alongside the grid — see `.lsft` version 3. The three
/// pointers are nullable; passing them null exports/imports the grid alone.
type ExportGridProc = unsafe extern "C" fn(
    u64,      // token
    u32,      // num_harmonics
    u32,      // num_buckets
    *mut f32, // out_amp
    *mut f32, // out_phase
    *mut f32, // out_pitch_ratio
    *mut u32, // out_bucket_lengths
    *mut f32, // out_dc
    *mut f32, // out_nyquist
) -> i64;
type ImportGridProc = unsafe extern "C" fn(
    u64,        // token
    u32,        // num_harmonics
    u32,        // num_buckets
    f32,        // base_freq
    f32,        // duration_secs
    f32,        // sample_rate
    f32,        // display_gain (0 = unknown; no source-level audition)
    *const f32, // amp
    *const f32, // phase
    *const f32, // pitch_ratio
    *const u32, // bucket_lengths (null = no exact inverse)
    *const f32, // dc
    *const f32, // nyquist
) -> i64;
/// The per-harmonic enable checkboxes (`.lsft` version 4). Separate symbols
/// rather than more arguments on `_export_grid`: a plugin built before them is
/// still a working plugin, and a missing symbol here simply means "this build
/// cannot report them", not a failed export.
type ExportFlagsProc = unsafe extern "C" fn(u64, u32, *mut u8, *mut u8) -> i64;
type ImportFlagsProc = unsafe extern "C" fn(u64, u32, *const u8, *const u8) -> i64;

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

/// A process-unique token used to tag a plugin instance for state export/import.
/// Pass it to [`Vst3Instance::load`] right before creating the instance.
pub fn next_instance_token() -> u64 {
    NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)
}

/// The complete harmonic grid the plugin extracts from one subtrack — what
/// [`Vst3Instance::analyze_full`] returns and [`Vst3Instance::resynthesize`]
/// consumes. `amplitude`/`phase` are row-major `[h * num_buckets + b]`.
#[derive(Debug, Clone)]
pub struct AnalysisGrid {
    pub num_harmonics: usize,
    pub num_buckets: usize,
    pub amplitude: Vec<f32>,
    pub phase: Vec<f32>,
    /// Per-bucket fundamental relative to `base_freq` (the vibrato contour).
    pub pitch_ratio: Vec<f32>,
    /// Per-bucket length in **whole source samples** — the length of the inverse
    /// FFT that reproduces that bucket. These sum to the analysed subtrack, which
    /// is what makes [`Vst3Instance::resynthesize_exact`] exact.
    pub bucket_periods: Vec<f32>,
    /// Per-bucket DC (bin 0) and Nyquist (bin `N/2`) terms. Neither is a harmonic
    /// so neither has a row in the grid or a curve on the charts, but the inverse
    /// transform needs both: dropping DC alone costs ~120 dB of reconstruction
    /// accuracy, because one pitch period of real audio does not have zero mean.
    pub dc: Vec<f32>,
    pub nyquist: Vec<f32>,
    /// Gain the plugin's display normalisation applied to this grid
    /// (`grid_amplitude = source_amplitude × display_gain`). Pass it to
    /// [`Vst3Instance::resynthesize`] to get audio back at the analysed
    /// source's own level; the grid itself is scaled for chart legibility, which
    /// on a quiet recording is a boost of ~19 dB.
    pub display_gain: f32,
    pub base_freq: f32,
    pub sample_rate: f32,
    pub duration_secs: f32,
}

impl AnalysisGrid {
    pub fn amp(&self, harmonic: usize, bucket: usize) -> f32 {
        self.amplitude[harmonic * self.num_buckets + bucket]
    }

    pub fn phase(&self, harmonic: usize, bucket: usize) -> f32 {
        self.phase[harmonic * self.num_buckets + bucket]
    }

    /// Rendered period (samples, **fractional**) for `bucket` at `base_period`,
    /// matching the plugin's `bucket_period`: the base period scaled by the pitch
    /// ratio. Fractional because the renderer carries a fractional phase
    /// accumulator — rounding here would reintroduce the tuning error that
    /// accumulator exists to remove.
    pub fn rendered_period(&self, base_period: f32, bucket: usize) -> f32 {
        let r = self.pitch_ratio.get(bucket).copied().unwrap_or(1.0);
        (base_period / r.max(1e-3)).max(2.0)
    }

    /// The same grid as a [`TrackState`] — what an instance is loaded from and a
    /// `.lsft` is written in, so an analysis can become a playable instrument without
    /// an editor opening. `bucket_periods` are whole sample counts, `f32` only
    /// because that is the ABI's array type.
    pub fn to_track_state(&self) -> TrackState {
        TrackState {
            num_harmonics: self.num_harmonics,
            num_buckets: self.num_buckets,
            base_freq: self.base_freq,
            duration_secs: self.duration_secs,
            sample_rate: self.sample_rate,
            display_gain: self.display_gain,
            amplitude: self.amplitude.clone(),
            phase: self.phase.clone(),
            pitch_ratio: self.pitch_ratio.clone(),
            bucket_lengths: self.bucket_periods.iter().map(|&p| p as u32).collect(),
            dc: self.dc.clone(),
            nyquist: self.nyquist.clone(),
            // A fresh analysis has no editor behind it, so every harmonic is on.
            amp_enabled: Vec::new(),
            phase_enabled: Vec::new(),
        }
    }
}

/// Represents a loaded and initialized VST3 plugin instance.
pub struct Vst3Instance {
    pub component: ComPtr<IComponent>,
    pub processor: ComPtr<IAudioProcessor>,
    pub controller: ComPtr<IEditController>,
    /// The two halves of a plugin whose controller is a separate object, so they
    /// can be disconnected before either is terminated.
    connection: Option<(ComPtr<IConnectionPoint>, ComPtr<IConnectionPoint>)>,
    /// Held for the plugin: it keeps our context pointer and calls back into it.
    _host_context: ComPtr<FUnknown>,
    /// Ditto for the parameter handler we gave the controller.
    _param_handler: ComPtr<IComponentHandler>,
    /// What that handler collects: the edits the plugin's own editor made and
    /// is waiting to be told about in `process()`. See [`ParamEdits`].
    param_edits: ParamEdits,
    /// True once the controller is a distinct object that we initialised (and so
    /// must terminate) ourselves.
    separate_controller: bool,
    /// The bus layout negotiated in `initialize_audio`.
    io: RwLock<PluginIo>,
    /// Whether the component is active / processing, so teardown is ordered.
    active: AtomicBool,
    /// Display name from the factory's class info — the editor window's title.
    name: String,
    /// Token this instance was tagged with (LeSynth only), so its live grid can
    /// be addressed by the state export/import C ABI. `None` for untagged loads.
    token: Option<u64>,
    // Must outlive every COM pointer above: dropping it unloads the library they
    // all live in, so it is deliberately the last field.
    module: Arc<Vst3Module>,
}

impl Vst3Instance {
    /// The parameter edits the plugin's editor is waiting to be handed back in
    /// `process()`. Whoever drives the plugin's audio has to drain this into
    /// [`ProcessData::inputParameterChanges`]; a driver that does not leaves
    /// every control in the plugin's own editor dead.
    ///
    /// [`ProcessData::inputParameterChanges`]: vst3::Steinberg::Vst::ProcessData
    pub fn param_edits(&self) -> &ParamEdits {
        &self.param_edits
    }

    /// Load a VST3 plugin and initialize it.
    ///
    /// `plugin_path` may be a bare shared library or a `.vst3` bundle directory;
    /// see [`crate::vst::module::resolve_module_path`]. `class_id` is the 16-byte
    /// factory class to select (`None` = the first audio-module class). `token`,
    /// if given, tags the instance before it is created so its live grid can be
    /// exported/imported by token; ignored if the plugin lacks the symbol.
    pub fn load(
        plugin_path: &Path,
        class_id: Option<&[i8; 16]>,
        token: Option<u64>,
    ) -> Result<Self> {
        let module = Arc::new(Vst3Module::open(plugin_path)?);
        Self::from_module(module, class_id, token)
    }

    /// Create an instance from a module that is already open.
    ///
    /// A VST3 module is meant to be loaded once per process, and its
    /// `ModuleEntry` is not something to race: a caller making several instances
    /// of the same plugin opens the module once and calls this per instance.
    /// Note that this being safe to *call* concurrently says nothing about the
    /// plugin behind it — see `load_instances` in the Sequencer's player.
    pub fn from_module(
        module: Arc<Vst3Module>,
        class_id: Option<&[i8; 16]>,
        token: Option<u64>,
    ) -> Result<Self> {
        unsafe {
            // Tag the next-created instance, if requested and supported. Must
            // happen before `createInstance` triggers the plugin's constructor.
            if let Some(tok) = token {
                if let Ok(prepare) = module
                    .library()
                    .get::<PrepareInstanceProc>(b"lesynth_fourier_prepare_instance\0")
                {
                    prepare(tok);
                }
            }

            let factory = module.factory()?;
            let factory_ref = factory.as_com_ref();
            let (cid, name) = select_class(factory_ref, class_id)
                .with_context(|| format!("in {}", crate::file_label(module.path())))?;

            // Instantiate audio component
            let mut comp_ptr: *mut c_void = std::ptr::null_mut();
            let hr = factory_ref.createInstance(
                cid.as_ptr(),
                IComponent::IID.as_ptr() as *const i8,
                &mut comp_ptr,
            );
            anyhow::ensure!(
                hr == kResultOk && !comp_ptr.is_null(),
                "the factory refused to create '{name}' as an IComponent ({hr:#X})"
            );
            let component: ComPtr<IComponent> =
                ComPtr::from_raw(comp_ptr as *mut IComponent).context("IComponent ptr was null")?;

            // The host context. A plugin keeps this pointer for its whole life and
            // allocates its inter-component messages through it, so it is stored
            // on the instance rather than dropped at the end of this function.
            let host_context = HostApplication::new()
                .to_com_ptr::<IHostApplication>()
                .context("Failed to create the host context")?
                .upcast::<FUnknown>();

            let hr = component.as_com_ref().initialize(host_context.as_ptr());
            anyhow::ensure!(
                hr == kResultOk,
                "'{name}' refused to initialize ({hr:#X})"
            );

            // The controller is either the same object (single-component plugins,
            // ours included) or a second class the component names.
            let mut separate_controller = false;
            let controller = match component.cast::<IEditController>() {
                Some(c) => c,
                None => {
                    let c = create_separate_controller(
                        factory_ref,
                        component.as_com_ref(),
                        host_context.as_ptr(),
                    )
                    .with_context(|| format!("'{name}' has no usable edit controller"))?;
                    separate_controller = true;
                    c
                }
            };

            // A separate controller starts out knowing nothing about the state the
            // processor was created with; the host is what carries it across.
            if separate_controller {
                transfer_component_state(component.as_com_ref(), controller.as_com_ref());
            }

            // Set component handler for parameter changes. Its queue is kept
            // on the instance, because acknowledging an edit is not delivering
            // it — whoever drives `process()` has to hand it back.
            let param_edits = ParamEdits::default();
            let param_handler =
                ComWrapper::new(ParamChangeHandler { edits: param_edits.clone() })
                    .to_com_ptr::<IComponentHandler>()
                    .context("Failed to create component handler")?;
            controller
                .as_com_ref()
                .setComponentHandler(param_handler.as_ptr());

            // Connect the two halves, so the plugin's own messages get through.
            let connection = if separate_controller {
                connect(component.as_com_ref(), controller.as_com_ref())
            } else {
                None
            };

            // Get audio processor
            let processor = component
                .clone()
                .cast::<IAudioProcessor>()
                .with_context(|| format!("'{name}' is not an audio processor"))?;

            log::info!("Loaded '{}' from {}", name, module.path().display());

            Ok(Vst3Instance {
                component,
                processor,
                controller,
                connection,
                _host_context: host_context,
                _param_handler: param_handler,
                param_edits,
                separate_controller,
                io: RwLock::new(PluginIo::default()),
                active: AtomicBool::new(false),
                name,
                token,
                module,
            })
        }
    }

    /// The plugin's display name, from the factory's class info.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The bus layout negotiated by [`Self::initialize_audio`]; empty before it.
    pub fn io(&self) -> PluginIo {
        self.io.read().unwrap().clone()
    }

    /// The library this instance lives in, for the LeSynth C ABI below.
    fn lib(&self) -> &Library {
        self.module.library()
    }

    /// Export this instance's live harmonic grid into a [`TrackState`]. Requires
    /// the instance to have been loaded with a token (LeSynth only).
    pub fn export_state(&self) -> Result<TrackState> {
        let token = self
            .token
            .context("this instance was not tagged for export")?;
        unsafe {
            let dims: libloading::Symbol<ExportDimsProc> = self
                .lib()
                .get(b"lesynth_fourier_export_dims\0")
                .context("plugin does not export lesynth_fourier_export_dims")?;
            let (mut nh, mut nb, mut base, mut dur, mut sr) = (0u32, 0u32, 0f32, 0f32, 0f32);
            let mut gain = 0f32;
            let rc = dims(token, &mut nh, &mut nb, &mut base, &mut dur, &mut sr, &mut gain);
            anyhow::ensure!(rc == 0, "export_dims failed ({rc}); instance not found");
            let (nhz, nbz) = (nh as usize, nb as usize);
            anyhow::ensure!(nhz > 0 && nbz > 0, "instance has an empty grid to export");

            let grid = nhz * nbz;
            let mut amplitude = vec![0f32; grid];
            let mut phase = vec![0f32; grid];
            let mut pitch_ratio = vec![0f32; nbz];
            let mut bucket_lengths = vec![0u32; nbz];
            let mut dc = vec![0f32; nbz];
            let mut nyquist = vec![0f32; nbz];

            let grid_fn: libloading::Symbol<ExportGridProc> = self
                .lib()
                .get(b"lesynth_fourier_export_grid\0")
                .context("plugin does not export lesynth_fourier_export_grid")?;
            let rc2 = grid_fn(
                token,
                nh,
                nb,
                amplitude.as_mut_ptr(),
                phase.as_mut_ptr(),
                pitch_ratio.as_mut_ptr(),
                bucket_lengths.as_mut_ptr(),
                dc.as_mut_ptr(),
                nyquist.as_mut_ptr(),
            );
            anyhow::ensure!(rc2 >= 0, "export_grid failed ({rc2})");

            // Zeroed lengths mean this grid was never analysed period-
            // synchronously (a hand-drawn Synth grid), so there is no exact
            // inverse to save and all three are dropped together.
            if bucket_lengths.iter().any(|&n| n < 2) {
                bucket_lengths.clear();
                dc.clear();
                nyquist.clear();
            }

            // The enable checkboxes, which live beside the grid rather than in
            // it. Optional: a plugin built before this ABI exports the grid
            // fine, and "all enabled" is what such a build plays anyway.
            let (mut amp_enabled, mut phase_enabled) = (Vec::new(), Vec::new());
            if let Ok(flags_fn) = self
                .lib()
                .get::<ExportFlagsProc>(b"lesynth_fourier_export_flags\0")
            {
                let mut amp_flags = vec![0u8; nhz];
                let mut phase_flags = vec![0u8; nhz];
                let rc = flags_fn(token, nh, amp_flags.as_mut_ptr(), phase_flags.as_mut_ptr());
                if rc >= 0 {
                    // All-on is the default, and stays the empty vector so a
                    // grid nobody edited is not saved as an explicit selection.
                    if amp_flags.contains(&0) {
                        amp_enabled = amp_flags.iter().map(|&f| f != 0).collect();
                    }
                    if phase_flags.contains(&0) {
                        phase_enabled = phase_flags.iter().map(|&f| f != 0).collect();
                    }
                } else {
                    log::warn!("export_flags failed ({rc}); saving with every harmonic enabled");
                }
            }

            let state = TrackState {
                num_harmonics: nhz,
                num_buckets: nbz,
                base_freq: base,
                duration_secs: dur,
                sample_rate: sr,
                display_gain: gain,
                amplitude,
                phase,
                pitch_ratio,
                bucket_lengths,
                dc,
                nyquist,
                amp_enabled,
                phase_enabled,
            };
            state.validate()?;
            Ok(state)
        }
    }

    /// Load a saved [`TrackState`] into this instance (Analysis mode). Requires
    /// the instance to have been loaded with a token (LeSynth only).
    pub fn import_state(&self, state: &TrackState) -> Result<()> {
        let token = self
            .token
            .context("this instance was not tagged for import")?;
        state.validate()?;
        unsafe {
            let import: libloading::Symbol<ImportGridProc> = self
                .lib()
                .get(b"lesynth_fourier_import_grid\0")
                .context("plugin does not export lesynth_fourier_import_grid")?;
            let rc = import(
                token,
                state.num_harmonics as u32,
                state.num_buckets as u32,
                state.base_freq,
                state.duration_secs,
                state.sample_rate,
                state.display_gain,
                state.amplitude.as_ptr(),
                state.phase.as_ptr(),
                state.pitch_ratio.as_ptr(),
                // A pre-v3 file or a hand-drawn grid has no exact inverse to
                // restore; null tells the plugin to leave that path switched off
                // rather than invert a grid with lengths it never recorded.
                if state.supports_exact_inverse() {
                    state.bucket_lengths.as_ptr()
                } else {
                    std::ptr::null()
                },
                if state.supports_exact_inverse() {
                    state.dc.as_ptr()
                } else {
                    std::ptr::null()
                },
                if state.supports_exact_inverse() {
                    state.nyquist.as_ptr()
                } else {
                    std::ptr::null()
                },
            );
            anyhow::ensure!(rc == 0, "import_grid failed ({rc})");

            // The enable checkboxes, after the grid: loading a grid leaves them
            // alone but rebuilds the key buffers, so the flags have to be
            // applied (and the buffers re-dirtied) once it has landed.
            if !state.amp_enabled.is_empty() || !state.phase_enabled.is_empty() {
                match self
                    .lib()
                    .get::<ImportFlagsProc>(b"lesynth_fourier_import_flags\0")
                {
                    Ok(flags_fn) => {
                        let to_bytes = |v: &Vec<bool>| -> Vec<u8> {
                            if v.is_empty() {
                                vec![1u8; state.num_harmonics]
                            } else {
                                v.iter().map(|&e| u8::from(e)).collect()
                            }
                        };
                        let amp = to_bytes(&state.amp_enabled);
                        let phase = to_bytes(&state.phase_enabled);
                        let rc = flags_fn(
                            token,
                            state.num_harmonics as u32,
                            amp.as_ptr(),
                            phase.as_ptr(),
                        );
                        if rc < 0 {
                            log::warn!("import_flags failed ({rc}); harmonics stay as loaded");
                        }
                    }
                    // Worth saying out loud: the instrument plays with harmonics the
                    // user switched off, and nothing on screen would show why.
                    Err(e) => log::warn!(
                        "plugin has no lesynth_fourier_import_flags ({e}); the saved \
                         per-harmonic selection cannot be restored"
                    ),
                }
            }
        }
        Ok(())
    }

    /// Prepare the plugin for audio processing: negotiate its bus layout,
    /// activate the buses, and switch it on.
    ///
    /// The layout is *asked for*, not assumed. Only an instrument has no audio
    /// input, and only some plugins are happy with stereo — so the buses the
    /// plugin declares are what we set up, activate and later feed. The result is
    /// stored on the instance ([`Self::io`]) because the audio callback has to
    /// hand `process()` exactly the buffers this negotiated.
    pub fn initialize_audio(&self, sample_rate: f64, max_block_size: i32) -> Result<()> {
        unsafe {
            let comp_ref = self.component.as_com_ref();
            let proc_ref = self.processor.as_com_ref();

            // Re-entering this (a second editor for the same instrument) must not
            // reconfigure a running plugin underneath itself.
            if self.active.swap(true, Ordering::SeqCst) {
                return Ok(());
            }

            // 1) What buses does it have?
            let inputs = bus_channel_counts(comp_ref, MediaTypes_::kAudio as i32, BusDirections_::kInput as i32);
            let outputs = bus_channel_counts(comp_ref, MediaTypes_::kAudio as i32, BusDirections_::kOutput as i32);

            // 2) Ask for exactly those arrangements. A plugin may answer with a
            //    layout of its own, so the negotiated one is read back below.
            let in_arr: Vec<u64> = inputs.iter().map(|&n| speaker_arrangement(n)).collect();
            let out_arr: Vec<u64> = outputs.iter().map(|&n| speaker_arrangement(n)).collect();
            let res = proc_ref.setBusArrangements(
                in_arr.as_ptr() as *mut _,
                in_arr.len() as i32,
                out_arr.as_ptr() as *mut _,
                out_arr.len() as i32,
            );
            if res != kResultOk {
                log::info!(
                    "'{}' declined the {}-in/{}-out arrangement ({res:#X}); using its own",
                    self.name,
                    in_arr.len(),
                    out_arr.len()
                );
            }

            // 3) Setup processing
            let setup = ProcessSetup {
                processMode: 0,
                sampleRate: sample_rate,
                maxSamplesPerBlock: max_block_size,
                symbolicSampleSize:
                    vst3::Steinberg::Vst::SymbolicSampleSizes_::kSample32 as i32,
            };
            let res = proc_ref.setupProcessing(&setup as *const _ as *mut _);
            if res != kResultOk {
                log::warn!("'{}' rejected setupProcessing ({res:#X})", self.name);
            }

            // 4) Activate every bus. A bus left inactive may be skipped by the
            //    plugin entirely — silence out of the audio buses, and no notes
            //    at all through the event bus, which is how a hosted synth ends
            //    up looking like it "opened but does nothing".
            let mut io = PluginIo {
                max_block: max_block_size.max(0) as usize,
                ..PluginIo::default()
            };
            for (idx, _) in inputs.iter().enumerate() {
                comp_ref.activateBus(MediaTypes_::kAudio as i32, BusDirections_::kInput as i32, idx as i32, 1);
            }
            for (idx, _) in outputs.iter().enumerate() {
                comp_ref.activateBus(MediaTypes_::kAudio as i32, BusDirections_::kOutput as i32, idx as i32, 1);
            }
            let event_ins = comp_ref.getBusCount(MediaTypes_::kEvent as i32, BusDirections_::kInput as i32);
            for idx in 0..event_ins {
                comp_ref.activateBus(MediaTypes_::kEvent as i32, BusDirections_::kInput as i32, idx, 1);
            }

            // Read the arrangement back: it is the plugin's answer, not our ask,
            // that says how many channel pointers `process()` will dereference.
            for (idx, &requested) in inputs.iter().enumerate() {
                io.inputs.push(negotiated_channels(
                    proc_ref,
                    BusDirections_::kInput as i32,
                    idx,
                    requested,
                ));
            }
            for (idx, &requested) in outputs.iter().enumerate() {
                io.outputs.push(negotiated_channels(
                    proc_ref,
                    BusDirections_::kOutput as i32,
                    idx,
                    requested,
                ));
            }

            // 5) Switch on
            let res = comp_ref.setActive(1);
            if res != kResultOk {
                log::warn!("'{}' rejected setActive ({res:#X})", self.name);
            }
            proc_ref.setProcessing(1);

            log::info!(
                "'{}' initialized: sr={}, block={}, in={:?}, out={:?}, event buses={}",
                self.name,
                sample_rate,
                max_block_size,
                io.inputs,
                io.outputs,
                event_ins
            );
            *self.io.write().unwrap() = io;
        }
        Ok(())
    }

    /// Push a recorded subtrack to this instance for analysis; its editor picks
    /// the job up, switches to Analysis mode and shows the grid.
    ///
    /// Addressed by token, so it waits for *this* editor — the untargeted entry
    /// point would let another open editor claim it and leave this one empty.
    /// `contour` is the per-position fundamental in Hz; empty = flat.
    pub fn push_analysis(
        &self,
        samples: &[f32],
        sample_rate: f32,
        base_freq: f32,
        contour: &[f32],
    ) -> Result<()> {
        let token = self
            .token
            .context("this instance was not tagged; cannot target an analysis push")?;
        unsafe {
            let func: libloading::Symbol<PushAnalysisToProc> = self
                .lib()
                .get(b"lesynth_fourier_push_analysis_to\0")
                .context("plugin does not export lesynth_fourier_push_analysis_to")?;
            let rc = func(
                token,
                samples.as_ptr(),
                samples.len(),
                sample_rate,
                base_freq,
                contour.as_ptr(),
                contour.len(),
            );
            anyhow::ensure!(rc == 0, "push_analysis_to failed ({rc})");
        }
        Ok(())
    }

    /// Stateless harmonic analysis via the plugin's exported DSP, for the
    /// host's own preview plotting. Returns `(amp, phase)` grids shaped
    /// `[num_harmonics][num_buckets]`.
    pub fn analyze(
        &self,
        samples: &[f32],
        sample_rate: f32,
        base_freq: f32,
        contour: &[f32],
        num_buckets: usize,
        num_harmonics: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let mut amp_flat = vec![0.0f32; num_harmonics * num_buckets];
        let mut phase_flat = vec![0.0f32; num_harmonics * num_buckets];
        let written = unsafe {
            let func: libloading::Symbol<AnalyzeProc> = self
                .lib()
                .get(b"lesynth_fourier_analyze\0")
                .context("plugin does not export lesynth_fourier_analyze")?;
            func(
                samples.as_ptr(),
                samples.len(),
                sample_rate,
                base_freq,
                contour.as_ptr(),
                contour.len(),
                num_buckets,
                num_harmonics,
                amp_flat.as_mut_ptr(),
                phase_flat.as_mut_ptr(),
            )
        };
        anyhow::ensure!(written >= 0, "plugin analyze returned error {}", written);

        let amp = (0..num_harmonics)
            .map(|h| amp_flat[h * num_buckets..(h + 1) * num_buckets].to_vec())
            .collect();
        let phase = (0..num_harmonics)
            .map(|h| phase_flat[h * num_buckets..(h + 1) * num_buckets].to_vec())
            .collect();
        Ok((amp, phase))
    }

    /// Full stateless analysis through the plugin's DSP: the grid
    /// `analyze_and_load` builds, including the per-bucket pitch ratio and period
    /// that [`Self::analyze`] leaves out. `num_buckets == 0` selects the plugin's
    /// own period-synchronous bucketing, whose count comes from the source —
    /// hence the two-call probe below.
    pub fn analyze_full(
        &self,
        samples: &[f32],
        sample_rate: f32,
        base_freq: f32,
        contour: &[f32],
        num_buckets: usize,
        num_harmonics: usize,
    ) -> Result<AnalysisGrid> {
        unsafe {
            let func: libloading::Symbol<AnalyzeFullProc> = self
                .lib()
                .get(b"lesynth_fourier_analyze_full\0")
                .context("plugin does not export lesynth_fourier_analyze_full")?;

            // Probe for the bucket count the analysis will derive, then allocate
            // exactly that and repeat (the analysis is deterministic).
            let nb = func(
                samples.as_ptr(),
                samples.len(),
                sample_rate,
                base_freq,
                contour.as_ptr(),
                contour.len(),
                num_buckets,
                num_harmonics,
                0,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            anyhow::ensure!(nb > 0, "analyze_full probe returned {nb}");
            let nb = nb as usize;

            let mut amp = vec![0.0f32; num_harmonics * nb];
            let mut phase = vec![0.0f32; num_harmonics * nb];
            let mut pitch_ratio = vec![0.0f32; nb];
            let mut bucket_periods = vec![0.0f32; nb];
            let mut dc = vec![0.0f32; nb];
            let mut nyquist = vec![0.0f32; nb];
            let mut display_gain = 0.0f32;
            let rc = func(
                samples.as_ptr(),
                samples.len(),
                sample_rate,
                base_freq,
                contour.as_ptr(),
                contour.len(),
                num_buckets,
                num_harmonics,
                0,
                nb,
                amp.as_mut_ptr(),
                phase.as_mut_ptr(),
                pitch_ratio.as_mut_ptr(),
                bucket_periods.as_mut_ptr(),
                &mut display_gain,
                dc.as_mut_ptr(),
                nyquist.as_mut_ptr(),
            );
            anyhow::ensure!(rc == nb as i64, "analyze_full returned {rc}, expected {nb}");

            Ok(AnalysisGrid {
                num_harmonics,
                num_buckets: nb,
                amplitude: amp,
                phase,
                pitch_ratio,
                bucket_periods,
                dc,
                nyquist,
                display_gain,
                base_freq,
                sample_rate,
                duration_secs: if sample_rate > 0.0 {
                    samples.len() as f32 / sample_rate
                } else {
                    0.0
                },
            })
        }
    }

    /// Reproduce the analysed source from its grid — the exact inverse of
    /// [`Self::analyze_full`], exact to float rounding. Use
    /// [`Self::resynthesize`] for anything transposed, which must resample.
    ///
    /// `restore_source_level` divides out [`AnalysisGrid::display_gain`] for the
    /// file's own level. `output_sample_rate` is what this will be played at: the
    /// grid's own rate is the bit-exact case, any other band-limit-resamples the
    /// finished reconstruction.
    pub fn resynthesize_exact(
        &self,
        grid: &AnalysisGrid,
        restore_source_level: bool,
        output_sample_rate: f32,
    ) -> Result<Vec<f32>> {
        unsafe {
            let func: libloading::Symbol<ResynthesizeExactProc> = self
                .lib()
                .get(b"lesynth_fourier_resynthesize_exact\0")
                .context("plugin does not export lesynth_fourier_resynthesize_exact")?;

            let lengths: Vec<u32> = grid.bucket_periods.iter().map(|&p| p as u32).collect();
            let gain = if restore_source_level { grid.display_gain } else { 0.0 };
            // Bucket lengths are in the sample rate the audio was analysed at.
            // Rendering them into a stream at any other rate has to scale, or the
            // note plays at `out/analysis` times its pitch.
            let ratio = if grid.sample_rate > 0.0 && output_sample_rate > 0.0 {
                output_sample_rate / grid.sample_rate
            } else {
                1.0
            };
            let n = func(
                grid.num_harmonics,
                grid.num_buckets,
                grid.amplitude.as_ptr(),
                grid.phase.as_ptr(),
                lengths.as_ptr(),
                grid.dc.as_ptr(),
                grid.nyquist.as_ptr(),
                gain,
                ratio,
                std::ptr::null_mut(),
                0,
            );
            anyhow::ensure!(n > 0, "resynthesize_exact returned {n}");
            let mut out = vec![0.0f32; n as usize];
            let rc = func(
                grid.num_harmonics,
                grid.num_buckets,
                grid.amplitude.as_ptr(),
                grid.phase.as_ptr(),
                lengths.as_ptr(),
                grid.dc.as_ptr(),
                grid.nyquist.as_ptr(),
                gain,
                ratio,
                out.as_mut_ptr(),
                out.len(),
            );
            anyhow::ensure!(rc == n, "resynthesize_exact returned {rc}, expected {n}");
            Ok(out)
        }
    }

    /// Render a grid back to audio through the plugin's own playback path, with
    /// no plugin state involved. `base_period` is the **fractional** fundamental
    /// period in samples, `max_harmonic` an anti-alias cap (`0` = Nyquist only),
    /// `target_samples` the "preserve seconds" length (`0` = one cycle per
    /// bucket). `restore_source_level` divides out
    /// [`AnalysisGrid::display_gain`] for the level comparable against the file;
    /// `false` renders at the grid's level, which is what a key plays.
    pub fn resynthesize(
        &self,
        grid: &AnalysisGrid,
        base_period: f32,
        max_harmonic: usize,
        target_samples: usize,
        restore_source_level: bool,
    ) -> Result<Vec<f32>> {
        unsafe {
            let func: libloading::Symbol<ResynthesizeProc> = self
                .lib()
                .get(b"lesynth_fourier_resynthesize\0")
                .context("plugin does not export lesynth_fourier_resynthesize")?;
            let call = |out: *mut f32, cap: usize| {
                func(
                    grid.num_harmonics,
                    grid.num_buckets,
                    grid.amplitude.as_ptr(),
                    grid.phase.as_ptr(),
                    grid.pitch_ratio.as_ptr(),
                    base_period,
                    max_harmonic,
                    target_samples,
                    if restore_source_level { grid.display_gain } else { 0.0 },
                    out,
                    cap,
                )
            };
            let n = call(std::ptr::null_mut(), 0);
            anyhow::ensure!(n >= 0, "resynthesize returned error {n}");
            let mut out = vec![0.0f32; n as usize];
            let written = call(out.as_mut_ptr(), out.len());
            anyhow::ensure!(written == n, "resynthesize length changed ({written} vs {n})");
            Ok(out)
        }
    }

    /// Render a key **the way the keyboard does** — through the plugin's
    /// `PlaybackGrid` and the source's own two clocks.
    ///
    /// [`Self::resynthesize`] is not that path — it renders the analysis grid's
    /// *rounded* buckets on a uniform time grid, so a dump made through it cannot
    /// show a keyboard defect. `base_period` is the key's period in output
    /// samples, fractional.
    #[allow(clippy::too_many_arguments)]
    pub fn resynthesize_key(
        &self,
        grid: &AnalysisGrid,
        base_period: f32,
        max_harmonic: usize,
        target_samples: usize,
        output_sample_rate: f32,
        restore_source_level: bool,
    ) -> Result<Vec<f32>> {
        unsafe {
            let func: libloading::Symbol<ResynthesizeKeyProc> = self
                .lib()
                .get(b"lesynth_fourier_resynthesize_key\0")
                .context("plugin does not export lesynth_fourier_resynthesize_key")?;
            let lengths: Vec<u32> = grid.bucket_periods.iter().map(|&p| p as u32).collect();
            let call = |out: *mut f32, cap: usize| {
                func(
                    grid.num_harmonics,
                    grid.num_buckets,
                    grid.amplitude.as_ptr(),
                    grid.phase.as_ptr(),
                    lengths.as_ptr(),
                    grid.dc.as_ptr(),
                    grid.nyquist.as_ptr(),
                    grid.pitch_ratio.as_ptr(),
                    base_period,
                    grid.base_freq,
                    grid.sample_rate,
                    output_sample_rate,
                    max_harmonic,
                    target_samples,
                    if restore_source_level { grid.display_gain } else { 0.0 },
                    out,
                    cap,
                )
            };
            let n = call(std::ptr::null_mut(), 0);
            anyhow::ensure!(n >= 0, "resynthesize_key returned error {n}");
            let mut out = vec![0.0f32; n as usize];
            let written = call(out.as_mut_ptr(), out.len());
            anyhow::ensure!(written == n, "resynthesize_key length changed ({written} vs {n})");
            Ok(out)
        }
    }

    /// Render a key through a **live engine** inside the plugin — the same
    /// `load_analysis` + `assemble_buffer_for_key` an editor key press makes.
    ///
    /// [`Self::resynthesize_key`] hands the renderer a `PlaybackGrid` built for
    /// it, so it can only prove the renderer sound. The live path builds that
    /// grid from the plugin's own `SharedParams` and falls back to the contour
    /// renderer if anything is missing — silently, same call, different signal.
    /// `used_playback_grid` is `false` when that happened, which answers "the
    /// offline dump is clean but the plugin still buzzes".
    #[allow(clippy::too_many_arguments)]
    pub fn render_key_live(
        &self,
        samples: &[f32],
        sample_rate: f32,
        out_sample_rate: f32,
        base_freq: f32,
        contour: &[f32],
        num_buckets: usize,
        num_harmonics: usize,
        key: usize,
    ) -> Result<(Vec<f32>, bool)> {
        unsafe {
            let func: libloading::Symbol<RenderKeyLiveProc> = self
                .lib()
                .get(b"lesynth_fourier_render_key_live\0")
                .context("plugin does not export lesynth_fourier_render_key_live")?;
            let mut used: i32 = 0;
            let call = |out: *mut f32, cap: usize, used: *mut i32| {
                func(
                    samples.as_ptr(),
                    samples.len(),
                    sample_rate,
                    out_sample_rate,
                    base_freq,
                    // The pitch contour. Passing null here analyses a *flat*
                    // source — every bucket the same period, no vibrato — and
                    // lands 52 dB from `resynthesize_key` on the same key, which
                    // reads as "the keyboard plays something else" when the probe
                    // was simply asking for a different note.
                    if contour.is_empty() { std::ptr::null() } else { contour.as_ptr() },
                    contour.len(),
                    num_buckets,
                    num_harmonics,
                    key,
                    used,
                    out,
                    cap,
                )
            };
            let n = call(std::ptr::null_mut(), 0, std::ptr::null_mut());
            anyhow::ensure!(n >= 0, "render_key_live returned error {n}");
            let mut out = vec![0.0f32; n as usize];
            let written = call(out.as_mut_ptr(), out.len(), &mut used);
            anyhow::ensure!(written == n, "render_key_live length changed ({written} vs {n})");
            Ok((out, used != 0))
        }
    }

    /// The plugin's own saved state — `IComponent::getState`, which is how a
    /// VST3 hands over everything it wants remembered (its parameters, and
    /// whatever else it keeps) in a format only it understands.
    ///
    /// This is the general mechanism, and the only one a third-party plugin
    /// has. LeSynth's harmonic grid travels separately, over
    /// [`Self::export_state`], because the host itself has to read and write
    /// that one.
    pub fn component_state(&self) -> Result<Vec<u8>> {
        unsafe {
            let stream = MemoryStream::new();
            let stream_ptr = stream
                .to_com_ptr::<IBStream>()
                .context("Failed to create a state stream")?;
            let hr = self.component.as_com_ref().getState(stream_ptr.as_ptr());
            anyhow::ensure!(hr == kResultOk, "'{}' would not save its state ({hr:#X})", self.name);
            Ok(stream.bytes())
        }
    }

    /// Put a state from [`Self::component_state`] back into this instance.
    ///
    /// The controller is told as well: the processor is what plays, but the
    /// editor is what the user sees, and a plugin whose two halves disagree
    /// opens showing values it is not using.
    pub fn set_component_state(&self, bytes: &[u8]) -> Result<()> {
        anyhow::ensure!(!bytes.is_empty(), "empty plugin state");
        unsafe {
            let stream = MemoryStream::from_bytes(bytes);
            let stream_ptr = stream
                .to_com_ptr::<IBStream>()
                .context("Failed to create a state stream")?;
            let hr = self.component.as_com_ref().setState(stream_ptr.as_ptr());
            anyhow::ensure!(
                hr == kResultOk,
                "'{}' would not load the saved state ({hr:#X})",
                self.name
            );
            stream.rewind();
            let hr = self
                .controller
                .as_com_ref()
                .setComponentState(stream_ptr.as_ptr());
            if hr != kResultOk {
                log::debug!(
                    "'{}' controller declined the state ({hr:#X}); its editor may \
                     show stale values",
                    self.name
                );
            }
        }
        Ok(())
    }

    /// Create plugin editor view (returns raw pointer for window embedding).
    /// Returns None if the plugin has no editor.
    pub fn create_view(&self) -> Option<ComPtr<vst3::Steinberg::IPlugView>> {
        unsafe {
            let ctrl_ref = self.controller.as_com_ref();
            let raw_view =
                ctrl_ref.createView(CStr::from_bytes_with_nul(b"editor\0").unwrap().as_ptr());
            if raw_view.is_null() {
                None
            } else {
                ComPtr::from_raw(raw_view as *mut vst3::Steinberg::IPlugView)
            }
        }
    }
}

impl Drop for Vst3Instance {
    /// Shut the plugin down in the order the spec lays out. Skipping this is not
    /// harmless: a JUCE plugin that is still active when its library is unloaded
    /// takes the process with it.
    fn drop(&mut self) {
        unsafe {
            if self.active.swap(false, Ordering::SeqCst) {
                self.processor.as_com_ref().setProcessing(0);
                self.component.as_com_ref().setActive(0);
            }
            if let Some((comp_cp, ctrl_cp)) = self.connection.take() {
                comp_cp.as_com_ref().disconnect(ctrl_cp.as_ptr());
                ctrl_cp.as_com_ref().disconnect(comp_cp.as_ptr());
            }
            self.controller.as_com_ref().setComponentHandler(std::ptr::null_mut());
            if self.separate_controller {
                self.controller.as_com_ref().terminate();
            }
            self.component.as_com_ref().terminate();
        }
    }
}

/// Pick the class to instantiate out of the factory.
///
/// With a `class_id` it is an exact lookup. Without one, the first **audio
/// module** class wins — not simply the first class, which in a third-party
/// bundle is as likely to be the edit-controller class (or an ARA extension) and
/// would fail to create as an `IComponent`.
///
/// `class_id` comes in — and the classes are compared — in the crate's own byte
/// order; the id returned is in the factory's, ready for `createInstance`. On
/// every platform but Windows those are the same order. See
/// [`platform_cid`](crate::vst::module::platform_cid).
fn select_class(
    factory: ComRef<'_, IPluginFactory>,
    class_id: Option<&[i8; 16]>,
) -> Result<([i8; 16], String)> {
    const AUDIO_MODULE_CLASS: &str = "Audio Module Class";
    let mut fallback: Option<([i8; 16], String)> = None;
    for (idx, class) in classes(factory).into_iter().enumerate() {
        log::info!(
            "  class {idx}: '{}' [{}] {}",
            class.name,
            class.category,
            class.subcategories
        );
        if let Some(target) = class_id {
            if class.cid == *target {
                return Ok((platform_cid(class.cid), class.name));
            }
            continue;
        }
        if class.category == AUDIO_MODULE_CLASS {
            return Ok((platform_cid(class.cid), class.name));
        }
        fallback.get_or_insert((platform_cid(class.cid), class.name));
    }
    if class_id.is_some() {
        bail!("the requested plugin class is not in this factory");
    }
    // No class called itself an audio module: take whatever there was, so a
    // plugin with an unusual category is still worth a try.
    fallback.context("the plugin's factory offers no classes at all")
}

/// Create the edit controller a component names, for plugins whose two halves are
/// separate objects (most Steinberg-SDK plugins; JUCE keeps them together).
unsafe fn create_separate_controller(
    factory: ComRef<'_, IPluginFactory>,
    component: ComRef<'_, IComponent>,
    host_context: *mut FUnknown,
) -> Result<ComPtr<IEditController>> {
    let mut cid: TUID = zeroed();
    let hr = component.getControllerClassId(&mut cid);
    anyhow::ensure!(hr == kResultOk, "getControllerClassId failed ({hr:#X})");

    let mut ptr: *mut c_void = std::ptr::null_mut();
    let hr = factory.createInstance(
        cid.as_ptr(),
        IEditController::IID.as_ptr() as *const i8,
        &mut ptr,
    );
    anyhow::ensure!(
        hr == kResultOk && !ptr.is_null(),
        "the factory refused to create the edit controller ({hr:#X})"
    );
    let controller: ComPtr<IEditController> =
        ComPtr::from_raw(ptr as *mut IEditController).context("controller ptr was null")?;

    let hr = controller.as_com_ref().initialize(host_context);
    anyhow::ensure!(hr == kResultOk, "the edit controller refused to initialize ({hr:#X})");
    Ok(controller)
}

/// Pump the component's state through to the controller, so the editor opens
/// showing what the processor is actually set to.
unsafe fn transfer_component_state(
    component: ComRef<'_, IComponent>,
    controller: ComRef<'_, IEditController>,
) {
    let stream = MemoryStream::new();
    let Some(stream_ptr) = stream.to_com_ptr::<IBStream>() else {
        return;
    };
    if component.getState(stream_ptr.as_ptr()) != kResultOk {
        return;
    }
    stream.rewind();
    let hr = controller.setComponentState(stream_ptr.as_ptr());
    if hr != kResultOk {
        log::debug!("setComponentState returned {hr:#X} ({} bytes)", stream.byte_len());
    }
}

/// Wire the component and controller to each other. Returns the pair so they can
/// be disconnected again before either is terminated.
unsafe fn connect(
    component: ComRef<'_, IComponent>,
    controller: ComRef<'_, IEditController>,
) -> Option<(ComPtr<IConnectionPoint>, ComPtr<IConnectionPoint>)> {
    let comp_cp = component.cast::<IConnectionPoint>()?;
    let ctrl_cp = controller.cast::<IConnectionPoint>()?;
    comp_cp.as_com_ref().connect(ctrl_cp.as_ptr());
    ctrl_cp.as_com_ref().connect(comp_cp.as_ptr());
    Some((comp_cp, ctrl_cp))
}

/// Channel count of every bus of one media type and direction.
unsafe fn bus_channel_counts(
    component: ComRef<'_, IComponent>,
    media: i32,
    direction: i32,
) -> Vec<usize> {
    let count = component.getBusCount(media, direction);
    (0..count)
        .map(|idx| {
            let mut info: BusInfo = zeroed();
            if component.getBusInfo(media, direction, idx, &mut info) == kResultOk {
                info.channelCount.max(0) as usize
            } else {
                0
            }
        })
        .collect()
}

/// What the plugin actually settled on for one bus, falling back to what we asked
/// for if it does not answer.
unsafe fn negotiated_channels(
    processor: ComRef<'_, IAudioProcessor>,
    direction: i32,
    index: usize,
    requested: usize,
) -> usize {
    let mut arrangement: SpeakerArrangement = 0;
    if processor.getBusArrangement(direction, index as i32, &mut arrangement) == kResultOk {
        return arrangement.count_ones() as usize;
    }
    requested
}

/// The speaker arrangement for a plain `n`-channel bus.
fn speaker_arrangement(channels: usize) -> u64 {
    match channels {
        0 => SpeakerArr::kEmpty,
        1 => SpeakerArr::kMono,
        2 => SpeakerArr::kStereo,
        // No named arrangement: the low `n` speaker bits, which is what hosts
        // and plugins both fall back to for unusual counts.
        n if n < 64 => (1u64 << n) - 1,
        _ => SpeakerArr::kStereo,
    }
}


/// Well-known class IDs
pub mod class_ids {
    /// LeSynth Fourier: ASCII bytes of "LeSynthFourier01"
    pub const FOURIER_SYNTH: [i8; 16] = [
        76, 101, 83, 121, 110, 116, 104, 70, 111, 117, 114, 105, 101, 114, 48, 49,
    ];
}
#[cfg(test)]
mod tests {
    use super::*;
    use vst3::Steinberg::Vst::ParameterInfo;

    /// The embedded plugin, which every checkout has.
    fn internal_plugin() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("internal_plugins")
            .join("liblesynth_fourier.so")
    }

    /// `(id, normalized value)` of the plugin's first parameter.
    fn first_parameter(plugin: &Vst3Instance) -> Option<(u32, f64)> {
        unsafe {
            let ctrl = plugin.controller.as_com_ref();
            let mut info: ParameterInfo = zeroed();
            if ctrl.getParameterCount() < 1 || ctrl.getParameterInfo(0, &mut info) != kResultOk {
                return None;
            }
            Some((info.id, ctrl.getParamNormalized(info.id)))
        }
    }

    /// A control in a plugin's *own* editor only works if the host closes the
    /// loop. The editor does not write the parameter: it calls the host back on
    /// `IComponentHandler::performEdit` and waits to be handed the value in
    /// `process()` — and while it is processing audio it will accept it by no
    /// other route (nih-plug returns early from `setParamNormalized` too). So a
    /// host that acknowledges `performEdit` and drops it leaves every slider in
    /// every plugin editor dead, the handle springing back on the next frame,
    /// with nothing anywhere to say why.
    ///
    /// This drives the whole path the way an editor does: the handler's queue,
    /// the `IParameterChanges` list, one `process()` call, and the parameter
    /// read back off the controller.
    #[test]
    fn a_plugins_editor_moving_a_knob_reaches_the_plugin() {
        use crate::vst::param_changes::ParamChanges;
        use vst3::Steinberg::Vst::{
            IParameterChanges, ParamID, ParamValue, ProcessData, SymbolicSampleSizes_,
        };

        let path = internal_plugin();
        let Ok(plugin) = Vst3Instance::load(&path, Some(&class_ids::FOURIER_SYNTH), None) else {
            println!("no internal plugin at {} — nothing to test", path.display());
            return;
        };
        let Some((id, original)) = first_parameter(&plugin) else {
            println!("the plugin exposes no parameters — nothing to test");
            return;
        };
        plugin
            .initialize_audio(44100.0, 512)
            .expect("the plugin must set up for audio");

        // What the editor would do: one `performEdit`, no writing of its own.
        let changed: f64 = if original > 0.5 { 0.125 } else { 0.875 };
        plugin.param_edits().push(id as ParamID, changed);

        // Nothing has reached it yet — this is the state the bug left it in.
        let before = unsafe { plugin.controller.as_com_ref().getParamNormalized(id) };
        assert!(
            (before - original).abs() < 1e-6,
            "the plugin took the edit before it was ever handed one ({before})"
        );

        // One block, with the edits handed over the way a driver must.
        let changes = ParamChanges::default();
        let changes_ptr = ComWrapper::new(changes.clone())
            .to_com_ptr::<IParameterChanges>()
            .expect("parameter changes COM ptr");
        let mut edits: Vec<(ParamID, ParamValue)> = Vec::new();
        plugin.param_edits().drain_into(&mut edits);
        assert!(changes.load(&edits), "the handler's queue lost the edit");

        let io = plugin.io();
        let out_channels: Vec<usize> =
            if io.outputs.is_empty() { vec![2] } else { io.outputs.clone() };
        let in_channels: usize = io.inputs.iter().sum();
        let frames = 64usize;
        let mut scratch =
            crate::plugin::processor::AudioScratch::new(in_channels + out_channels.iter().sum::<usize>(), frames);
        scratch.reset(frames);
        let mut in_buses = crate::vst::realtime::bus_buffers(
            &io.inputs,
            &mut scratch.ptrs_mut()[..in_channels],
        );
        let mut out_buses = crate::vst::realtime::bus_buffers(
            &out_channels,
            &mut scratch.ptrs_mut()[in_channels..],
        );
        unsafe {
            let mut data = ProcessData {
                numInputs: in_buses.len() as i32,
                inputs: if in_buses.is_empty() {
                    std::ptr::null_mut()
                } else {
                    in_buses.as_mut_ptr()
                },
                numOutputs: out_buses.len() as i32,
                outputs: out_buses.as_mut_ptr(),
                numSamples: frames as i32,
                processMode: 0,
                symbolicSampleSize: SymbolicSampleSizes_::kSample32 as i32,
                ..zeroed::<ProcessData>()
            };
            data.inputParameterChanges = changes_ptr.as_ptr() as *mut _;
            let res = plugin.processor.as_com_ref().process(&mut data as *mut _);
            assert_eq!(res, kResultOk, "the plugin refused the block ({res:#X})");
        }

        let after = unsafe { plugin.controller.as_com_ref().getParamNormalized(id) };
        assert!(
            (after - changed).abs() < 1e-6,
            "the editor's edit never reached the plugin: it is at {after}, not {changed}"
        );
    }

    /// A plugin's own state is how the host carries the knobs a user set —
    /// out of an editor's instance and into the one the Sequencer plays, and into
    /// a project file. Anything less and a custom VST3 always plays its defaults.
    ///
    /// This is the general path: it goes through `IComponent::getState`, which
    /// every VST3 has, rather than LeSynth's grid ABI, which only ours does.
    #[test]
    fn a_plugins_own_state_carries_its_parameters_to_another_instance() {
        let path = internal_plugin();
        let Ok(edited) = Vst3Instance::load(&path, Some(&class_ids::FOURIER_SYNTH), None) else {
            println!("no internal plugin at {} — nothing to test", path.display());
            return;
        };
        let Some((id, original)) = first_parameter(&edited) else {
            println!("the plugin exposes no parameters — nothing to test");
            return;
        };

        // Something unmistakably different from where it started.
        let changed = if original > 0.5 { 0.125 } else { 0.875 };
        unsafe {
            edited.controller.as_com_ref().setParamNormalized(id, changed);
        }
        let state = edited.component_state().expect("save state");
        assert!(!state.is_empty(), "the plugin saved an empty state");

        // A second instance starts where the first one did, not where it ended.
        let fresh = Vst3Instance::load(&path, Some(&class_ids::FOURIER_SYNTH), None)
            .expect("load a second instance");
        let (_, before) = first_parameter(&fresh).expect("parameters");
        assert!(
            (before - original).abs() < 1e-6,
            "a fresh instance should start at {original}, not {before}"
        );

        fresh.set_component_state(&state).expect("restore state");
        let (_, after) = first_parameter(&fresh).expect("parameters");
        assert!(
            (after - changed).abs() < 1e-6,
            "the restored instance is at {after}, not the {changed} that was saved"
        );

        // And it can hand the same state on again, which is what a project save
        // after a load has to do.
        let again = fresh.component_state().expect("save state again");
        assert_eq!(again, state, "the state did not survive a round trip");
    }
}
