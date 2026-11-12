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
//! Hosting a **CLAP** plugin.
//!
//! A `.clap` is a shared library (on macOS, a bundle around one) exporting one
//! symbol, `clap_entry`. The entry is initialised once per process with the
//! plugin's own path and deinitialised when the last instance is gone — not
//! once per instance — so libraries are shared through [`ClapLibrary`]. Its
//! factory may hold several plugins, told apart by a reverse-DNS id string,
//! which is what an instrument's `plugin_id` records.
//!
//! A CLAP plugin asks the host for extensions rather than the other way round,
//! and a plugin written against a full DAW asks for a lot: thread checks, a
//! log, parameter and state notifications, port rescans, and — for its GUI —
//! resize requests, timers and (on Linux) file descriptors to watch. The host
//! side of each is here; the ones that report changes this host has no use for
//! (latency, a parameter rescan) are accepted and ignored, which the spec
//! allows.
//!
//! Notes go in as CLAP note events where the plugin's note port takes them,
//! and as raw MIDI where it only speaks MIDI.
//!
//! **Each instance has a main thread of its own.** CLAP is strict about
//! threads, and JUCE-built plugins (Surge XT among them) enforce it: a GUI
//! call from any thread but the one the plugin was created on waits for that
//! thread to run JUCE's message loop — forever, if it is the DAW's GUI thread
//! busy drawing egui. So everything the spec calls `[main-thread]` — creation,
//! activation, state, every GUI call, `on_main_thread` — runs on a
//! [`MainThread`] owned by the instance, and that thread is also the plugin's
//! event loop: it fires the timers and services the file descriptors the
//! plugin registers, and on Windows pumps the messages of the windows the
//! plugin makes there.

use std::cell::Cell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap_sys::audio_buffer::clap_audio_buffer;
use clap_sys::entry::clap_plugin_entry;
use clap_sys::events::{
    clap_event_header, clap_event_midi, clap_event_note, clap_input_events, clap_output_events,
    CLAP_CORE_EVENT_SPACE_ID, CLAP_EVENT_MIDI, CLAP_EVENT_NOTE_OFF, CLAP_EVENT_NOTE_ON,
};
use clap_sys::ext::audio_ports::{
    clap_audio_port_info, clap_host_audio_ports, clap_plugin_audio_ports, CLAP_EXT_AUDIO_PORTS,
};
use clap_sys::ext::gui::{
    clap_host_gui, clap_plugin_gui, clap_window, clap_window_handle, CLAP_EXT_GUI,
    CLAP_WINDOW_API_WIN32, CLAP_WINDOW_API_X11,
};
use clap_sys::ext::latency::{clap_host_latency, CLAP_EXT_LATENCY};
use clap_sys::ext::log::{clap_host_log, clap_log_severity, CLAP_EXT_LOG, CLAP_LOG_ERROR, CLAP_LOG_WARNING};
use clap_sys::ext::note_ports::{
    clap_host_note_ports, clap_note_dialect, clap_note_port_info, clap_plugin_note_ports,
    CLAP_EXT_NOTE_PORTS, CLAP_NOTE_DIALECT_CLAP, CLAP_NOTE_DIALECT_MIDI,
};
use clap_sys::ext::params::{
    clap_host_params, clap_param_clear_flags, clap_param_rescan_flags, CLAP_EXT_PARAMS,
};
use clap_sys::ext::posix_fd_support::{
    clap_host_posix_fd_support, clap_plugin_posix_fd_support, clap_posix_fd_flags,
    CLAP_EXT_POSIX_FD_SUPPORT,
};
#[cfg(unix)]
use clap_sys::ext::posix_fd_support::CLAP_POSIX_FD_READ;
use clap_sys::ext::state::{clap_host_state, clap_plugin_state, CLAP_EXT_STATE};
use clap_sys::ext::tail::{clap_host_tail, CLAP_EXT_TAIL};
use clap_sys::ext::thread_check::{clap_host_thread_check, CLAP_EXT_THREAD_CHECK};
use clap_sys::ext::timer_support::{
    clap_host_timer_support, clap_plugin_timer_support, CLAP_EXT_TIMER_SUPPORT,
};
use clap_sys::factory::plugin_factory::{clap_plugin_factory, CLAP_PLUGIN_FACTORY_ID};
use clap_sys::host::clap_host;
use clap_sys::id::clap_id;
use clap_sys::plugin::{clap_plugin, clap_plugin_descriptor};
use clap_sys::process::clap_process;
use clap_sys::stream::{clap_istream, clap_ostream};
use clap_sys::version::{clap_version_is_compatible, CLAP_VERSION};
use libloading::Library;

use super::processor::{AudioScratch, MidiEvent, RealtimeProcess, MAX_EVENTS_PER_BLOCK};
use super::{display_stem, FoundPlugin, ParentWindow, PluginEditor, PluginFormat, PluginIo};

// ---------------------------------------------------------------------------
// The library and its entry
// ---------------------------------------------------------------------------

/// One `.clap` library, its entry initialised. Shared by every instance made
/// from it: the entry's `init` and `deinit` bracket the library's whole time in
/// the process, not each instance.
struct ClapLibrary {
    entry: *const clap_plugin_entry,
    path: PathBuf,
    // Dropped last, after `deinit` in `Drop` has run.
    _library: Library,
}

unsafe impl Send for ClapLibrary {}
unsafe impl Sync for ClapLibrary {}

/// Libraries already open, so a second instance shares the first one's entry.
static LIBRARIES: Mutex<Vec<(PathBuf, Weak<ClapLibrary>)>> = Mutex::new(Vec::new());

impl ClapLibrary {
    fn open(path: &Path) -> Result<Arc<Self>> {
        let path = path.to_path_buf();
        let mut open = LIBRARIES.lock().unwrap();
        open.retain(|(_, lib)| lib.strong_count() > 0);
        if let Some(lib) = open
            .iter()
            .find(|(p, _)| *p == path)
            .and_then(|(_, lib)| lib.upgrade())
        {
            return Ok(lib);
        }

        let binary = binary_path(&path)?;
        let library = crate::plugin::open_pinned(&binary)
            .with_context(|| format!("Failed to open {}", crate::file_label(&binary)))?;
        let entry = unsafe {
            *library
                .get::<*const clap_plugin_entry>(b"clap_entry\0")
                .with_context(|| {
                    format!("{} is not a CLAP plugin (no clap_entry)", crate::file_label(&path))
                })?
        };
        anyhow::ensure!(!entry.is_null(), "{} has a null clap_entry", crate::file_label(&path));
        let entry_ref = unsafe { &*entry };
        anyhow::ensure!(
            clap_version_is_compatible(entry_ref.clap_version),
            "{} was built for CLAP {}.{}, which this host does not speak",
            crate::file_label(&path),
            entry_ref.clap_version.major,
            entry_ref.clap_version.minor
        );
        // `init` is handed the path the plugin was found at — the bundle on
        // macOS — which is how it finds its own resources.
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .context("plugin path contains a NUL byte")?;
        let init = entry_ref.init.context("clap_entry has no init")?;
        anyhow::ensure!(
            unsafe { init(c_path.as_ptr()) },
            "{} refused to initialise",
            crate::file_label(&path)
        );

        let lib = Arc::new(ClapLibrary { entry, path: path.clone(), _library: library });
        open.push((path, Arc::downgrade(&lib)));
        Ok(lib)
    }

    fn factory(&self) -> Result<&clap_plugin_factory> {
        unsafe {
            let get = (*self.entry).get_factory.context("clap_entry has no get_factory")?;
            let factory = get(CLAP_PLUGIN_FACTORY_ID.as_ptr()) as *const clap_plugin_factory;
            factory
                .as_ref()
                .with_context(|| format!("{} has no plugin factory", crate::file_label(&self.path)))
        }
    }

    /// Every plugin descriptor in the factory.
    fn descriptors(&self) -> Result<Vec<&clap_plugin_descriptor>> {
        let factory = self.factory()?;
        let count = factory.get_plugin_count.context("factory has no get_plugin_count")?;
        let get = factory
            .get_plugin_descriptor
            .context("factory has no get_plugin_descriptor")?;
        let mut out = Vec::new();
        unsafe {
            for i in 0..count(factory) {
                if let Some(d) = get(factory, i).as_ref() {
                    out.push(d);
                }
            }
        }
        Ok(out)
    }
}

impl Drop for ClapLibrary {
    fn drop(&mut self) {
        unsafe {
            if let Some(deinit) = (*self.entry).deinit {
                deinit();
            }
        }
    }
}

/// The shared library inside whatever the user picked: the `.clap` itself on
/// Linux and Windows, the executable inside the bundle on macOS.
fn binary_path(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let macos = path.join("Contents").join("MacOS");
    if macos.is_dir() {
        let stem = path.file_stem().map(|s| s.to_os_string());
        let mut files: Vec<PathBuf> = std::fs::read_dir(&macos)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        if let Some(found) = files
            .iter()
            .find(|p| p.file_stem().map(|s| s.to_os_string()) == stem)
            .or_else(|| files.first())
        {
            return Ok(found.clone());
        }
    }
    bail!("{} is not a CLAP plugin file or bundle", crate::file_label(path))
}

/// A C string from the plugin, or empty.
unsafe fn c_str(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// A descriptor's feature list.
unsafe fn features(desc: &clap_plugin_descriptor) -> Vec<String> {
    let mut out = Vec::new();
    let mut p = desc.features;
    if p.is_null() {
        return out;
    }
    while !(*p).is_null() {
        out.push(c_str(*p));
        p = p.add(1);
    }
    out
}

/// The descriptor `plugin_id` names, or — with none — the first instrument,
/// else the first effect, else the first plugin there is.
fn pick<'a>(
    descriptors: &[&'a clap_plugin_descriptor],
    plugin_id: Option<&str>,
    path: &Path,
) -> Result<&'a clap_plugin_descriptor> {
    if let Some(id) = plugin_id {
        return descriptors
            .iter()
            .copied()
            .find(|d| unsafe { c_str(d.id) } == id)
            .with_context(|| format!("{} holds no plugin '{id}'", crate::file_label(path)));
    }
    let has = |d: &clap_plugin_descriptor, f: &str| unsafe { features(d).iter().any(|x| x == f) };
    descriptors
        .iter()
        .copied()
        .find(|d| has(d, "instrument"))
        .or_else(|| descriptors.iter().copied().find(|d| has(d, "audio-effect")))
        .or_else(|| descriptors.first().copied())
        .with_context(|| format!("{} holds no plugins", crate::file_label(path)))
}

/// Check that `path` is a CLAP file holding `plugin_id` (or any plugin).
pub fn validate(path: &Path, plugin_id: Option<&str>) -> Result<()> {
    let lib = ClapLibrary::open(path)?;
    let descriptors = lib.descriptors()?;
    pick(&descriptors, plugin_id, path)?;
    Ok(())
}

/// Every plugin in the CLAP file at `path`, for the picker. A file that will
/// not open is listed by its name alone, so the error surfaces when it is
/// picked rather than the plugin silently missing from the list.
pub fn list_plugins(path: &Path) -> Vec<FoundPlugin> {
    let listed = ClapLibrary::open(path).and_then(|lib| {
        Ok(lib
            .descriptors()?
            .iter()
            .map(|d| unsafe { (c_str(d.name), c_str(d.id)) })
            .collect::<Vec<_>>())
    });
    match listed {
        Ok(plugins) if !plugins.is_empty() => plugins
            .into_iter()
            .map(|(name, id)| FoundPlugin {
                name: if name.is_empty() { display_stem(path) } else { name },
                format: PluginFormat::Clap,
                path: path.to_path_buf(),
                plugin_id: Some(id),
            })
            .collect(),
        Ok(_) => Vec::new(),
        Err(e) => {
            log::debug!("CLAP scan: {} — {e:#}", path.display());
            vec![FoundPlugin {
                name: display_stem(path),
                format: PluginFormat::Clap,
                path: path.to_path_buf(),
                plugin_id: None,
            }]
        }
    }
}

/// A plugin's name and its features, `|`-separated like a VST3's
/// subcategories, so a "drum" feature reads as one.
pub fn describe(path: &Path, plugin_id: Option<&str>) -> Option<(String, String)> {
    let lib = ClapLibrary::open(path).ok()?;
    let descriptors = lib.descriptors().ok()?;
    let desc = pick(&descriptors, plugin_id, path).ok()?;
    unsafe { Some((c_str(desc.name), features(desc).join("|"))) }
}

// ---------------------------------------------------------------------------
// The host side
// ---------------------------------------------------------------------------

thread_local! {
    /// Set on the thread inside `process()`, so the plugin's thread checks get
    /// a true answer.
    static IN_AUDIO_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// A timer a plugin's GUI registered.
struct Timer {
    id: clap_id,
    interval: Duration,
    next: Instant,
}

/// What the plugin's calls into the host leave behind, for the editor loop and
/// the instance to act on. Lives behind `clap_host::host_data`.
#[derive(Default)]
struct HostState {
    /// The instance's [`MainThread`], for the plugin's thread checks.
    main_thread: std::sync::OnceLock<ThreadId>,
    /// The plugin wants `on_main_thread` called.
    callback_requested: AtomicBool,
    resize_request: Mutex<Option<(u32, u32)>>,
    timers: Mutex<Vec<Timer>>,
    next_timer_id: AtomicU32,
    fds: Mutex<Vec<(i32, clap_posix_fd_flags)>>,
}

/// The `clap_host` handed to the plugin and the state its callbacks write to.
/// Boxed so both have an address that never moves; the plugin keeps the
/// pointer for its whole life.
struct Host {
    host: clap_host,
    state: HostState,
}

const HOST_NAME: &CStr = c"ReSynth Studio";
const HOST_VENDOR: &CStr = c"Jakub Hlavnicka";
const HOST_URL: &CStr = c"https://github.com/hlavnjak";
const HOST_VERSION: &CStr =
    match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(v) => v,
        Err(_) => panic!("the package version has a NUL in it"),
    };

impl Host {
    fn new() -> Box<Host> {
        let mut host = Box::new(Host {
            host: clap_host {
                clap_version: CLAP_VERSION,
                host_data: std::ptr::null_mut(),
                name: HOST_NAME.as_ptr(),
                vendor: HOST_VENDOR.as_ptr(),
                url: HOST_URL.as_ptr(),
                version: HOST_VERSION.as_ptr(),
                get_extension: Some(host_get_extension),
                request_restart: Some(host_request_restart),
                request_process: Some(host_request_process),
                request_callback: Some(host_request_callback),
            },
            state: HostState::default(),
        });
        host.host.host_data = &host.state as *const HostState as *mut c_void;
        host
    }
}

unsafe fn state<'a>(host: *const clap_host) -> Option<&'a HostState> {
    host.as_ref()
        .and_then(|h| (h.host_data as *const HostState).as_ref())
}

unsafe extern "C" fn host_get_extension(_host: *const clap_host, id: *const c_char) -> *const c_void {
    if id.is_null() {
        return std::ptr::null();
    }
    let id = CStr::from_ptr(id);
    let ext: *const c_void = if id == CLAP_EXT_THREAD_CHECK {
        &HOST_THREAD_CHECK as *const _ as *const c_void
    } else if id == CLAP_EXT_LOG {
        &HOST_LOG as *const _ as *const c_void
    } else if id == CLAP_EXT_PARAMS {
        &HOST_PARAMS as *const _ as *const c_void
    } else if id == CLAP_EXT_STATE {
        &HOST_STATE as *const _ as *const c_void
    } else if id == CLAP_EXT_AUDIO_PORTS {
        &HOST_AUDIO_PORTS as *const _ as *const c_void
    } else if id == CLAP_EXT_NOTE_PORTS {
        &HOST_NOTE_PORTS as *const _ as *const c_void
    } else if id == CLAP_EXT_LATENCY {
        &HOST_LATENCY as *const _ as *const c_void
    } else if id == CLAP_EXT_TAIL {
        &HOST_TAIL as *const _ as *const c_void
    } else if id == CLAP_EXT_GUI {
        &HOST_GUI as *const _ as *const c_void
    } else if id == CLAP_EXT_TIMER_SUPPORT {
        &HOST_TIMER_SUPPORT as *const _ as *const c_void
    } else if id == CLAP_EXT_POSIX_FD_SUPPORT && cfg!(unix) {
        &HOST_POSIX_FD as *const _ as *const c_void
    } else {
        std::ptr::null()
    };
    ext
}

unsafe extern "C" fn host_request_restart(_host: *const clap_host) {
    // Restarting means deactivating and reactivating; an instrument that needs it
    // gets it the next time its editor or the transport starts.
    log::debug!("CLAP plugin asked for a restart");
}

unsafe extern "C" fn host_request_process(_host: *const clap_host) {
    // Every instance this host runs is processed continuously anyway.
}

unsafe extern "C" fn host_request_callback(host: *const clap_host) {
    if let Some(s) = state(host) {
        s.callback_requested.store(true, Ordering::Release);
    }
}

static HOST_THREAD_CHECK: clap_host_thread_check = clap_host_thread_check {
    is_main_thread: Some(thread_is_main),
    is_audio_thread: Some(thread_is_audio),
};

unsafe extern "C" fn thread_is_main(host: *const clap_host) -> bool {
    match state(host).and_then(|s| s.main_thread.get()) {
        Some(id) => *id == std::thread::current().id(),
        None => !IN_AUDIO_THREAD.with(Cell::get),
    }
}

unsafe extern "C" fn thread_is_audio(_host: *const clap_host) -> bool {
    IN_AUDIO_THREAD.with(Cell::get)
}

static HOST_LOG: clap_host_log = clap_host_log { log: Some(host_log) };

unsafe extern "C" fn host_log(_host: *const clap_host, severity: clap_log_severity, msg: *const c_char) {
    let msg = c_str(msg);
    match severity {
        s if s >= CLAP_LOG_ERROR => log::warn!("CLAP plugin: {msg}"),
        CLAP_LOG_WARNING => log::info!("CLAP plugin: {msg}"),
        _ => log::debug!("CLAP plugin: {msg}"),
    }
}

static HOST_PARAMS: clap_host_params = clap_host_params {
    rescan: Some(params_rescan),
    clear: Some(params_clear),
    request_flush: Some(params_request_flush),
};

unsafe extern "C" fn params_rescan(_host: *const clap_host, _flags: clap_param_rescan_flags) {}
unsafe extern "C" fn params_clear(_host: *const clap_host, _id: clap_id, _flags: clap_param_clear_flags) {}
unsafe extern "C" fn params_request_flush(_host: *const clap_host) {
    // Every instance is processing whenever anything can change it, and
    // `process()` is where a flush happens anyway.
}

static HOST_STATE: clap_host_state = clap_host_state { mark_dirty: Some(state_mark_dirty) };

unsafe extern "C" fn state_mark_dirty(_host: *const clap_host) {
    // The state is read whenever the editor closes and whenever a project is
    // saved, dirty or not.
}

static HOST_AUDIO_PORTS: clap_host_audio_ports = clap_host_audio_ports {
    is_rescan_flag_supported: Some(audio_ports_flag_supported),
    rescan: Some(audio_ports_rescan),
};

unsafe extern "C" fn audio_ports_flag_supported(_host: *const clap_host, _flag: u32) -> bool {
    false
}
unsafe extern "C" fn audio_ports_rescan(_host: *const clap_host, _flags: u32) {}

static HOST_NOTE_PORTS: clap_host_note_ports = clap_host_note_ports {
    supported_dialects: Some(note_ports_dialects),
    rescan: Some(note_ports_rescan),
};

unsafe extern "C" fn note_ports_dialects(_host: *const clap_host) -> clap_note_dialect {
    CLAP_NOTE_DIALECT_CLAP | CLAP_NOTE_DIALECT_MIDI
}
unsafe extern "C" fn note_ports_rescan(_host: *const clap_host, _flags: u32) {}

static HOST_LATENCY: clap_host_latency = clap_host_latency { changed: Some(latency_changed) };
unsafe extern "C" fn latency_changed(_host: *const clap_host) {}

static HOST_TAIL: clap_host_tail = clap_host_tail { changed: Some(tail_changed) };
unsafe extern "C" fn tail_changed(_host: *const clap_host) {}

static HOST_GUI: clap_host_gui = clap_host_gui {
    resize_hints_changed: Some(gui_resize_hints_changed),
    request_resize: Some(gui_request_resize),
    request_show: Some(gui_request_show),
    request_hide: Some(gui_request_hide),
    closed: Some(gui_closed),
};

unsafe extern "C" fn gui_resize_hints_changed(_host: *const clap_host) {}

unsafe extern "C" fn gui_request_resize(host: *const clap_host, width: u32, height: u32) -> bool {
    match state(host) {
        Some(s) => {
            *s.resize_request.lock().unwrap() = Some((width, height));
            true
        }
        None => false,
    }
}

unsafe extern "C" fn gui_request_show(_host: *const clap_host) -> bool {
    true
}

unsafe extern "C" fn gui_request_hide(_host: *const clap_host) -> bool {
    false
}

unsafe extern "C" fn gui_closed(_host: *const clap_host, _was_destroyed: bool) {
    // Only a floating window can be closed by the plugin, and this host only
    // ever embeds.
}

static HOST_TIMER_SUPPORT: clap_host_timer_support = clap_host_timer_support {
    register_timer: Some(timer_register),
    unregister_timer: Some(timer_unregister),
};

unsafe extern "C" fn timer_register(host: *const clap_host, period_ms: u32, timer_id: *mut clap_id) -> bool {
    let Some(s) = state(host) else { return false };
    if timer_id.is_null() {
        return false;
    }
    let id = s.next_timer_id.fetch_add(1, Ordering::Relaxed);
    // A zero period means "as often as you can"; clamp it so one plugin cannot
    // spin the editor thread.
    let interval = Duration::from_millis(period_ms.max(1) as u64);
    s.timers.lock().unwrap().push(Timer { id, interval, next: Instant::now() + interval });
    *timer_id = id;
    true
}

unsafe extern "C" fn timer_unregister(host: *const clap_host, timer_id: clap_id) -> bool {
    let Some(s) = state(host) else { return false };
    let mut timers = s.timers.lock().unwrap();
    let before = timers.len();
    timers.retain(|t| t.id != timer_id);
    timers.len() != before
}

static HOST_POSIX_FD: clap_host_posix_fd_support = clap_host_posix_fd_support {
    register_fd: Some(fd_register),
    modify_fd: Some(fd_modify),
    unregister_fd: Some(fd_unregister),
};

unsafe extern "C" fn fd_register(host: *const clap_host, fd: i32, flags: clap_posix_fd_flags) -> bool {
    let Some(s) = state(host) else { return false };
    let mut fds = s.fds.lock().unwrap();
    fds.retain(|(f, _)| *f != fd);
    fds.push((fd, flags));
    true
}

unsafe extern "C" fn fd_modify(host: *const clap_host, fd: i32, flags: clap_posix_fd_flags) -> bool {
    fd_register(host, fd, flags)
}

unsafe extern "C" fn fd_unregister(host: *const clap_host, fd: i32) -> bool {
    let Some(s) = state(host) else { return false };
    s.fds.lock().unwrap().retain(|(f, _)| *f != fd);
    true
}

// ---------------------------------------------------------------------------
// The main thread
// ---------------------------------------------------------------------------

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A raw pointer that may cross to the main thread. Read it with
/// [`Ptr::get`], so a closure captures the wrapper rather than the pointer.
struct Ptr<T>(*const T);

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Ptr<T> {}

unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

impl<T> Ptr<T> {
    fn get(self) -> *const T {
        self.0
    }
}

/// What the main thread services between jobs, once the plugin exists.
#[derive(Clone, Copy)]
struct Targets {
    plugin: Ptr<clap_plugin>,
    host: Ptr<HostState>,
    timer: Ptr<clap_plugin_timer_support>,
    fd: Ptr<clap_plugin_posix_fd_support>,
}

/// One CLAP instance's main thread: it runs the jobs handed to it with
/// [`MainThread::run`] and, between them, is the plugin's event loop.
struct MainThread {
    jobs: Mutex<Option<SyncSender<Job>>>,
    id: ThreadId,
    handle: Mutex<Option<JoinHandle<()>>>,
    targets: Arc<Mutex<Option<Targets>>>,
}

impl MainThread {
    fn spawn(name: &str) -> Result<Arc<Self>> {
        let (tx, rx) = mpsc::sync_channel::<Job>(16);
        let targets: Arc<Mutex<Option<Targets>>> = Arc::default();
        let loop_targets = targets.clone();
        let handle = std::thread::Builder::new()
            .name(format!("clap: {name}"))
            .spawn(move || main_loop(rx, loop_targets))
            .context("could not start the plugin's main thread")?;
        Ok(Arc::new(MainThread {
            jobs: Mutex::new(Some(tx)),
            id: handle.thread().id(),
            handle: Mutex::new(Some(handle)),
            targets,
        }))
    }

    /// Run `f` on the main thread and wait for its result — straight away if
    /// this already is the main thread.
    fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        if std::thread::current().id() == self.id {
            return f();
        }
        let (tx, rx) = mpsc::sync_channel::<R>(1);
        let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
            let _ = tx.send(f());
        });
        // SAFETY: this function does not return until the job has run (its
        // result arrives) or has been dropped unrun (the channel closes), so
        // nothing the job borrows can go away while the job still exists.
        let job: Job = unsafe { std::mem::transmute(job) };
        let sender = self.jobs.lock().unwrap().clone().expect("the CLAP main thread has stopped");
        sender.send(job).expect("the CLAP main thread has stopped");
        // On Windows the waiting thread may own the window the plugin is
        // creating a child of, and creating one *sends* its parent a message
        // — which waits for this thread to take it. Blocking here would be a
        // deadlock (a black editor, forever), so the wait keeps taking the
        // messages sent to this thread's windows.
        #[cfg(target_os = "windows")]
        loop {
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok(r) => return r,
                Err(RecvTimeoutError::Timeout) => unsafe { deliver_sent_messages() },
                Err(RecvTimeoutError::Disconnected) => panic!("the CLAP main thread stopped mid-job"),
            }
        }
        #[cfg(not(target_os = "windows"))]
        rx.recv().expect("the CLAP main thread stopped mid-job")
    }

    /// Stop the loop and wait for the thread.
    fn stop(&self) {
        self.jobs.lock().unwrap().take();
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

fn main_loop(rx: Receiver<Job>, targets: Arc<Mutex<Option<Targets>>>) {
    loop {
        let current = *targets.lock().unwrap();
        // Sleep until the next job, the nearest timer, or a short poll for the
        // plugin's descriptors — whichever is first.
        let mut wait = Duration::from_millis(20);
        if let Some(t) = current {
            let host = unsafe { &*t.host.get() };
            let now = Instant::now();
            if let Some(next) = host.timers.lock().unwrap().iter().map(|t| t.next).min() {
                wait = wait.min(next.saturating_duration_since(now));
            }
            if !host.fds.lock().unwrap().is_empty() {
                wait = wait.min(Duration::from_millis(5));
            }
            if host.callback_requested.load(Ordering::Acquire) {
                wait = Duration::ZERO;
            }
        }
        match rx.recv_timeout(wait) {
            Ok(job) => job(),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
        while let Ok(job) = rx.try_recv() {
            job();
        }
        if let Some(t) = *targets.lock().unwrap() {
            unsafe { service(t) };
        }
        #[cfg(target_os = "windows")]
        unsafe {
            pump_messages();
        }
    }
}

/// The plugin's event-loop work: ready descriptors, due timers, and an
/// `on_main_thread` it asked for.
unsafe fn service(t: Targets) {
    let plugin = t.plugin.get();
    let host = &*t.host.get();
    #[cfg(unix)]
    if let Some(on_fd) = t.fd.get().as_ref().and_then(|f| f.on_fd) {
        let fds: Vec<(i32, clap_posix_fd_flags)> = host.fds.lock().unwrap().clone();
        if !fds.is_empty() {
            let mut polls: Vec<libc::pollfd> = fds
                .iter()
                .map(|(fd, _)| libc::pollfd { fd: *fd, events: libc::POLLIN, revents: 0 })
                .collect();
            libc::poll(polls.as_mut_ptr(), polls.len() as libc::nfds_t, 0);
            for p in polls.iter().filter(|p| p.revents != 0) {
                on_fd(plugin, p.fd, CLAP_POSIX_FD_READ);
            }
        }
    }
    #[cfg(not(unix))]
    let _ = t.fd;
    if let Some(on_timer) = t.timer.get().as_ref().and_then(|t| t.on_timer) {
        let now = Instant::now();
        let due: Vec<clap_id> = host
            .timers
            .lock()
            .unwrap()
            .iter_mut()
            .filter(|t| t.next <= now)
            .map(|t| {
                t.next = now + t.interval;
                t.id
            })
            .collect();
        for id in due {
            // Still registered? A previous callback may have dropped it.
            if host.timers.lock().unwrap().iter().any(|t| t.id == id) {
                on_timer(plugin, id);
            }
        }
    }
    if host.callback_requested.swap(false, Ordering::AcqRel) {
        if let Some(cb) = (*plugin).on_main_thread {
            cb(plugin);
        }
    }
}

/// Deliver the messages other threads have *sent* to this thread's windows,
/// leaving everything posted to its queue for its own loop. `PeekMessage`
/// delivers sent messages whatever it is asked to look for.
#[cfg(target_os = "windows")]
unsafe fn deliver_sent_messages() {
    use winapi::um::winuser::{PeekMessageW, MSG, PM_NOREMOVE};
    let mut msg: MSG = std::mem::zeroed();
    PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_NOREMOVE);
}

/// Dispatch the messages of every window this thread owns — the plugin's
/// editor, on Windows, lives on this thread.
#[cfg(target_os = "windows")]
unsafe fn pump_messages() {
    use winapi::um::winuser::{DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE};
    let mut msg: MSG = std::mem::zeroed();
    while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) > 0 {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

// ---------------------------------------------------------------------------
// The instance
// ---------------------------------------------------------------------------

/// A loaded CLAP plugin instance.
pub struct ClapInstance {
    plugin: *const clap_plugin,
    /// The host the plugin was created with. It holds the pointer for its
    /// whole life, so this is dropped only after `destroy`.
    host: Box<Host>,
    name: String,
    io: RwLock<PluginIo>,
    active: AtomicBool,
    /// `start_processing` has been called (on the audio thread) and
    /// `stop_processing` not yet.
    processing: Arc<AtomicBool>,
    /// How the first note port wants its notes; `None` for an effect with no
    /// note input at all.
    note_dialect: Option<clap_note_dialect>,
    /// Where every `[main-thread]` call is made. Stopped in `Drop`, after the
    /// plugin is destroyed.
    main: Arc<MainThread>,
    /// Holds the library — and its entry's `init` — for as long as the plugin
    /// lives. Declared last, so it is dropped last.
    library: Arc<ClapLibrary>,
}

// The plugin is driven from the GUI thread, the editor's thread and the audio
// thread, the way the CLAP threading model lays out; the pointers themselves
// are only ever dereferenced while the instance is alive.
unsafe impl Send for ClapInstance {}
unsafe impl Sync for ClapInstance {}

impl ClapInstance {
    /// Load and initialise the plugin `plugin_id` (or the first instrument) in
    /// the CLAP file at `path`.
    pub fn load(path: &Path, plugin_id: Option<&str>) -> Result<Self> {
        let library = ClapLibrary::open(path)?;
        let descriptors = library.descriptors()?;
        let desc = pick(&descriptors, plugin_id, path)?;
        let (name, id) = unsafe { (c_str(desc.name), c_str(desc.id)) };
        let c_id = CString::new(id.clone()).context("plugin id contains a NUL byte")?;

        let host = Host::new();
        let main = MainThread::spawn(&name)?;
        let _ = host.state.main_thread.set(main.id);
        let factory = Ptr(library.factory()? as *const clap_plugin_factory);
        let host_ptr = Ptr(&host.host as *const clap_host);
        let created = main.run(|| unsafe {
            let factory = &*factory.get();
            let create = factory.create_plugin?;
            let plugin = create(factory, host_ptr.get(), c_id.as_ptr());
            if plugin.is_null() {
                return None;
            }
            if !(*plugin).init.is_some_and(|init| init(plugin)) {
                if let Some(destroy) = (*plugin).destroy {
                    destroy(plugin);
                }
                return None;
            }
            Some(Ptr(plugin))
        });
        let Some(plugin) = created.map(Ptr::get) else {
            main.stop();
            bail!("'{name}' could not be created and initialised");
        };

        let mut instance = ClapInstance {
            plugin,
            host,
            name,
            io: RwLock::new(PluginIo::default()),
            active: AtomicBool::new(false),
            processing: Arc::new(AtomicBool::new(false)),
            note_dialect: None,
            main,
            library,
        };
        instance.note_dialect = instance.main.run(|| instance.input_note_dialect());
        *instance.main.targets.lock().unwrap() = Some(Targets {
            plugin: Ptr(instance.plugin),
            host: Ptr(&instance.host.state as *const HostState),
            timer: Ptr(instance
                .extension::<clap_plugin_timer_support>(CLAP_EXT_TIMER_SUPPORT)
                .map_or(std::ptr::null(), |t| t as *const _)),
            fd: Ptr(instance
                .extension::<clap_plugin_posix_fd_support>(CLAP_EXT_POSIX_FD_SUPPORT)
                .map_or(std::ptr::null(), |t| t as *const _)),
        });
        log::info!(
            "Loaded CLAP '{}' ({id}) from {}",
            instance.name,
            instance.library.path.display()
        );
        Ok(instance)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn io(&self) -> PluginIo {
        self.io.read().unwrap().clone()
    }

    /// One of the plugin's extensions, if it has it.
    fn extension<T>(&self, id: &CStr) -> Option<&T> {
        unsafe {
            let get = (*self.plugin).get_extension?;
            (get(self.plugin, id.as_ptr()) as *const T).as_ref()
        }
    }

    /// The dialect to send notes in: CLAP note events where the first input
    /// note port takes them, MIDI where it only takes that.
    fn input_note_dialect(&self) -> Option<clap_note_dialect> {
        let ports = self.extension::<clap_plugin_note_ports>(CLAP_EXT_NOTE_PORTS)?;
        unsafe {
            let count = ports.count?(self.plugin, true);
            if count == 0 {
                return None;
            }
            let mut info: clap_note_port_info = std::mem::zeroed();
            if !ports.get?(self.plugin, 0, true, &mut info) {
                return Some(CLAP_NOTE_DIALECT_CLAP);
            }
            if info.supported_dialects & CLAP_NOTE_DIALECT_CLAP != 0 {
                Some(CLAP_NOTE_DIALECT_CLAP)
            } else if info.supported_dialects & CLAP_NOTE_DIALECT_MIDI != 0 {
                Some(CLAP_NOTE_DIALECT_MIDI)
            } else {
                None
            }
        }
    }

    /// Channel counts of the plugin's audio ports in one direction.
    fn audio_ports(&self, is_input: bool) -> Vec<usize> {
        let Some(ports) = self.extension::<clap_plugin_audio_ports>(CLAP_EXT_AUDIO_PORTS) else {
            return Vec::new();
        };
        let (Some(count), Some(get)) = (ports.count, ports.get) else {
            return Vec::new();
        };
        unsafe {
            (0..count(self.plugin, is_input))
                .map(|i| {
                    let mut info: clap_audio_port_info = std::mem::zeroed();
                    if get(self.plugin, i, is_input, &mut info) {
                        info.channel_count as usize
                    } else {
                        0
                    }
                })
                .collect()
        }
    }

    pub fn initialize_audio(&self, sample_rate: f64, max_block_size: i32) -> Result<()> {
        self.main.run(|| self.activate(sample_rate, max_block_size))
    }

    fn activate(&self, sample_rate: f64, max_block_size: i32) -> Result<()> {
        if self.active.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let io = PluginIo {
            inputs: self.audio_ports(true),
            outputs: self.audio_ports(false),
            max_block: max_block_size.max(1) as usize,
        };
        let activate = unsafe { (*self.plugin).activate }.context("plugin has no activate")?;
        let ok = unsafe { activate(self.plugin, sample_rate, 1, io.max_block as u32) };
        if !ok {
            self.active.store(false, Ordering::SeqCst);
            bail!("'{}' refused to activate at {sample_rate} Hz", self.name);
        }
        log::info!(
            "CLAP '{}' activated: sr={sample_rate}, block={}, in={:?}, out={:?}",
            self.name,
            io.max_block,
            io.inputs,
            io.outputs
        );
        *self.io.write().unwrap() = io;
        Ok(())
    }

    pub fn save_state(&self) -> Result<Vec<u8>> {
        self.main.run(|| self.save_state_here())
    }

    fn save_state_here(&self) -> Result<Vec<u8>> {
        let state = self
            .extension::<clap_plugin_state>(CLAP_EXT_STATE)
            .with_context(|| format!("'{}' has no state to save", self.name))?;
        let save = state.save.context("state extension has no save")?;
        let mut bytes: Vec<u8> = Vec::new();
        let stream = clap_ostream {
            ctx: &mut bytes as *mut Vec<u8> as *mut c_void,
            write: Some(ostream_write),
        };
        anyhow::ensure!(
            unsafe { save(self.plugin, &stream) },
            "'{}' would not save its state",
            self.name
        );
        Ok(bytes)
    }

    pub fn restore_state(&self, bytes: &[u8]) -> Result<()> {
        self.main.run(|| self.restore_state_here(bytes))
    }

    fn restore_state_here(&self, bytes: &[u8]) -> Result<()> {
        let state = self
            .extension::<clap_plugin_state>(CLAP_EXT_STATE)
            .with_context(|| format!("'{}' has no state to load", self.name))?;
        let load = state.load.context("state extension has no load")?;
        let mut reader = Reader { bytes, pos: 0 };
        let stream = clap_istream {
            ctx: &mut reader as *mut Reader as *mut c_void,
            read: Some(istream_read),
        };
        anyhow::ensure!(
            unsafe { load(self.plugin, &stream) },
            "'{}' would not load the saved state",
            self.name
        );
        Ok(())
    }

    /// The window API this platform embeds editors with.
    fn window_api() -> &'static CStr {
        if cfg!(target_os = "windows") {
            CLAP_WINDOW_API_WIN32
        } else {
            CLAP_WINDOW_API_X11
        }
    }

    pub fn has_editor(&self) -> bool {
        self.main.run(|| {
            self.extension::<clap_plugin_gui>(CLAP_EXT_GUI)
                .and_then(|gui| gui.is_api_supported)
                .is_some_and(|supported| unsafe {
                    supported(self.plugin, Self::window_api().as_ptr(), false)
                })
        })
    }

    pub fn create_editor(&self) -> Result<Box<dyn PluginEditor>> {
        let gui = self
            .extension::<clap_plugin_gui>(CLAP_EXT_GUI)
            .with_context(|| format!("'{}' has no editor", self.name))?;
        anyhow::ensure!(
            self.has_editor(),
            "'{}' has no editor that embeds in a {} window",
            self.name,
            Self::window_api().to_string_lossy()
        );
        Ok(Box::new(ClapEditor {
            plugin: self.plugin,
            gui: gui as *const clap_plugin_gui,
            host: &self.host.state as *const HostState,
            main: self.main.clone(),
            created: false,
        }))
    }

    pub(crate) fn realtime(
        &self,
        scratch: &mut AudioScratch,
        in_buses: &[usize],
        out_buses: &[usize],
    ) -> Result<Box<dyn RealtimeProcess>> {
        let in_channels: usize = in_buses.iter().sum();
        let ptrs = scratch.ptrs_mut();
        let lay_out = |buses: &[usize], ptrs: &mut [*mut f32]| -> Vec<clap_audio_buffer> {
            let mut offset = 0;
            buses
                .iter()
                .map(|&n| {
                    let buffer = clap_audio_buffer {
                        data32: unsafe { ptrs.as_mut_ptr().add(offset) },
                        data64: std::ptr::null_mut(),
                        channel_count: n as u32,
                        latency: 0,
                        constant_mask: 0,
                    };
                    offset += n;
                    buffer
                })
                .collect()
        };
        let inputs = lay_out(in_buses, &mut ptrs[..in_channels]);
        let outputs = lay_out(out_buses, &mut ptrs[in_channels..]);
        // A plugin that declared no output port at all was given a stereo bus
        // by the processor; it has nowhere to write, so it is handed none.
        let declared_outputs = self.io.read().unwrap().outputs.len();
        let mut events = Box::new(EventBuffer {
            slots: vec![EventSlot::empty(); MAX_EVENTS_PER_BLOCK],
            len: 0,
        });
        let in_events = clap_input_events {
            ctx: &mut *events as *mut EventBuffer as *mut c_void,
            size: Some(events_size),
            get: Some(events_get),
        };
        Ok(Box::new(ClapRealtime {
            plugin: self.plugin,
            processing: self.processing.clone(),
            inputs,
            outputs,
            declared_outputs,
            events,
            in_events,
            out_events: clap_output_events {
                ctx: std::ptr::null_mut(),
                try_push: Some(out_events_push),
            },
            note_dialect: self.note_dialect,
            steady_time: 0,
        }))
    }
}

impl Drop for ClapInstance {
    fn drop(&mut self) {
        // Nothing is processing it any more — every stream that could is gone
        // before the last reference to the instance — so the audio thread's
        // half of the shutdown happens here.
        // The plugin's thread check is answered as the audio thread for the
        // length of the call, which is the role this thread is playing.
        if self.processing.swap(false, Ordering::SeqCst) {
            IN_AUDIO_THREAD.with(|f| f.set(true));
            unsafe {
                if let Some(stop) = (*self.plugin).stop_processing {
                    stop(self.plugin);
                }
            }
            IN_AUDIO_THREAD.with(|f| f.set(false));
        }
        // The loop must not service a plugin that is being destroyed.
        self.main.targets.lock().unwrap().take();
        let plugin = Ptr(self.plugin);
        let active = &self.active;
        self.main.run(|| unsafe {
            let plugin = plugin.get();
            if active.swap(false, Ordering::SeqCst) {
                if let Some(deactivate) = (*plugin).deactivate {
                    deactivate(plugin);
                }
            }
            if let Some(destroy) = (*plugin).destroy {
                destroy(plugin);
            }
        });
        self.main.stop();
    }
}

/// Appends whatever the plugin writes to a `Vec<u8>`.
unsafe extern "C" fn ostream_write(stream: *const clap_ostream, buffer: *const c_void, size: u64) -> i64 {
    let Some(stream) = stream.as_ref() else { return -1 };
    let Some(bytes) = (stream.ctx as *mut Vec<u8>).as_mut() else { return -1 };
    if buffer.is_null() {
        return -1;
    }
    bytes.extend_from_slice(std::slice::from_raw_parts(buffer as *const u8, size as usize));
    size as i64
}

/// A saved state being read back.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

unsafe extern "C" fn istream_read(stream: *const clap_istream, buffer: *mut c_void, size: u64) -> i64 {
    let Some(stream) = stream.as_ref() else { return -1 };
    let Some(reader) = (stream.ctx as *mut Reader).as_mut() else { return -1 };
    if buffer.is_null() {
        return -1;
    }
    let n = (size as usize).min(reader.bytes.len() - reader.pos);
    std::ptr::copy_nonoverlapping(reader.bytes.as_ptr().add(reader.pos), buffer as *mut u8, n);
    reader.pos += n;
    n as i64
}

// ---------------------------------------------------------------------------
// Processing
// ---------------------------------------------------------------------------

/// Room for either kind of note event.
#[repr(C)]
#[derive(Clone, Copy)]
union EventSlot {
    note: clap_event_note,
    midi: clap_event_midi,
}

impl EventSlot {
    fn empty() -> Self {
        // Plain C data; all-zero is a valid (if meaningless) value of both.
        unsafe { std::mem::zeroed() }
    }
}

/// The block's input events, read by the plugin through `clap_input_events`.
struct EventBuffer {
    slots: Vec<EventSlot>,
    len: usize,
}

unsafe extern "C" fn events_size(list: *const clap_input_events) -> u32 {
    list.as_ref()
        .and_then(|l| (l.ctx as *const EventBuffer).as_ref())
        .map_or(0, |b| b.len as u32)
}

unsafe extern "C" fn events_get(list: *const clap_input_events, index: u32) -> *const clap_event_header {
    let Some(buffer) = list.as_ref().and_then(|l| (l.ctx as *const EventBuffer).as_ref()) else {
        return std::ptr::null();
    };
    match buffer.slots.get(index as usize) {
        Some(slot) if (index as usize) < buffer.len => slot as *const EventSlot as *const clap_event_header,
        _ => std::ptr::null(),
    }
}

unsafe extern "C" fn out_events_push(_list: *const clap_output_events, _event: *const clap_event_header) -> bool {
    // What a plugin reports back (parameter changes from its own GUI, note
    // ends) has nowhere to go in this host; accepting it is enough.
    true
}

struct ClapRealtime {
    plugin: *const clap_plugin,
    processing: Arc<AtomicBool>,
    inputs: Vec<clap_audio_buffer>,
    outputs: Vec<clap_audio_buffer>,
    declared_outputs: usize,
    /// Boxed so the pointer in `in_events` stays put when this moves.
    events: Box<EventBuffer>,
    in_events: clap_input_events,
    out_events: clap_output_events,
    note_dialect: Option<clap_note_dialect>,
    steady_time: i64,
}

unsafe impl Send for ClapRealtime {}

impl ClapRealtime {
    fn load_events(&mut self, events: &[MidiEvent]) {
        let buffer = &mut *self.events;
        buffer.len = 0;
        let Some(dialect) = self.note_dialect else { return };
        for ev in events {
            let Some(slot) = buffer.slots.get_mut(buffer.len) else { break };
            let status = ev.data[0] & 0xF0;
            let channel = (ev.data[0] & 0x0F) as i16;
            if dialect == CLAP_NOTE_DIALECT_CLAP {
                let (type_, velocity) = match status {
                    0x90 if ev.data[2] > 0 => (CLAP_EVENT_NOTE_ON, ev.data[2] as f64 / 127.0),
                    0x90 | 0x80 => (CLAP_EVENT_NOTE_OFF, ev.data[2] as f64 / 127.0),
                    _ => continue,
                };
                slot.note = clap_event_note {
                    header: clap_event_header {
                        size: std::mem::size_of::<clap_event_note>() as u32,
                        time: ev.offset,
                        space_id: CLAP_CORE_EVENT_SPACE_ID,
                        type_,
                        flags: 0,
                    },
                    note_id: -1,
                    port_index: 0,
                    channel,
                    key: ev.data[1] as i16,
                    velocity,
                };
            } else {
                slot.midi = clap_event_midi {
                    header: clap_event_header {
                        size: std::mem::size_of::<clap_event_midi>() as u32,
                        time: ev.offset,
                        space_id: CLAP_CORE_EVENT_SPACE_ID,
                        type_: CLAP_EVENT_MIDI,
                        flags: 0,
                    },
                    port_index: 0,
                    data: ev.data,
                };
            }
            buffer.len += 1;
        }
    }
}

impl RealtimeProcess for ClapRealtime {
    fn process(&mut self, _scratch: &mut AudioScratch, frames: usize, events: &[MidiEvent]) {
        IN_AUDIO_THREAD.with(|f| f.set(true));
        unsafe {
            let p = &*self.plugin;
            // `start_processing` belongs on the audio thread, so it is called
            // from the first block rather than from `initialize_audio`.
            if !self.processing.load(Ordering::Acquire) {
                let started = p.start_processing.is_none_or(|start| start(self.plugin));
                self.processing.store(started, Ordering::Release);
                if !started {
                    IN_AUDIO_THREAD.with(|f| f.set(false));
                    return;
                }
            }
            self.load_events(events);
            let process = clap_process {
                steady_time: self.steady_time,
                frames_count: frames as u32,
                transport: std::ptr::null(),
                audio_inputs: self.inputs.as_ptr(),
                audio_outputs: self.outputs.as_mut_ptr(),
                audio_inputs_count: self.inputs.len() as u32,
                audio_outputs_count: self.declared_outputs as u32,
                in_events: &self.in_events,
                out_events: &self.out_events,
            };
            if let Some(process_fn) = p.process {
                process_fn(self.plugin, &process);
            }
        }
        self.steady_time += frames as i64;
        self.events.len = 0;
        IN_AUDIO_THREAD.with(|f| f.set(false));
    }
}

// ---------------------------------------------------------------------------
// The editor
// ---------------------------------------------------------------------------

/// A CLAP plugin's GUI, embedded. Every call is made on the instance's main
/// thread, which also runs the GUI's timers and descriptors; the editor
/// window's own loop has nothing to pump.
struct ClapEditor {
    plugin: *const clap_plugin,
    gui: *const clap_plugin_gui,
    host: *const HostState,
    main: Arc<MainThread>,
    /// `create` succeeded, so `destroy` is owed.
    created: bool,
}

// Made on the GUI thread, then used by the editor window's thread, which hands
// every call to the plugin's main thread.
unsafe impl Send for ClapEditor {}
unsafe impl Sync for ClapEditor {}

impl ClapEditor {
    fn gui(&self) -> &clap_plugin_gui {
        unsafe { &*self.gui }
    }

    fn host(&self) -> &HostState {
        unsafe { &*self.host }
    }

    fn on_main<R: Send>(&self, f: impl FnOnce(&Self) -> R + Send) -> R {
        let main = self.main.clone();
        main.run(|| f(self))
    }
}

impl PluginEditor for ClapEditor {
    fn open(&mut self) -> Result<()> {
        let created = self.on_main(|e| unsafe {
            let create = e.gui().create.context("gui extension has no create")?;
            anyhow::ensure!(
                create(e.plugin, ClapInstance::window_api().as_ptr(), false),
                "the plugin would not create its editor"
            );
            if let Some(set_scale) = e.gui().set_scale {
                // Physical pixels throughout; a plugin that sizes itself from
                // the window system (as X11 and Windows plugins do) declines.
                set_scale(e.plugin, 1.0);
            }
            Ok(())
        });
        created?;
        self.created = true;
        Ok(())
    }

    fn size(&mut self) -> Option<(u32, u32)> {
        self.on_main(|e| {
            let get_size = e.gui().get_size?;
            let (mut w, mut h) = (0u32, 0u32);
            let ok = unsafe { get_size(e.plugin, &mut w, &mut h) };
            (ok && w > 0 && h > 0).then_some((w, h))
        })
    }

    fn can_resize(&mut self) -> bool {
        self.on_main(|e| e.gui().can_resize.is_some_and(|can| unsafe { can(e.plugin) }))
    }

    fn attach(&mut self, parent: ParentWindow) -> Result<()> {
        let handle = match parent {
            ParentWindow::X11 { window, .. } => Handle::X11(window),
            ParentWindow::Win32 { hwnd } => Handle::Win32(Ptr(hwnd as *const c_void)),
            // A Cocoa GUI must run on the process's main thread, and a CLAP
            // plugin's "main thread" here is one of its own.
            ParentWindow::Cocoa { .. } => {
                anyhow::bail!("CLAP plugin editors are not supported on macOS yet")
            }
        };
        self.on_main(|e| unsafe {
            let specific = match handle {
                Handle::X11(window) => clap_window_handle { x11: window as _ },
                Handle::Win32(hwnd) => clap_window_handle { win32: hwnd.get() as *mut c_void },
            };
            let window = clap_window { api: ClapInstance::window_api().as_ptr(), specific };
            let set_parent = e.gui().set_parent.context("gui extension has no set_parent")?;
            anyhow::ensure!(
                set_parent(e.plugin, &window),
                "the plugin would not embed its editor in the window"
            );
            if let Some(show) = e.gui().show {
                show(e.plugin);
            }
            Ok(())
        })
    }

    fn set_size(&mut self, width: u32, height: u32) {
        self.on_main(|e| unsafe {
            let (mut w, mut h) = (width, height);
            if let Some(adjust) = e.gui().adjust_size {
                adjust(e.plugin, &mut w, &mut h);
            }
            if let Some(set) = e.gui().set_size {
                set(e.plugin, w, h);
            }
        })
    }

    fn take_resize_request(&mut self) -> Option<(u32, u32)> {
        self.host().resize_request.lock().unwrap().take()
    }

    fn detach(&mut self) {
        if !std::mem::take(&mut self.created) {
            return;
        }
        self.on_main(|e| unsafe {
            if let Some(hide) = e.gui().hide {
                hide(e.plugin);
            }
            if let Some(destroy) = e.gui().destroy {
                destroy(e.plugin);
            }
        })
    }
}

/// A parent window handle on its way to the main thread.
#[derive(Clone, Copy)]
enum Handle {
    X11(u64),
    Win32(Ptr<c_void>),
}

impl Drop for ClapEditor {
    fn drop(&mut self) {
        self.detach();
    }
}
