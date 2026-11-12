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
//! Loading a VST3 **module** — the shared library behind a plugin — the way the
//! spec asks a host to.
//!
//! Two things here are easy to get wrong and both make a perfectly good
//! third-party plugin look broken:
//!
//!   * **A VST3 is a bundle, not a file.** What a user installs is a directory
//!     `Foo.vst3/` holding `Contents/x86_64-linux/Foo.so`. Only our own internal
//!     plugin is ever a bare `.so`, so [`resolve_module_path`] accepts either and
//!     digs the real library out of a bundle.
//!   * **`ModuleEntry` must be called.** On Linux the entry point is not
//!     `GetPluginFactory` alone: the host must call `ModuleEntry(handle)` after
//!     `dlopen` and `ModuleExit()` before `dlclose`. JUCE initialises its whole
//!     runtime there, so a JUCE plugin whose `ModuleEntry` was skipped hands back
//!     a factory that crashes or an editor that never draws. macOS spells the
//!     pair `bundleEntry(CFBundleRef)` / `bundleExit()`, and SDK plugins find
//!     their resources through that bundle, so it has to be a real one.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use vst3::Steinberg::{
    kResultOk, IPluginFactory, IPluginFactory2, IPluginFactory2Trait, IPluginFactoryTrait,
    PClassInfo, PClassInfo2,
};
use vst3::{ComPtr, ComRef};

type GetPluginFactoryProc = unsafe extern "C" fn() -> *mut IPluginFactory;
/// `bool ModuleEntry(void* sharedLibraryHandle)` — Linux; the handle is the
/// `dlopen` one, which SDK plugins use to find their own bundle's resources.
type ModuleEntryProc = unsafe extern "C" fn(*mut c_void) -> bool;
type ModuleExitProc = unsafe extern "C" fn() -> bool;
/// Windows' spelling of the same pair (no handle argument).
#[cfg(target_os = "windows")]
type InitDllProc = unsafe extern "C" fn() -> bool;
/// macOS' spelling: `bool bundleEntry(CFBundleRef)`.
#[cfg(target_os = "macos")]
type BundleEntryProc = unsafe extern "C" fn(*mut c_void) -> bool;

/// Architecture subdirectories of `Contents/`, in the order we try them.
#[cfg(target_os = "linux")]
const ARCH_DIRS: &[&str] = &["x86_64-linux", "aarch64-linux", "armv7l-linux", "i386-linux"];
#[cfg(target_os = "windows")]
const ARCH_DIRS: &[&str] = &["x86_64-win", "arm64-win", "x86-win"];
#[cfg(target_os = "macos")]
const ARCH_DIRS: &[&str] = &["MacOS"];
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
const ARCH_DIRS: &[&str] = &[];

/// Extensions a module inside a bundle may carry. Windows bundles name the DLL
/// `.vst3`; a loose `.dll` is accepted too, as is a loose `.so` on Linux.
#[cfg(target_os = "linux")]
const MODULE_EXTS: &[&str] = &["so"];
#[cfg(target_os = "windows")]
const MODULE_EXTS: &[&str] = &["vst3", "dll"];
#[cfg(target_os = "macos")]
const MODULE_EXTS: &[&str] = &["dylib", "so"];
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
const MODULE_EXTS: &[&str] = &["so"];

/// Turn whatever the user picked into the shared library to load.
///
/// Accepts a bare library file (our internal plugin, or a loose `.so`), a
/// `.vst3` bundle directory, or any level inside one (`Contents`, or the
/// architecture directory itself) — a file dialog makes all four easy to land on.
pub fn resolve_module_path(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if !path.exists() {
        bail!("no such file or directory: {}", crate::file_label(path));
    }

    // A bundle, or something inside one. Walk down as far as we recognise.
    let contents = if path.join("Contents").is_dir() {
        path.join("Contents")
    } else {
        path.to_path_buf()
    };
    for arch in ARCH_DIRS {
        let dir = contents.join(arch);
        if dir.is_dir() {
            if let Some(found) = pick_module_in(&dir, path) {
                return Ok(found);
            }
        }
    }
    // The user may have picked the architecture directory itself.
    if let Some(found) = pick_module_in(path, path) {
        return Ok(found);
    }

    bail!(
        "{} is not a VST3 bundle — expected {}/Contents/{}/*.{} inside it",
        crate::file_label(path),
        path.file_name().unwrap_or_default().to_string_lossy(),
        ARCH_DIRS.first().copied().unwrap_or("<arch>"),
        MODULE_EXTS.first().copied().unwrap_or("so"),
    );
}

/// The library file in `dir`, preferring one named after the bundle (bundles may
/// also carry helper libraries beside the module).
fn pick_module_in(dir: &Path, bundle: &Path) -> Option<PathBuf> {
    let wanted = bundle.file_stem().map(|s| s.to_ascii_lowercase());
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            // A macOS bundle's executable, `Contents/MacOS/Foo`, has no
            // extension at all.
            p.extension()
                .map(|e| MODULE_EXTS.iter().any(|m| e.eq_ignore_ascii_case(m)))
                .unwrap_or(cfg!(target_os = "macos"))
        })
        .collect();
    candidates.sort();
    candidates
        .iter()
        .find(|p| p.file_stem().map(|s| s.to_ascii_lowercase()) == wanted)
        .or_else(|| candidates.first())
        .cloned()
}

/// Check that `path` really is a loadable VST3, without starting the plugin up.
///
/// This is what the Instruments panel calls the moment a plugin is picked, so the two
/// ways a choice can be wrong — a library the loader cannot satisfy, and a file
/// that is not a VST3 at all — are reported there and then rather than surfacing
/// much later as "Open editor failed". Returns the resolved module path.
pub fn validate_module(path: &Path) -> Result<PathBuf> {
    let resolved = resolve_module_path(path)?;
    let (library, _handle) = open_library(&resolved)
        .with_context(|| format!("Failed to open {}", crate::file_label(&resolved)))?;
    // Deliberately no `ModuleEntry`: this only asks whether the entry point is
    // there, so nothing of the plugin actually runs.
    unsafe {
        if library
            .get::<GetPluginFactoryProc>(b"GetPluginFactory\0")
            .is_ok()
        {
            return Ok(resolved);
        }
        Err(not_a_vst3(&library, &resolved))
    }
}

/// What a factory says about one of its classes.
#[derive(Clone, Debug, Default)]
pub struct ModuleClass {
    /// The class id, in the crate's own byte order — see [`platform_cid`], which
    /// is what turns it back into the order the factory wants.
    pub cid: [i8; 16],
    /// The plugin's own display name.
    pub name: String,
    /// `Audio Module Class` for something that makes sound.
    pub category: String,
    /// `Instrument|Drum`, `Fx|Delay`, … — the plugin's declaration of what it
    /// *is*, and the only precise way to know a drum kit from a synth. Empty
    /// when the factory is too old to be asked (`IPluginFactory2`).
    pub subcategories: String,
}

/// A class id in the byte order the *factory* uses, from one in the order the
/// rest of this crate uses — or back again.
///
/// VST3 class ids do not have one byte order. On Windows the SDK lays a class id
/// out as a COM `GUID`, whose first three fields are little-endian; everywhere
/// else it is the plain 16-byte string. The same plugin, built for the two
/// platforms, therefore announces two different ids for the same class — which
/// is why `lesynth_fourier.dll` under Wine answered "the requested plugin class
/// is not in this factory" to the id `liblesynth_fourier.so` answers to.
///
/// So the crate deals in one order — the non-Windows one, which is also the
/// order [`crate::vst::class_ids`] and a saved project are written in, so a
/// project stays portable — and converts only here, where a factory is read or
/// asked to create something. The permutation is its own inverse, so this one
/// function serves both directions.
///
/// A controller class id read from a component (`getControllerClassId`) never
/// passes through here: it comes from the factory's world and goes straight
/// back to it.
#[cfg(target_os = "windows")]
pub const fn platform_cid(cid: [i8; 16]) -> [i8; 16] {
    com_cid(cid)
}

#[cfg(not(target_os = "windows"))]
pub const fn platform_cid(cid: [i8; 16]) -> [i8; 16] {
    cid
}

/// The COM `GUID` reading of a class id: the first three fields byte-swapped
/// into little-endian, the last eight bytes left alone. Compiled on every
/// platform so it can be tested anywhere, and applied only on Windows.
const fn com_cid(cid: [i8; 16]) -> [i8; 16] {
    let c = cid;
    [
        c[3], c[2], c[1], c[0], // Data1: a little-endian u32
        c[5], c[4], // Data2: a little-endian u16
        c[7], c[6], // Data3: a little-endian u16
        c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15], // Data4: plain bytes
    ]
}

/// Ask a plugin what it is, without creating an instance of it.
///
/// This is a scan, not a load: the module is opened, its factory read, and the
/// module closed again. It costs one `dlopen` and the plugin's `ModuleEntry`,
/// which is what any host's plugin scan costs.
pub fn scan_classes(path: &Path) -> Result<Vec<ModuleClass>> {
    let module = Vst3Module::open(path)?;
    let factory = module.factory()?;
    Ok(classes(factory.as_com_ref()))
}

/// Every class in a factory, with the richer `getClassInfo2` fields where the
/// factory supports them.
pub fn classes(factory: ComRef<'_, IPluginFactory>) -> Vec<ModuleClass> {
    unsafe {
        let factory2 = factory.cast::<IPluginFactory2>();
        (0..factory.countClasses())
            .filter_map(|idx| {
                if let Some(f2) = &factory2 {
                    let mut info: PClassInfo2 = std::mem::zeroed();
                    if f2.as_com_ref().getClassInfo2(idx, &mut info) == kResultOk {
                        return Some(ModuleClass {
                            cid: platform_cid(info.cid),
                            name: c_string(&info.name),
                            category: c_string(&info.category),
                            subcategories: c_string(&info.subCategories),
                        });
                    }
                }
                let mut info: PClassInfo = std::mem::zeroed();
                if factory.getClassInfo(idx, &mut info) == kResultOk {
                    return Some(ModuleClass {
                        cid: platform_cid(info.cid),
                        name: c_string(&info.name),
                        category: c_string(&info.category),
                        subcategories: String::new(),
                    });
                }
                None
            })
            .collect()
    }
}

/// A NUL-terminated `char8` array from class info as a Rust string.
fn c_string(bytes: &[i8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    bytes[..end].iter().map(|&b| b as u8 as char).collect()
}

/// A loaded VST3 module: the library, kept alive, plus the `ModuleEntry` /
/// `ModuleExit` pair the spec puts around its lifetime.
pub struct Vst3Module {
    library: Library,
    /// The raw `dlopen` handle, which is what `ModuleEntry` wants. `libloading`
    /// only surfaces it by consuming the library, so it is captured at load.
    handle: *mut c_void,
    /// The resolved library path (not the bundle the user picked).
    path: PathBuf,
    /// Whether `ModuleEntry` succeeded, so `Drop` knows to call `ModuleExit`.
    entered: bool,
    /// macOS: the `CFBundleRef` handed to `bundleEntry`, released after
    /// `bundleExit`. Null for a bare `.dylib` (our internal plugin).
    #[cfg(target_os = "macos")]
    bundle: *mut c_void,
}

// The library and everything reached through it is used from the GUI thread, the
// editor thread and the audio thread alike; `Library` itself is `Send + Sync`.
unsafe impl Send for Vst3Module {}
unsafe impl Sync for Vst3Module {}

impl Vst3Module {
    /// Load the module at `path` (a bundle or a bare library) and run its entry
    /// point.
    pub fn open(path: &Path) -> Result<Self> {
        let resolved = resolve_module_path(path)
            .with_context(|| format!("Cannot load {}", crate::file_label(path)))?;

        // Match the reference host: RTLD_LAZY | RTLD_LOCAL. Binding lazily keeps
        // a plugin that carries unresolved host-API symbols (several JUCE builds
        // do) loadable, and LOCAL stops its symbols leaking into ours.
        let (library, handle) = open_library(&resolved)
            .with_context(|| format!("Failed to open {}", crate::file_label(&resolved)))?;

        let mut module = Vst3Module {
            library,
            handle,
            path: resolved,
            entered: false,
            #[cfg(target_os = "macos")]
            bundle: std::ptr::null_mut(),
        };
        module.enter()?;
        module.check_is_vst3()?;
        Ok(module)
    }

    /// The resolved library path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The library, for the LeSynth-specific C ABI symbols in `host.rs`.
    pub fn library(&self) -> &Library {
        &self.library
    }

    /// The plugin's factory. Every call re-enters `GetPluginFactory`, which the
    /// spec requires to hand back the same singleton with a fresh reference.
    pub fn factory(&self) -> Result<ComPtr<IPluginFactory>> {
        unsafe {
            let get_factory: libloading::Symbol<GetPluginFactoryProc> = self
                .library
                .get(b"GetPluginFactory\0")
                .context("not a VST3 module: no GetPluginFactory symbol")?;
            ComPtr::from_raw(get_factory()).context("GetPluginFactory returned null")
        }
    }

    /// Call `ModuleEntry`/`InitDll`. Absent is not an error — plugins built
    /// without one (including ours) simply have nothing to initialise — but a
    /// present one returning `false` means the plugin refused to start.
    fn enter(&mut self) -> Result<()> {
        unsafe {
            #[cfg(target_os = "windows")]
            if let Ok(init) = self.library.get::<InitDllProc>(b"InitDll\0") {
                anyhow::ensure!(init(), "the plugin's InitDll() returned false");
                self.entered = true;
                return Ok(());
            }

            #[cfg(target_os = "macos")]
            if let Ok(entry) = self.library.get::<BundleEntryProc>(b"bundleEntry\0") {
                self.bundle = macos::bundle_of(&self.path);
                anyhow::ensure!(
                    entry(self.bundle),
                    "the plugin's bundleEntry() returned false — it declined to load"
                );
                self.entered = true;
                return Ok(());
            }

            if let Ok(entry) = self.library.get::<ModuleEntryProc>(b"ModuleEntry\0") {
                anyhow::ensure!(
                    entry(self.handle),
                    "the plugin's ModuleEntry() returned false — it declined to load"
                );
                self.entered = true;
            } else {
                log::debug!("{} has no ModuleEntry", self.path.display());
            }
        }
        Ok(())
    }

    /// Fail early, and legibly, on a library that is not a VST3 at all — by far
    /// the most common way "open the plugin" goes wrong, since a VST2 `.so` and a
    /// VST3 `.so` look identical in a file dialog.
    fn check_is_vst3(&self) -> Result<()> {
        unsafe {
            if self
                .library
                .get::<GetPluginFactoryProc>(b"GetPluginFactory\0")
                .is_ok()
            {
                return Ok(());
            }
            Err(not_a_vst3(&self.library, &self.path))
        }
    }
}

impl Drop for Vst3Module {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        let _bundle = macos::Owned(self.bundle);
        if !self.entered {
            return;
        }
        unsafe {
            #[cfg(target_os = "windows")]
            if let Ok(exit) = self.library.get::<ModuleExitProc>(b"ExitDll\0") {
                exit();
                return;
            }
            #[cfg(target_os = "macos")]
            if let Ok(exit) = self.library.get::<ModuleExitProc>(b"bundleExit\0") {
                exit();
                return;
            }
            if let Ok(exit) = self.library.get::<ModuleExitProc>(b"ModuleExit\0") {
                exit();
            }
        }
    }
}

/// The error for a library with no VST3 entry point, naming the likely reason.
unsafe fn not_a_vst3(library: &Library, path: &Path) -> anyhow::Error {
    let vst2 = library.get::<*mut c_void>(b"VSTPluginMain\0").is_ok()
        || library.get::<*mut c_void>(b"main_plugin\0").is_ok();
    anyhow!(
        "{} is not a VST3 plugin — it has no GetPluginFactory entry point{}",
        crate::file_label(path),
        if vst2 {
            " (it exports VSTPluginMain, so it is a VST2 plugin, which this host cannot load)"
        } else {
            ""
        }
    )
}

/// Open the library and keep its raw handle: `into_raw` / `from_raw` is the only
/// way `libloading` lets a caller see it, and the round-trip is a no-op.
#[cfg(unix)]
fn open_library(path: &Path) -> Result<(Library, *mut c_void)> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};
    // NODELETE: see `crate::plugin::open_pinned` — unloading a plugin's code is
    // what crashed the host.
    let lib = unsafe { UnixLibrary::open(Some(path), RTLD_LAZY | RTLD_LOCAL | libc::RTLD_NODELETE)? };
    let handle = lib.into_raw();
    let lib = unsafe { UnixLibrary::from_raw(handle) };
    Ok((Library::from(lib), handle))
}

/// Just enough CoreFoundation to hand `bundleEntry` the plugin's own bundle.
#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::c_void;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFURLCreateFromFileSystemRepresentation(
            allocator: *const c_void,
            buffer: *const u8,
            length: isize,
            is_directory: u8,
        ) -> *mut c_void;
        fn CFBundleCreate(allocator: *const c_void, url: *mut c_void) -> *mut c_void;
        fn CFRelease(cf: *mut c_void);
    }

    /// The `CFBundleRef` of the `.vst3` bundle `Foo.vst3/Contents/MacOS/Foo`
    /// lives in, or null when the library is not inside one.
    pub fn bundle_of(library: &Path) -> *mut c_void {
        let Some(bundle) = library.ancestors().nth(3) else {
            return std::ptr::null_mut();
        };
        if !bundle.extension().is_some_and(|e| e.eq_ignore_ascii_case("vst3")) {
            return std::ptr::null_mut();
        }
        let bytes = bundle.as_os_str().as_bytes();
        unsafe {
            let url = CFURLCreateFromFileSystemRepresentation(
                std::ptr::null(),
                bytes.as_ptr(),
                bytes.len() as isize,
                1,
            );
            if url.is_null() {
                return std::ptr::null_mut();
            }
            let bundle = CFBundleCreate(std::ptr::null(), url);
            CFRelease(url);
            bundle
        }
    }

    /// Releases a CF object (if any) when dropped — after `bundleExit` ran.
    pub struct Owned(pub *mut c_void);

    impl Drop for Owned {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CFRelease(self.0) };
            }
        }
    }
}

#[cfg(not(unix))]
fn open_library(path: &Path) -> Result<(Library, *mut c_void)> {
    Ok((crate::plugin::open_pinned(path)?, std::ptr::null_mut()))
}

#[cfg(test)]
mod cid_tests {
    use super::com_cid;

    /// The layout the VST3 SDK's `INLINE_UID` writes on Windows: the id's first
    /// three GUID fields are byte-swapped, the trailing eight bytes are not.
    #[test]
    fn com_order_swaps_only_the_first_three_fields() {
        let raw: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(
            com_cid(raw),
            [3, 2, 1, 0, 5, 4, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15],
        );
    }

    /// Reading a factory and writing back to it use the same function, which is
    /// only sound because the permutation is its own inverse.
    #[test]
    fn com_order_is_its_own_inverse() {
        let raw = *b"LeSynthFourier01";
        let raw: [i8; 16] = std::array::from_fn(|i| raw[i] as i8);
        assert_eq!(com_cid(com_cid(raw)), raw);
    }
}
