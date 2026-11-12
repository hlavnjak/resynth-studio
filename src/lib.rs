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
pub mod analysis;
pub mod audio;
pub mod gui;
pub mod midi;
pub mod plugin;
pub mod track_format;
pub mod vst;

use std::path::{Path, PathBuf};

/// File name of the embedded LeSynth Fourier plugin on this platform.
#[cfg(target_os = "linux")]
pub const INTERNAL_PLUGIN_LIB: &str = "liblesynth_fourier.so";
#[cfg(target_os = "macos")]
pub const INTERNAL_PLUGIN_LIB: &str = "liblesynth_fourier.dylib";
#[cfg(target_os = "windows")]
pub const INTERNAL_PLUGIN_LIB: &str = "lesynth_fourier.dll";

/// Where the embedded plugin lives: `internal_plugins/` in the working
/// directory (a checkout run with `make run`), else beside the executable, else
/// in a macOS `.app`'s `Contents/PlugIns/` — an app started from Finder runs
/// with `/` as its working directory, so the first place alone never finds it
/// there. When none exists the working-directory path is returned, so the error
/// the caller reports names the place a checkout expects.
pub fn internal_plugin_path() -> Option<PathBuf> {
    let cwd = std::env::current_dir()
        .ok()
        .map(|d| d.join("internal_plugins").join(INTERNAL_PLUGIN_LIB));
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf));
    let beside_exe = exe_dir.iter().flat_map(|d| {
        [
            d.join("internal_plugins"),
            d.join("../PlugIns"),
        ]
    });
    cwd.iter()
        .cloned()
        .chain(beside_exe.map(|d| d.join(INTERNAL_PLUGIN_LIB)))
        .find(|p| p.is_file())
        .or(cwd)
}

/// What a path is called on screen: its last component, never the whole thing.
///
/// A window is not a terminal. A plugin two directories deep, a project folder
/// under a home directory, an exported `.wav` chosen from a dialog — printed in
/// full they push everything beside them off the line, wrap the status area, and
/// say nothing the user did not just type into a file dialog. The name is the
/// part they recognise.
///
/// This is for the **UI only**. Every `log::` line keeps the whole path, because
/// that is what a log is for: a message saying a write failed is worth nothing
/// without the directory it failed in, and the log file is where that belongs.
pub fn file_label(path: impl AsRef<Path>) -> String {
    let path = path.as_ref();
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        // A path that ends in `..` or is a bare root has no last component. It
        // is not a file name, but it is short, and it is what there is.
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::file_label;

    /// The UI shows what a file is called, never where it lives.
    #[test]
    fn a_path_shows_as_its_name_alone() {
        assert_eq!(file_label("/home/kuba/Music/My Song/My Song.rsyn"), "My Song.rsyn");
        assert_eq!(file_label("/opt/vst3/Dexed.vst3"), "Dexed.vst3");
        // A folder is named by its own last component, which is what the
        // Sequencer shows for a project directory.
        assert_eq!(file_label("/home/kuba/Music/My Song"), "My Song");
        // Relative paths, and a name that is already bare, come through as-is.
        assert_eq!(file_label("internal_plugins/liblesynth_fourier.so"), "liblesynth_fourier.so");
        assert_eq!(file_label("voice.lsft"), "voice.lsft");
        // Nothing to take a name from: better the odd string than an empty
        // label with no clue in it.
        assert_eq!(file_label("/"), "/");
        assert_eq!(file_label(".."), "..");
    }
}
