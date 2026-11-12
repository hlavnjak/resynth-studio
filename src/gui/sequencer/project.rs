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
//! `.rsyn` — the ReSynth Studio project format.
//!
//! A saved project is a **folder**, not a file: `MySong/` holds `MySong.rsyn`,
//! one `.lsft` per LeSynth Fourier instrument any row plays, and one `.vststate` per
//! plugin instrument — the plugin's own state, which is where it keeps the knobs
//! the user set. A LeSynth instrument has both, and needs both: the `.lsft` is the
//! harmonic grid, the `.vststate` is every control that drew it (curve type,
//! offset, granularity, and the nested-Fourier sliders), which the grid does
//! not contain. The folder is the unit you move, copy or hand to someone else,
//! and it carries its own sounds — the audio file behind a wav instrument included,
//! copied into `audio/` on save, because a recording kept only by path is lost
//! the day its folder is tidied away. What is left outside is a third-party
//! plugin's *binary*, which is not ours to copy, and is recorded as the path it
//! was added from.
//!
//! **The manifest is line-oriented text, deliberately.** The grids are already
//! binary (`.lsft`); what is left is a small structure whose main job is to hold
//! *file paths and names*, and whose main failure mode is a source that has gone
//! missing. Text means a user can repair that in an editor. It also means
//! `key = value` needs no quoting or escaping at all — the value is the rest of
//! the line — which is exactly what a format full of paths wants, and what JSON
//! would have made worse.
//!
//! ```text
//! resynth-project 1
//! name = My Song
//! tempo = 120
//! repeat-from = 1 none
//! repeat-to = 2 1/2
//!
//! [row]
//! name = lead
//! instrument = LeSynth Fourier 1
//! source = lesynth voice.lsft
//! state = LeSynth Fourier 1.vststate
//! gain = 1
//! lead = 0 none
//! enabled = 1
//! autosave = 1
//! note = 60 0 1/4 0 3/8
//! ```
//!
//! A note line is five fields, and a sixth when a **wav row's** note starts
//! somewhere other than the top of its file: `note = 60 0 1/4 0 3/8 1.25` plays
//! from 1.25 s in. A note that starts at zero — which is every note on every
//! instrument — writes the five it always did, so a project with no such note in
//! it is byte for byte what it was.
//!
//! A length is two fields — whole notes, then the fraction, nominator and all:
//! `0 3/8` is three eighths, `2 1/4` is two whole notes and a quarter, `1 none`
//! is one whole note. A length with no nominator writes exactly the token it
//! always did, so files that do not use one are unchanged and load in any build;
//! one that does is refused by a build too old to know `3/8`, which is a message
//! rather than a silent misreading.
//!
//! The two `repeat-` keys are the window a repeat loops over, each a length
//! **from the beginning** of the sequence. `repeat-to` of zero — which is
//! how it is written when it is absent — is the end of the sequence. Both
//! are written only when they are not zero, so a project that loops the whole
//! sequence, which is every project that has not been told otherwise, is
//! byte for byte the file it always was.
//!
//! Unknown keys are skipped, so a field added later loads in an older build
//! rather than failing — `enabled`, the newest of them, is exactly that: an
//! older build ignores the key and plays the row, and a project written before
//! it existed has no key and reads as `enabled`, which is what a row is.
//!
//! `name` means two things, told apart by where it is: above the first `[row]`
//! it is the project's name, and inside a row it is the **row's**. That is not a
//! collision to work around — it is the same word for the same thing at two
//! levels — and an older build, whose reader only knows the project's, skips the
//! row's and loads the row unnamed.
//!
//! The version line is checked, so a *newer* format is
//! refused with a message instead of being half-read. A *source kind* added
//! later — `wav` is the most recent — follows the rule the nominator does: a
//! project that does not use one is unchanged and loads in any build, and one
//! that does is refused by a build too old to know it, which is a message
//! rather than a silent misreading.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::{Duration, Fraction, Item};

/// What this writer produces. A reader accepts anything up to it.
const VERSION: u32 = 1;
const MAGIC: &str = "resynth-project";
/// What the app wrote before it was renamed from Gemstone DAW. Same format,
/// same version numbers — only the word on the first line differs — so it is
/// read exactly like [`MAGIC`] and the next save writes the new one.
const LEGACY_MAGIC: &str = "gemstone-project";

/// The file extension, and what a project folder's manifest is named after.
pub const EXTENSION: &str = "rsyn";
/// The extension projects were saved under as Gemstone DAW. Opened, never
/// written: saving such a project puts a `.rsyn` beside the old file.
pub const LEGACY_EXTENSION: &str = "gmstn";

/// Where a row's sound comes from, as recorded in the manifest.
#[derive(Clone, Debug, PartialEq)]
pub enum InstrumentSource {
    /// A LeSynth Fourier grid saved beside the manifest. The name is relative to
    /// the project folder, so the folder stays portable.
    ///
    /// `state` names the plugin's own `.vststate` beside it. The grid is the
    /// picture; the state is every control that drew it — each harmonic's curve
    /// type, offset and granularity, and the nested-Fourier sliders under them,
    /// none of which is in the grid. `None` is a project saved before this was
    /// kept, which reloads its curves with the controls at their defaults.
    LeSynth { file: String, state: Option<String> },
    /// A LeSynth Fourier instrument carrying no grid — the plugin's own synth mode.
    LeSynthDefault,
    /// A custom plugin — of any format, despite the name, which is the one
    /// the manifest has always used — by absolute path. Not portable, and
    /// cannot be: the plugin is not ours to copy into the folder. What *is*
    /// saved beside the manifest is `state` — the plugin's own saved state,
    /// named relative to the project folder. `None` means the project was saved
    /// before the plugin had any state to keep, or by a build that did not save
    /// it.
    ///
    /// `plugin_id` names the plugin inside the file where one file holds
    /// several (a CLAP plugin id, an LV2 URI) or there is no file at all (an
    /// Audio Unit's codes). It rides on its own `plugin-id =` line, which an
    /// older build skips — and then loads the first plugin in the file, which
    /// for a file holding one is the same thing.
    Vst {
        path: PathBuf,
        class_id: Option<[i8; 16]>,
        plugin_id: Option<String>,
        state: Option<String>,
    },
    /// An audio file played whole, as one note. Saving copies the file into
    /// the project folder's [`AUDIO_DIR`] (see [`bundle_audio`]) and names it
    /// *relative* to the folder, so the folder carries its recordings the way
    /// it carries its grids. An absolute path is a project saved before that,
    /// or a file that was already gone when it was saved; a relative one is
    /// resolved against the folder by [`Project::resolve_audio`].
    Wav { path: PathBuf },
    /// The row had no instrument assigned when it was saved.
    None,
}

impl InstrumentSource {
    /// What this source needs from the filesystem, for the caller to check
    /// before trying to load it. `None` for sources that need nothing.
    pub fn required_path(&self, dir: &Path) -> Option<PathBuf> {
        match self {
            Self::LeSynth { file, .. } => Some(dir.join(file)),
            // The plugin itself. A missing `.vststate` is not fatal — the instrument
            // loads with the plugin's defaults — so it is not required here.
            Self::Vst { path, plugin_id, .. } => crate::plugin::has_file(path, plugin_id.as_deref())
                .then(|| path.clone()),
            // The whole sound of the row: without the file there is nothing to
            // play at all. Relative is inside the folder; absolute stays as is.
            Self::Wav { path } => Some(dir.join(path)),
            Self::LeSynthDefault | Self::None => None,
        }
    }

    /// How to name this source in a message when it cannot be found.
    pub fn describe(&self) -> String {
        match self {
            Self::LeSynth { file, .. } => file.clone(),
            Self::Vst { path, .. } | Self::Wav { path } => crate::file_label(path),
            Self::LeSynthDefault => "LeSynth Fourier".to_string(),
            Self::None => "no instrument".to_string(),
        }
    }
}

/// One lane: which sound it plays, how loud, and the chain of frames on it.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectRow {
    /// What the row is called — see [`super::Row::name`]. Empty is a project
    /// written before rows had names, and reads as "unnamed": the row is given
    /// a fresh UUID when it is loaded, exactly as a new row would be.
    pub name: String,
    /// Display name of the instrument, kept so a row whose source has gone missing
    /// can still say what it was looking for.
    pub instrument_name: String,
    pub source: InstrumentSource,
    pub gain: f32,
    pub lead: Duration,
    /// Play this row at all — see [`super::Row::enabled`]. A project written
    /// before this existed has no such key, and a row from one reads as `true`:
    /// every row in every older project played, so that is what it must still
    /// mean.
    pub enabled: bool,
    /// Re-export this row's grid on every save — see [`super::Row::autosave`].
    pub autosave: bool,
    /// Follow the transport in this lane — see [`super::Row::autoscroll`]. A
    /// project written before this existed has no such key, and a row from one
    /// reads as `true`, which is the default a new row gets.
    pub autoscroll: bool,
    pub items: Vec<Item>,
}

/// A whole sequence, as it sits in the manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct Project {
    pub name: String,
    pub tempo_bpm: f32,
    /// The repeat window: where a repeat starts and where it stops, each a
    /// length from the beginning of the sequence. A stop of zero is the end
    /// of the sequence, which is what a repeat loops to without being told
    /// otherwise.
    pub repeat_from: Duration,
    pub repeat_to: Duration,
    pub rows: Vec<ProjectRow>,
}

impl Project {
    pub fn to_text(&self) -> String {
        let mut s = format!("{MAGIC} {VERSION}\n");
        s += &format!("name = {}\n", one_line(&self.name));
        s += &format!("tempo = {}\n", num(self.tempo_bpm));
        // Only when there is a window to record. A repeat that loops the whole
        // sequence writes nothing, so files that do not use one are
        // unchanged — and an older build, which skips keys it does not know,
        // loads one that does and simply loops the whole thing.
        if self.repeat_from.units() > 0 {
            s += &format!("repeat-from = {}\n", write_duration(self.repeat_from));
        }
        if self.repeat_to.units() > 0 {
            s += &format!("repeat-to = {}\n", write_duration(self.repeat_to));
        }
        for row in &self.rows {
            s += "\n[row]\n";
            // The row's own name first: inside a [row] block `name` is the
            // row's, and a build too old to know the key skips it and loads the
            // row it always did.
            if !row.name.is_empty() {
                s += &format!("name = {}\n", one_line(&row.name));
            }
            s += &format!("track = {}\n", one_line(&row.instrument_name));
            s += &format!("source = {}\n", write_source(&row.source));
            if let InstrumentSource::Vst { state: Some(file), .. }
            | InstrumentSource::LeSynth { state: Some(file), .. } = &row.source
            {
                s += &format!("state = {}\n", one_line(file));
            }
            if let InstrumentSource::Vst { plugin_id: Some(id), .. } = &row.source {
                s += &format!("plugin-id = {}\n", one_line(id));
            }
            s += &format!("gain = {}\n", num(row.gain));
            s += &format!("lead = {}\n", write_duration(row.lead));
            s += &format!("enabled = {}\n", u8::from(row.enabled));
            s += &format!("autosave = {}\n", u8::from(row.autosave));
            s += &format!("autoscroll = {}\n", u8::from(row.autoscroll));
            for item in &row.items {
                s += &format!(
                    "note = {} {} {}",
                    item.pitch,
                    write_duration(item.dur),
                    write_duration(item.space)
                );
                // Where in the file the note starts — a wav row only, and only
                // when it is not the top of the file. A note that starts there
                // writes the five fields this format has always written, so a
                // project with no wav row in it is byte for byte what it was.
                if item.start > 0.0 {
                    s += &format!(" {}", num(item.start));
                }
                s += "\n";
            }
        }
        s
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
        let header = lines.next().context("empty project file")?;
        let (magic, version) = header.split_once(' ').unwrap_or((header, ""));
        if magic != MAGIC && magic != LEGACY_MAGIC {
            bail!("not a ReSynth Studio project file (first line reads {header:?})");
        }
        let version: u32 = version.trim().parse().context("unreadable format version")?;
        if version > VERSION {
            bail!(
                "project was saved by a newer ReSynth Studio ({version} > {VERSION}); \
                 update before opening it"
            );
        }

        let mut project = Project {
            name: String::new(),
            tempo_bpm: 120.0,
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            rows: Vec::new(),
        };
        for line in lines {
            if line == "[row]" {
                project.rows.push(ProjectRow {
                    name: String::new(),
                    instrument_name: String::new(),
                    source: InstrumentSource::None,
                    gain: 1.0,
                    lead: Duration::new(0, Fraction::None),
                    enabled: true,
                    autosave: true,
                    autoscroll: true,
                    items: Vec::new(),
                });
                continue;
            }
            // The value is the rest of the line, so nothing needs escaping — see
            // the module docs. Unknown keys are skipped for forward compatibility.
            let Some((key, value)) = line.split_once('=') else { continue };
            let (key, value) = (key.trim(), value.trim());
            match (key, project.rows.last_mut()) {
                ("name", None) => project.name = value.to_string(),
                ("name", Some(row)) => row.name = value.to_string(),
                ("tempo", None) => project.tempo_bpm = value.parse().unwrap_or(120.0),
                ("repeat-from", None) => project.repeat_from = read_duration(value)?,
                ("repeat-to", None) => project.repeat_to = read_duration(value)?,
                ("track", Some(row)) => row.instrument_name = value.to_string(),
                ("source", Some(row)) => row.source = read_source(value)?,
                // Belongs to the source above it, which is where the writer puts
                // it. Ignored for any other kind of source.
                ("state", Some(row)) => {
                    if let InstrumentSource::Vst { state, .. }
                    | InstrumentSource::LeSynth { state, .. } = &mut row.source
                    {
                        *state = Some(value.to_string());
                    }
                }
                // The same: the plugin inside the file the source names.
                ("plugin-id", Some(row)) => {
                    if let InstrumentSource::Vst { plugin_id, .. } = &mut row.source {
                        *plugin_id = Some(value.to_string());
                    }
                }
                ("gain", Some(row)) => row.gain = value.parse().unwrap_or(1.0),
                ("lead", Some(row)) => row.lead = read_duration(value)?,
                ("enabled", Some(row)) => row.enabled = value != "0",
                ("autosave", Some(row)) => row.autosave = value != "0",
                ("autoscroll", Some(row)) => row.autoscroll = value != "0",
                ("note", Some(row)) => row.items.push(read_note(value)?),
                _ => {}
            }
        }
        Ok(project)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_text())
            .with_context(|| format!("write {}", crate::file_label(path)))
    }

    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", crate::file_label(path)))?;
        Self::parse(&text).with_context(|| format!("in {}", crate::file_label(path)))
    }

    /// Make every wav source's path absolute: one saved inside the folder is
    /// named relative to it, and everything after loading — the registry, the
    /// missing-file search — wants the file itself.
    pub fn resolve_audio(&mut self, dir: &Path) {
        for row in &mut self.rows {
            if let InstrumentSource::Wav { path } = &mut row.source {
                if path.is_relative() {
                    *path = dir.join(&*path);
                }
            }
        }
    }

    /// Re-point every wav and plugin source whose saved path no longer exists
    /// to an entry of the same name under `wav_dirs` / `plugin_dirs`, searched
    /// in order and up to [`RELOCATE_DEPTH`] levels deep. Returns what moved,
    /// as (instrument name, saved path, found path).
    ///
    /// Those two sources are saved by absolute path (see [`InstrumentSource::Wav`]),
    /// so a project folder copied to another machine — another home directory,
    /// another OS — names files that are not there, though the same recording
    /// sits in that machine's recordings folder and the same plugin bundle in
    /// its plugin folder. Matching by name finds them; a source found nowhere
    /// keeps its saved path, so the row still says what it was looking for. The
    /// next save writes the found path.
    pub fn relocate_missing(
        &mut self,
        wav_dirs: &[PathBuf],
        plugin_dirs: &[PathBuf],
    ) -> Vec<(String, PathBuf, PathBuf)> {
        let mut moved = Vec::new();
        for row in &mut self.rows {
            let (path, dirs) = match &mut row.source {
                InstrumentSource::Wav { path } => (path, wav_dirs),
                InstrumentSource::Vst { path, plugin_id, .. }
                    if crate::plugin::has_file(path, plugin_id.as_deref()) =>
                {
                    (path, plugin_dirs)
                }
                _ => continue,
            };
            if path.exists() {
                continue;
            }
            let Some(name) = path.file_name() else { continue };
            if let Some(found) = find_by_name(dirs, name, RELOCATE_DEPTH) {
                moved.push((row.instrument_name.clone(), path.clone(), found.clone()));
                *path = found;
            }
        }
        moved
    }
}

/// The sub-folder of a project that holds the audio its wav rows play.
pub const AUDIO_DIR: &str = "audio";

/// Make sure the audio file `src` is inside the project folder `dir`, and return
/// the path the manifest should name it by — relative to `dir`.
///
/// A file already somewhere in the folder (a project saved before, and loaded
/// back) is named where it is. Anything else is copied into [`AUDIO_DIR`] under
/// its own name; a different file already holding that name keeps it, and the
/// copy takes `name 2.wav`. A file with the same name *and* the same bytes is the
/// copy an earlier save made, and is reused rather than copied again. `taken`
/// is what this save has already placed, so two files that share a name never
/// land on one copy.
///
/// A file that is gone cannot be copied; its path is returned unchanged, so the
/// manifest still says what the row was looking for.
pub fn bundle_audio(dir: &Path, src: &Path, taken: &mut Vec<PathBuf>) -> Result<PathBuf> {
    let canon_dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let Ok(canon_src) = src.canonicalize() else {
        return Ok(src.to_path_buf());
    };
    if let Ok(inside) = canon_src.strip_prefix(&canon_dir) {
        taken.push(inside.to_path_buf());
        return Ok(inside.to_path_buf());
    }
    let audio = dir.join(AUDIO_DIR);
    std::fs::create_dir_all(&audio)
        .with_context(|| format!("create the folder {}", crate::file_label(&audio)))?;
    let name = src.file_name().context("an audio file with no name")?;
    let stem = Path::new(name).file_stem().unwrap_or(name).to_string_lossy().into_owned();
    let ext = Path::new(name).extension().map(|e| format!(".{}", e.to_string_lossy()));
    let mut n = 1;
    loop {
        let file = match n {
            1 => name.to_string_lossy().into_owned(),
            _ => format!("{stem} {n}{}", ext.as_deref().unwrap_or("")),
        };
        let rel = Path::new(AUDIO_DIR).join(&file);
        n += 1;
        if taken.contains(&rel) {
            continue;
        }
        let full = dir.join(&rel);
        if full.exists() && !same_bytes(&full, src)? {
            continue;
        }
        if !full.exists() {
            std::fs::copy(src, &full).with_context(|| {
                format!("copy {} into {}", crate::file_label(src), crate::file_label(&audio))
            })?;
        }
        taken.push(rel.clone());
        return Ok(rel);
    }
}

/// Whether two files hold the same bytes — the length first, which settles it
/// for nearly every pair without reading either.
fn same_bytes(a: &Path, b: &Path) -> Result<bool> {
    let len = |p: &Path| std::fs::metadata(p).map(|m| m.len());
    if len(a)? != len(b)? {
        return Ok(false);
    }
    Ok(std::fs::read(a)? == std::fs::read(b)?)
}

/// How deep below each directory [`Project::relocate_missing`] looks: enough
/// for a vendor folder in a plugin directory (`VST3/Vendor/Foo.vst3`) or a
/// sub-folder of recordings, as the plugin scan does.
pub const RELOCATE_DEPTH: usize = 3;

/// The first entry named `name` in `dirs`, each searched breadth-first to
/// `depth` levels, so a match near the top wins over one buried deeper.
fn find_by_name(dirs: &[PathBuf], name: &std::ffi::OsStr, depth: usize) -> Option<PathBuf> {
    for dir in dirs {
        let mut level = vec![dir.clone()];
        for _ in 0..=depth {
            let mut next = Vec::new();
            for d in &level {
                let Ok(entries) = std::fs::read_dir(d) else { continue };
                for entry in entries.flatten() {
                    let path = entry.path();
                    if entry.file_name() == name {
                        return Some(path);
                    }
                    if path.is_dir() {
                        next.push(path);
                    }
                }
            }
            level = next;
        }
    }
    None
}

/// Turn a typed project name into something safe to use as a folder and file
/// name: one path component, no separators, no surprises.
pub fn sanitize_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| match c {
            c if c.is_alphanumeric() => c,
            ' ' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect();
    let cleaned = cleaned.trim().trim_matches('.').trim().to_string();
    if cleaned.is_empty() {
        "Untitled".to_string()
    } else {
        cleaned
    }
}

/// A `.lsft` file name for an instrument, unique within `taken`.
pub fn grid_file_name(instrument_name: &str, taken: &[String]) -> String {
    unique_file_name(instrument_name, "lsft", taken)
}

/// A file named after an instrument, with extension `ext`, unique within `taken`.
pub fn unique_file_name(instrument_name: &str, ext: &str, taken: &[String]) -> String {
    let stem = sanitize_name(instrument_name);
    let mut candidate = format!("{stem}.{ext}");
    let mut n = 2;
    while taken.iter().any(|t| t.eq_ignore_ascii_case(&candidate)) {
        candidate = format!("{stem} {n}.{ext}");
        n += 1;
    }
    candidate
}

// ── field encodings ─────────────────────────────────────────────────────────

/// Trim a value to one line: the format's only real constraint on free text.
fn one_line(s: &str) -> String {
    s.replace(['\n', '\r'], " ").trim().to_string()
}

/// Compact but exact enough to round-trip a tempo or a gain.
fn num(v: f32) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() { "0".to_string() } else { s.to_string() }
}

fn write_duration(d: Duration) -> String {
    format!("{} {}", d.wholes, frac_token(d.num, d.frac))
}

fn read_duration(v: &str) -> Result<Duration> {
    let (w, f) = v.split_once(' ').unwrap_or((v, "none"));
    let (num, frac) = frac_from_token(f.trim())?;
    Ok(Duration::with_num(w.trim().parse().unwrap_or(0), num, frac))
}

fn read_note(v: &str) -> Result<Item> {
    let parts: Vec<&str> = v.split_whitespace().collect();
    if !matches!(parts.len(), 5 | 6) {
        bail!(
            "a note takes 5 fields (pitch, note length, space length) and may \
             carry a sixth (where in the file it starts), got {v:?}"
        );
    }
    Ok(Item {
        // Filled in by the loader, which owns row-local ids.
        id: 0,
        pitch: parts[0].parse().context("note pitch")?,
        dur: read_duration(&format!("{} {}", parts[1], parts[2]))?,
        space: read_duration(&format!("{} {}", parts[3], parts[4]))?,
        // A note written before starts existed, or one that starts at the top of
        // its file, has no sixth field — and starts at zero either way.
        start: match parts.get(5) {
            Some(t) => t.parse::<f32>().context("where in the file the note starts")?.max(0.0),
            None => 0.0,
        },
    })
}

/// The fractional part, nominator and all: `1/4`, `3/8`, or `none` where there
/// is no fraction. Plain ASCII, not [`Fraction::label`]'s em dash — this is a
/// file, and it is meant to be typed by hand when a project needs repairing.
///
/// The nominator rides inside the token rather than in a field of its own, so a
/// length that has none writes the token this format has always written and the
/// note line keeps its five fields.
fn frac_token(num: u8, f: Fraction) -> String {
    match f {
        Fraction::None => "none".to_string(),
        f => format!("{num}/{}", f.denom()),
    }
}

/// The nominator and the fraction the token names. A hand-written nominator
/// larger than the select box offers is kept as written: the file says what is
/// playing, and quietly rounding someone's edit down is worse than showing them
/// a value the box did not offer.
fn frac_from_token(t: &str) -> Result<(u8, Fraction)> {
    if matches!(t, "none" | "-" | "0") {
        return Ok((1, Fraction::None));
    }
    let bad = || format!("unknown note fraction {t:?}");
    let (n, d) = t.split_once('/').with_context(bad)?;
    let num: u8 = n.parse().with_context(bad)?;
    if num == 0 {
        bail!(
            "a note fraction is counted at least once: {t:?} — a length with no \
             fraction at all writes \"none\""
        );
    }
    let denom: i64 = d.parse().with_context(bad)?;
    let frac = Fraction::ALL
        .into_iter()
        .find(|f| *f != Fraction::None && f.denom() == denom)
        .with_context(bad)?;
    Ok((num, frac))
}

fn write_source(s: &InstrumentSource) -> String {
    match s {
        InstrumentSource::LeSynth { file, .. } => format!("lesynth {}", one_line(file)),
        InstrumentSource::LeSynthDefault => "lesynth -".to_string(),
        // The state file, if any, rides on its own `state =` line: the path here
        // is the rest of the line, so nothing can follow it.
        InstrumentSource::Vst { path, class_id, .. } => {
            let mut out = format!("vst {}", one_line(&path.display().to_string()));
            if let Some(id) = class_id {
                out = format!("vst:{} {}", hex_class(id), one_line(&path.display().to_string()));
            }
            out
        }
        InstrumentSource::Wav { path } => format!("wav {}", one_line(&path.display().to_string())),
        InstrumentSource::None => "none".to_string(),
    }
}

fn read_source(v: &str) -> Result<InstrumentSource> {
    let (kind, rest) = v.split_once(' ').unwrap_or((v, ""));
    let rest = rest.trim();
    if kind == "none" {
        return Ok(InstrumentSource::None);
    }
    if kind == "lesynth" {
        return Ok(if rest.is_empty() || rest == "-" {
            InstrumentSource::LeSynthDefault
        } else {
            InstrumentSource::LeSynth { file: rest.to_string(), state: None }
        });
    }
    if let Some(hex) = kind.strip_prefix("vst:") {
        return Ok(InstrumentSource::Vst {
            path: PathBuf::from(rest),
            class_id: Some(class_from_hex(hex)?),
            plugin_id: None,
            state: None,
        });
    }
    if kind == "wav" {
        return Ok(InstrumentSource::Wav { path: PathBuf::from(rest) });
    }
    if kind == "vst" {
        return Ok(InstrumentSource::Vst {
            path: PathBuf::from(rest),
            class_id: None,
            plugin_id: None,
            state: None,
        });
    }
    bail!("unknown instrument source {v:?}")
}

fn hex_class(id: &[i8; 16]) -> String {
    id.iter().map(|b| format!("{:02x}", *b as u8)).collect()
}

fn class_from_hex(hex: &str) -> Result<[i8; 16]> {
    if hex.len() != 32 {
        bail!("a VST3 class id is 32 hex digits, got {}", hex.len());
    }
    let mut out = [0i8; 16];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).context("class id")? as i8;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::sequencer::UNITS_PER_WHOLE;

    fn sample() -> Project {
        Project {
            name: "My Song".to_string(),
            tempo_bpm: 132.5,
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            rows: vec![
                ProjectRow {
                    name: "lead".to_string(),
                    instrument_name: "LeSynth Fourier 1".to_string(),
                    source: InstrumentSource::LeSynth { file: "voice.lsft".to_string(), state: None },
                    gain: 0.75,
                    lead: Duration::new(0, Fraction::Eighth),
                    enabled: true,
                    autosave: true,
                    autoscroll: true,
                    items: vec![
                        Item {
                            id: 0,
                            pitch: 60,
                            start: 0.0,
                            dur: Duration::new(0, Fraction::Quarter),
                            space: Duration::new(0, Fraction::Eighth),
                        },
                        Item {
                            id: 0,
                            pitch: 67,
                            start: 0.0,
                            // A nominator on the note and none on the space, so
                            // the round trip proves the field travels rather
                            // than that both ends happen to say `1`.
                            dur: Duration::with_num(2, 3, Fraction::Eighth),
                            space: Duration::new(0, Fraction::None),
                        },
                    ],
                },
                ProjectRow {
                    // A name with spaces in it, and one that is not a UUID:
                    // whatever the user typed is what comes back.
                    name: "the other one".to_string(),
                    instrument_name: "some-plugin.so".to_string(),
                    source: InstrumentSource::Vst {
                        path: PathBuf::from("/opt/vst3/some plugin.so"),
                        class_id: Some([1, 2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16]),
                        plugin_id: None,
                        state: Some("some plugin.vststate".to_string()),
                    },
                    gain: 1.0,
                    lead: Duration::new(1, Fraction::Sixteenth),
                    // Every per-row switch off on this row, so the round trip
                    // proves each carries rather than that `true` survives.
                    enabled: false,
                    autosave: false,
                    autoscroll: false,
                    items: vec![],
                },
            ],
        }
    }

    /// A wav instrument is a path and nothing else: no file beside the manifest, no
    /// state, no class id. The row that plays one must come back naming the same
    /// file — losing it would leave the row silent with nothing to say why.
    #[test]
    fn a_wav_row_round_trips_as_its_path() {
        let mut p = sample();
        p.rows[1].source = InstrumentSource::Wav {
            path: PathBuf::from("/home/kuba/Music/my voice.wav"),
        };
        // And its notes carry where in the file they start — the field only a
        // wav row has anything to say about.
        p.rows[1].items = vec![
            Item {
                id: 0,
                pitch: 60,
                start: 1.25,
                dur: Duration::new(0, Fraction::Quarter),
                space: Duration::new(0, Fraction::Eighth),
            },
            Item {
                id: 0,
                pitch: 60,
                start: 0.0,
                dur: Duration::new(0, Fraction::Quarter),
                space: Duration::new(0, Fraction::None),
            },
        ];
        let text = p.to_text();
        assert!(
            text.contains("source = wav /home/kuba/Music/my voice.wav"),
            "the path is the whole source line: {text}"
        );
        // No `state =` rides behind it: there is nothing beside the manifest to
        // point at.
        assert!(!text.contains("state ="), "a wav instrument saves no state file: {text}");
        // A start writes a sixth field; a note that starts at the top of its
        // file writes the five this format has always written.
        assert!(text.contains("note = 60 0 1/4 0 1/8 1.25\n"), "{text}");
        assert!(text.contains("note = 60 0 1/4 0 none\n"), "{text}");
        assert_eq!(Project::parse(&text).expect("parses"), p);
    }

    /// The start rides on the note line as an optional sixth field, so a project
    /// written before it existed reads as a note at the top of its file — and a
    /// line with one field too many is refused rather than half-read.
    #[test]
    fn a_note_without_a_start_reads_as_the_top_of_the_file() {
        let text = "resynth-project 1\nname = X\n\n\
                    [row]\nsource = none\nnote = 60 0 1/4 0 none\n\
                    note = 62 0 1/4 0 none 0.5\n";
        let p = Project::parse(text).expect("parses");
        assert_eq!(p.rows[0].items[0].start, 0.0);
        assert_eq!(p.rows[0].items[1].start, 0.5);

        assert!(read_note("60 0 1/4 0 none 0.5 7").is_err(), "seven fields is not a note");
        assert!(read_note("60 0 1/4 0 none x").is_err(), "a start is a number of seconds");
        // A negative start is not a place in a file; it reads as the top of one
        // rather than as a note that plays before the file begins.
        assert_eq!(read_note("60 0 1/4 0 none -2").expect("parses").start, 0.0);
    }

    /// The whole point of a save format: what comes back is what went in.
    #[test]
    fn a_project_round_trips_through_the_file() {
        let p = sample();
        let back = Project::parse(&p.to_text()).expect("parses");
        assert_eq!(back, p);
    }

    /// A repeat's window is two lengths from the beginning of the sequence,
    /// and it is saved with it: a loop the user set up is part of the project,
    /// not of the session that happened to be open.
    ///
    /// Written only when there *is* a window, so a project that loops the whole
    /// sequence — which is every project that has not been told otherwise —
    /// is byte for byte the file it always was, and an older build, which skips
    /// keys it does not know, loads one that does and loops the whole thing.
    #[test]
    fn a_repeat_window_is_saved_and_only_when_there_is_one() {
        let plain = sample();
        assert!(
            !plain.to_text().contains("repeat-"),
            "a project with no window must write no window: {}",
            plain.to_text()
        );

        let mut p = sample();
        p.repeat_from = Duration::new(1, Fraction::None);
        p.repeat_to = Duration::with_num(2, 3, Fraction::Eighth);
        let text = p.to_text();
        assert!(text.contains("repeat-from = 1 none"), "{text}");
        assert!(text.contains("repeat-to = 2 3/8"), "{text}");
        assert_eq!(Project::parse(&text).expect("parses"), p);

        // A window reaches far past the 16 whole notes a frame can hold.
        p.repeat_from = Duration::new(2000, Fraction::None);
        p.repeat_to = Duration::new(2048, Fraction::None);
        let text = p.to_text();
        assert!(text.contains("repeat-from = 2000 none"), "{text}");
        assert_eq!(Project::parse(&text).expect("parses"), p);

        // A file written before the keys existed reads as the whole sequence,
        // which is what those projects were played as: no start, and a stop of
        // zero, which the panel reads as the end.
        let old = "resynth-project 1\nname = X\ntempo = 120\n";
        let back = Project::parse(old).expect("parses");
        assert_eq!(back.repeat_from.units(), 0);
        assert_eq!(back.repeat_to.units(), 0);
    }

    /// Paths are the one thing this format exists to carry, and they contain
    /// spaces, `=` and other characters a quoted format would have to escape.
    /// The value is the rest of the line, so none of them need it.
    #[test]
    fn a_path_needs_no_escaping() {
        let mut p = sample();
        p.rows[1].source = InstrumentSource::Vst {
            path: PathBuf::from("/opt/vst3/weird = name (v2) [x86].so"),
            class_id: None,
            plugin_id: None,
            state: Some("weird = name.vststate".to_string()),
        };
        p.name = "Song = 2".to_string();
        let back = Project::parse(&p.to_text()).expect("parses");
        assert_eq!(back, p);
    }

    /// The nominator rides in the fraction token. A length without one writes
    /// the token this format has always written — that is what keeps old files
    /// and old builds working — and one with it writes the fraction whole.
    #[test]
    fn a_length_writes_its_nominator_into_the_fraction() {
        assert_eq!(write_duration(Duration::new(0, Fraction::Quarter)), "0 1/4");
        assert_eq!(write_duration(Duration::with_num(2, 3, Fraction::Eighth)), "2 3/8");
        assert_eq!(write_duration(Duration::new(1, Fraction::None)), "1 none");

        // And back, including the tokens written before nominators existed.
        for (text, units) in [
            ("0 1/4", UNITS_PER_WHOLE / 4),
            ("0 3/8", UNITS_PER_WHOLE * 3 / 8),
            ("2 3/8", UNITS_PER_WHOLE * 2 + UNITS_PER_WHOLE * 3 / 8),
            ("1 none", UNITS_PER_WHOLE),
            ("0 none", 0),
            ("0 16/256", UNITS_PER_WHOLE / 16),
        ] {
            let d = read_duration(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(d.units(), units, "{text:?}");
        }

        // A fraction that is no fraction at all is `none`, not a zero nominator:
        // the file is refused rather than read as some other length.
        assert!(read_duration("0 0/8").is_err());
        assert!(read_duration("0 1/7").is_err());
        assert!(read_duration("0 3").is_err());
    }

    /// A row saved before a per-row switch existed reads as the default a new
    /// row gets, not as `false`. The format skips keys it does not know, which
    /// is exactly how a missing one arrives.
    #[test]
    fn a_row_without_the_newer_switches_takes_their_defaults() {
        let text = "resynth-project 1\nname = X\n\n\
                    [row]\nsource = none\ngain = 1\nnote = 60 0 1/4 0 none\n";
        let p = Project::parse(text).expect("parses");
        assert!(p.rows[0].autosave, "autosave should default on");
        assert!(p.rows[0].autoscroll, "a lane should follow the transport by default");
        assert!(p.rows[0].enabled, "a row saved before the switch existed must still play");
    }

    /// A field this build does not know must not stop it loading — that is what
    /// lets an older ReSynth Studio open a newer project's rows.
    #[test]
    fn an_unknown_field_is_skipped_not_fatal() {
        let text = "resynth-project 1\nname = X\nsomething_new = 4\n\n\
                    [row]\nsource = none\nfuture_field = yes\nnote = 60 0 1/4 0 none\n";
        let p = Project::parse(text).expect("parses");
        assert_eq!(p.name, "X");
        assert_eq!(p.rows.len(), 1);
        assert_eq!(p.rows[0].items.len(), 1);
    }

    /// A newer format is refused outright rather than half-read, because a row
    /// silently missing its notes is worse than a message.
    #[test]
    fn a_newer_format_is_refused() {
        let err = Project::parse("resynth-project 99\nname = X\n").unwrap_err().to_string();
        assert!(err.contains("newer ReSynth Studio"), "{err}");
        assert!(Project::parse("something else\n").is_err());
    }

    /// A project saved as Gemstone DAW opens as it always did, and saving it
    /// again writes the new first line.
    #[test]
    fn a_gemstone_project_still_opens_and_saves_as_resynth() {
        let old = "gemstone-project 1\nname = Old\ntempo = 96\n\n\
                   [row]\nsource = none\nnote = 60 0 1/4 0 none\n";
        let p = Project::parse(old).expect("a Gemstone DAW project parses");
        assert_eq!(p.name, "Old");
        assert_eq!(p.tempo_bpm, 96.0);
        assert_eq!(p.rows.len(), 1);
        let text = p.to_text();
        assert!(text.starts_with("resynth-project 1\n"), "{text}");
        assert_eq!(Project::parse(&text).unwrap(), p);
        assert!(Project::parse("gemstone-project 99\nname = X\n").is_err());
    }

    #[test]
    fn every_source_kind_round_trips() {
        for src in [
            InstrumentSource::None,
            InstrumentSource::LeSynthDefault,
            InstrumentSource::LeSynth { file: "a b.lsft".to_string(), state: None },
            InstrumentSource::LeSynth {
                file: "a b.lsft".to_string(),
                state: Some("a b.vststate".to_string()),
            },
            InstrumentSource::Vst {
                path: PathBuf::from("/x/y.so"),
                class_id: None,
                plugin_id: None,
                state: None,
            },
            InstrumentSource::Vst {
                path: PathBuf::from("/x/y.so"),
                class_id: Some([-1i8; 16]),
                plugin_id: None,
                state: None,
            },
            InstrumentSource::Vst {
                path: PathBuf::from("/x/y.so"),
                class_id: None,
                plugin_id: None,
                state: Some("y.vststate".to_string()),
            },
            // A CLAP file holding several plugins, and an LV2 bundle: the
            // plugin inside rides on its own line, `=` and spaces and all.
            InstrumentSource::Vst {
                path: PathBuf::from("/x/Surge XT.clap"),
                class_id: None,
                plugin_id: Some("org.surge-synth-team.surge-xt".to_string()),
                state: Some("Surge XT.vststate".to_string()),
            },
            InstrumentSource::Vst {
                path: PathBuf::from("/x/Kars.lv2"),
                class_id: None,
                plugin_id: Some("http://distrho.sf.net/plugins/Kars?a=b c".to_string()),
                state: None,
            },
            InstrumentSource::Wav { path: PathBuf::from("/x/a whole take.wav") },
        ] {
            let mut p = sample();
            p.rows.truncate(1);
            p.rows[0].source = src.clone();
            let back = Project::parse(&p.to_text()).expect("parses");
            assert_eq!(back.rows[0].source, src);
        }
    }

    /// A project written before a LeSynth instrument kept its plugin state has no
    /// `state =` line under `source = lesynth …`. It must still load — with the
    /// grid, and with the controls that drew it left at their defaults — rather
    /// than being refused or half-read.
    #[test]
    fn a_lesynth_row_without_a_state_line_still_loads() {
        let text = "resynth-project 1\nname = X\ntempo = 120\n\n[row]\n\
                    instrument = LeSynth Fourier 1\nsource = lesynth voice.lsft\n\
                    gain = 1\nlead = 0 none\n";
        let p = Project::parse(text).expect("parses");
        assert_eq!(
            p.rows[0].source,
            InstrumentSource::LeSynth { file: "voice.lsft".to_string(), state: None }
        );
    }

    /// A project name is a folder name, so it must survive being typed.
    #[test]
    fn a_name_becomes_one_safe_path_component() {
        assert_eq!(sanitize_name("My Song"), "My Song");
        assert_eq!(sanitize_name("a/b"), "a_b");
        assert_eq!(sanitize_name("  "), "Untitled");
        assert_eq!(sanitize_name(""), "Untitled");
        // Nothing that escapes the folder, starts a hidden one, or ends the
        // string somewhere a filesystem will not follow.
        for hostile in ["../etc", "..", "/", ".hidden", "a\0b", "c:\\x", "  ..  "] {
            let got = sanitize_name(hostile);
            assert!(!got.is_empty(), "{hostile:?} sanitised to nothing");
            assert!(!got.starts_with('.'), "{hostile:?} -> {got:?} is hidden");
            assert!(
                !got.contains(['/', '\\', '\0']),
                "{hostile:?} -> {got:?} still has a separator"
            );
            assert_eq!(std::path::Path::new(&got).components().count(), 1, "{got:?}");
        }
    }

    /// Two instruments may share a name; their grids may not share a file.
    #[test]
    fn grid_file_names_do_not_collide() {
        let mut taken = Vec::new();
        for want in ["Voice.lsft", "Voice 2.lsft", "Voice 3.lsft"] {
            let got = grid_file_name("Voice", &taken);
            assert_eq!(got, want);
            taken.push(got);
        }
    }
}

#[cfg(test)]
mod folder_tests {
    use super::*;
    use std::fs;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rsyn-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A project folder is the unit that moves: the manifest names its grids by
    /// a *relative* file, so copying the folder somewhere else still resolves.
    #[test]
    fn a_saved_folder_carries_its_own_grids() {
        let dir = tmp("portable");
        let p = Project {
            name: "Song".to_string(),
            tempo_bpm: 100.0,
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            rows: vec![ProjectRow {
                name: "Voice row".to_string(),
                instrument_name: "Voice".to_string(),
                source: InstrumentSource::LeSynth { file: "Voice.lsft".to_string(), state: None },
                gain: 1.0,
                lead: Duration::new(0, Fraction::None),
                enabled: true,
                autosave: true,
                autoscroll: true,
                items: vec![],
            }],
        };
        let manifest = dir.join(format!("Song.{EXTENSION}"));
        p.write(&manifest).unwrap();
        fs::write(dir.join("Voice.lsft"), b"not a real grid").unwrap();

        let moved = tmp("portable-moved");
        for f in fs::read_dir(&dir).unwrap() {
            let f = f.unwrap();
            fs::copy(f.path(), moved.join(f.file_name())).unwrap();
        }
        let back = Project::read(&moved.join(format!("Song.{EXTENSION}"))).unwrap();
        let want = back.rows[0].source.required_path(&moved).unwrap();
        assert!(want.exists(), "{} did not follow the folder", want.display());
        assert_eq!(back, p);

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&moved);
    }

    /// The requirement: a grid that has been deleted must be *detectable*, and
    /// the manifest must still say what it was, so the row can ask for a
    /// replacement instead of silently going quiet.
    #[test]
    fn a_deleted_grid_is_detected_and_still_named() {
        let dir = tmp("missing");
        let p = Project {
            name: "Song".to_string(),
            tempo_bpm: 120.0,
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            rows: vec![ProjectRow {
                name: "Voice row".to_string(),
                instrument_name: "Voice".to_string(),
                source: InstrumentSource::LeSynth { file: "Voice.lsft".to_string(), state: None },
                gain: 1.0,
                lead: Duration::new(0, Fraction::None),
                enabled: true,
                autosave: true,
                autoscroll: true,
                items: vec![],
            }],
        };
        p.write(&dir.join(format!("Song.{EXTENSION}"))).unwrap();
        // The grid was never written — the same state as a user deleting it.
        let back = Project::read(&dir.join(format!("Song.{EXTENSION}"))).unwrap();
        let wanted = back.rows[0].source.required_path(&dir).unwrap();
        assert!(!wanted.exists());
        assert_eq!(back.rows[0].source.describe(), "Voice.lsft");
        assert_eq!(back.rows[0].instrument_name, "Voice");

        let _ = fs::remove_dir_all(&dir);
    }

    /// A missing custom VST names its path, which is the only thing that can
    /// help — the plugin is not ours to have copied into the folder.
    #[test]
    fn a_missing_vst_names_its_path() {
        let src = InstrumentSource::Vst {
            path: PathBuf::from("/nowhere/plugin.so"),
            class_id: None,
            plugin_id: None,
            state: Some("plugin.vststate".to_string()),
        };
        assert_eq!(src.required_path(Path::new("/x")).unwrap(), PathBuf::from("/nowhere/plugin.so"));
        assert!(src.describe().contains("plugin.so"));
    }

    fn row_with(source: InstrumentSource) -> ProjectRow {
        ProjectRow {
            name: String::new(),
            instrument_name: "T".to_string(),
            source,
            gain: 1.0,
            lead: Duration::new(0, Fraction::None),
            enabled: true,
            autosave: true,
            autoscroll: true,
            items: vec![],
        }
    }

    /// A project copied from another machine names a recording and a plugin
    /// bundle by that machine's absolute paths. Both are found by name — the
    /// plugin in a vendor folder — and a source that exists, or that is found
    /// nowhere, keeps the path it was saved with.
    #[test]
    fn a_moved_project_finds_its_wavs_and_plugins_by_name() {
        let dir = tmp("relocate");
        let recordings = dir.join("ReSynthRecordings");
        let plugins = dir.join("VST3");
        fs::create_dir_all(&recordings).unwrap();
        fs::create_dir_all(plugins.join("Vendor/Drums.vst3/Contents")).unwrap();
        fs::write(recordings.join("take.wav"), b"").unwrap();
        let present = dir.join("present.wav");
        fs::write(&present, b"").unwrap();

        let vst = |path: &str| InstrumentSource::Vst {
            path: PathBuf::from(path),
            class_id: None,
            plugin_id: None,
            state: None,
        };
        let mut p = Project {
            name: "Song".to_string(),
            tempo_bpm: 100.0,
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            rows: vec![
                row_with(InstrumentSource::Wav { path: PathBuf::from("/home/other/ReSynthRecordings/take.wav") }),
                row_with(vst("/home/other/.vst3/Drums.vst3")),
                row_with(InstrumentSource::Wav { path: present.clone() }),
                row_with(InstrumentSource::Wav { path: PathBuf::from("/home/other/lost.wav") }),
                // A wav is not looked for among plugins, nor a plugin among wavs.
                row_with(vst("/home/other/take.wav")),
            ],
        };
        let moved = p.relocate_missing(&[recordings.clone()], &[plugins.clone()]);

        assert_eq!(moved.len(), 2, "{moved:?}");
        assert_eq!(p.rows[0].source, InstrumentSource::Wav { path: recordings.join("take.wav") });
        assert_eq!(p.rows[1].source, vst(&plugins.join("Vendor/Drums.vst3").display().to_string()));
        assert_eq!(p.rows[2].source, InstrumentSource::Wav { path: present });
        assert_eq!(p.rows[3].source, InstrumentSource::Wav { path: PathBuf::from("/home/other/lost.wav") });
        assert_eq!(p.rows[4].source, vst("/home/other/take.wav"));

        let _ = fs::remove_dir_all(&dir);
    }

    /// A recording saved into a project is copied into its `audio/` folder,
    /// once: a second save reuses the copy, a different file with the same name
    /// gets a name of its own, and a file already in the folder is named where
    /// it is. The manifest names them all relative, so the folder moves whole.
    #[test]
    fn a_saved_project_carries_its_own_recordings() {
        let dir = tmp("bundle");
        let project = dir.join("Song");
        fs::create_dir_all(&project).unwrap();
        let elsewhere = dir.join("ReSynthRecordings");
        let other = dir.join("other");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::create_dir_all(&other).unwrap();
        fs::write(elsewhere.join("take.wav"), b"first take").unwrap();
        fs::write(other.join("take.wav"), b"another take").unwrap();

        let mut taken = Vec::new();
        let a = bundle_audio(&project, &elsewhere.join("take.wav"), &mut taken).unwrap();
        let b = bundle_audio(&project, &other.join("take.wav"), &mut taken).unwrap();
        assert_eq!(a, Path::new(AUDIO_DIR).join("take.wav"));
        assert_eq!(b, Path::new(AUDIO_DIR).join("take 2.wav"));
        assert_eq!(fs::read(project.join(&a)).unwrap(), b"first take");
        assert_eq!(fs::read(project.join(&b)).unwrap(), b"another take");

        // The next save, in the other order: each finds its own copy.
        let mut taken = Vec::new();
        let b2 = bundle_audio(&project, &other.join("take.wav"), &mut taken).unwrap();
        let a2 = bundle_audio(&project, &elsewhere.join("take.wav"), &mut taken).unwrap();
        assert_eq!((a2, b2), (a.clone(), b.clone()));
        assert_eq!(fs::read_dir(project.join(AUDIO_DIR)).unwrap().count(), 2);

        // Already inside: named where it is.
        let mut taken = Vec::new();
        let inside = bundle_audio(&project, &project.join(&a), &mut taken).unwrap();
        assert_eq!(inside, a);

        // Gone: nothing to copy, and the path still says what it was.
        let gone = dir.join("gone.wav");
        assert_eq!(bundle_audio(&project, &gone, &mut Vec::new()).unwrap(), gone);

        // Through the manifest, and back from somewhere else.
        let mut p = Project {
            name: "Song".to_string(),
            tempo_bpm: 100.0,
            repeat_from: Duration::new(0, Fraction::None),
            repeat_to: Duration::new(0, Fraction::None),
            rows: vec![row_with(InstrumentSource::Wav { path: a.clone() })],
        };
        p.write(&project.join(format!("Song.{EXTENSION}"))).unwrap();
        fs::remove_dir_all(&elsewhere).unwrap();
        let moved = dir.join("Moved");
        fs::rename(&project, &moved).unwrap();
        p = Project::read(&moved.join(format!("Song.{EXTENSION}"))).unwrap();
        p.resolve_audio(&moved);
        let InstrumentSource::Wav { path } = &p.rows[0].source else { panic!() };
        assert_eq!(fs::read(path).unwrap(), b"first take");

        let _ = fs::remove_dir_all(&dir);
    }

    /// An Audio Unit has no file a project could find missing: the system's
    /// component registry is what finds it, by the codes in its id.
    #[test]
    fn an_audio_unit_needs_no_file() {
        let src = InstrumentSource::Vst {
            path: PathBuf::from(crate::plugin::AU_PATH),
            class_id: None,
            plugin_id: Some("au:aumu:dls :appl".to_string()),
            state: None,
        };
        assert_eq!(src.required_path(Path::new("/x")), None);
    }
}
