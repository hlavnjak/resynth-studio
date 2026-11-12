# ReSynth Studio

ReSynth Studio is a small DAW written in **Rust** with an `egui`/`eframe` GUI. It
hosts **VST3, CLAP, VST2 and LV2** plugins (and **Audio Units** on macOS) and
ships with an embedded **[LeSynth Fourier](https://github.com/hlavnjak/lesynth-fourier)** additive
synthesizer, which powers its headline feature: turning any audio file into a
playable, editable additive-synthesis instrument.

ReSynth Studio and LeSynth Fourier are companion projects: LeSynth Fourier
([github.com/hlavnjak/lesynth-fourier](https://github.com/hlavnjak/lesynth-fourier))
is the VST3/CLAP plugin that does the analysis and synthesis, and ReSynth Studio
is the host that feeds it recordings.

<p align="center">
  <img src="assets/track_0.2.0_1.png" alt="ReSynth Studio 0.2.0: Instruments panel and Sequencer rows" width="800">
</p>
<p align="center">
  <img src="assets/track_0.2.0_2.png" alt="ReSynth Studio 0.2.0: Sequencer transport and repeat window, MIDI and Resynthesis panels" width="800">
</p>

## Headline feature — Resynthesis

The **Resynthesis** panel (`.wav` / `.mp3` / `.m4a` → LeSynth Fourier) takes an
arbitrary recording and rebuilds it as additive synthesis you can play and edit:

1. **Pick a file** — any `.wav`, `.mp3`, or `.m4a`.
2. **Segment** — the audio is split host-side into pitch-stable *subtracks*.
3. **Analyse** — each usable subtrack is handed to LeSynth Fourier, which
   subdivides it into per-period *buckets* and extracts an amplitude/phase value
   for every harmonic in every bucket (the FFT step). Pitch contours (vibrato)
   are tracked, so the analysis follows the note rather than smearing it.
4. **Play & edit** — the analysed grid is previewed inline ("Preview FFT") and
   can be opened in a full LeSynth Fourier editor instance ("Open in LeSynth")
   running in Analysis mode, where individual harmonics can be toggled and the
   result played back on the keyboard.

A file can also skip all of it. **"Add whole file as Instrument"** publishes the
recording *itself* — not an analysis of it — as a **wav instrument**: an Instrument a
Sequencer row plays for the length of one note. There is no pitch to pick, because
a recording has the pitch it was made at; what a note frame carries in the pitch
box's place is a **start** — a slider saying where in the file the note begins,
which is how a take gets lined up with the beat. It is what a take no
additive analysis does justice to is for — a drum loop, a spoken line, a whole
performance — and a project saves it as the path to the file it was added from,
so the file stays where it is rather than being copied into the project folder.

## Sequencer

Arrange the instruments in rows of frames and play them together. Each row plays
exactly one Instrument — a LeSynth Fourier instrument, a custom plugin instrument, a subtrack
published from the Resynthesis panel with "Add as Instrument", or a whole audio file
published there as a wav instrument — and any number of rows may share the same one.

- **Rows** are added by hand ("➕ Add Instrument Row"); a fresh instance starts empty.
  The select box at the head of a row picks its Instrument, and follows the instrument
  list: delete the instrument a row is playing and the row moves to another one, or
  to a placeholder while no instruments exist.
- **Enabled — a lane that is silent, not gone.** Every row carries an
  **Enabled** box, ticked on every new row. Untick it and the row is left out of
  the transport and out of an exported wav, while everything on it stays exactly
  where it is: the notes, the block, the instrument it plays, the length it gives the
  sequence. That is how a part is heard without the two rows drowning it, and
  how they come back afterwards — deleting the rows and building them again is
  the alternative, and it loses the work. **Enable all**, beside "➕ Add Instrument
  Row", is the same switch for every row at once: it is unticked while any row
  is off, and a click makes them all agree with it. The setting is saved with
  the project; a project written before it existed loads with every row playing,
  which is what those rows did. A switched-off lane is also left alone by the
  transport: no frame lit as sounding, and no scrolling under your hands —
  "⏵ follow" has nothing to follow, so it is greyed out until the row is back on.
  Editing is untouched, blocks and cloning included: the row is silent, not
  read-only.
- **A row is a sequence of frames**, laid left to right and simply played one
  after another: nothing is positioned by hand, nothing is dragged, and nothing
  can overlap. Deleting a frame pulls everything behind it forward.
- **Drum instruments name their notes.** On a row playing a plugin recognised as a
  drum kit, the pitch box reads `C2 · Bass Drum (Kick)` rather than `C2`, and a
  new note starts on the kick instead of middle C. The names are General MIDI's,
  which is the map nearly every kit follows; they are added to the pitch, never
  swapped for it, so a kit laid out some other way still shows what is sent. The
  map and the rule that spots a drum plugin are in `src/midi/drums.rs`.
- **Notes and spaces.** "➕ Add Note" appends two frames: a **note** frame
  (blue), and behind it a **space** frame (amber) which is the silence that
  follows. A note frame carries a pitch select box (`C0`–`B8`) and three
  boxes its length is typed into: the whole notes (0–1024), then the fractional
  part written as a fraction, `+ nominator / denominator` (0–255 over 1–256). A
  space frame carries the same three length boxes; it has no pitch — and neither does a note on a **wav row**, which plays
  its file for as long as its length says. That frame is wider than the rest, and
  carries a **start** slider where a pitch would be: a take does not begin on its
  first sample, so the start is what puts the sound on the beat instead of the
  room tone in front of it. Drag the slider to find it, drag the number beside it
  for milliseconds, or double-click to type one. The **⏱** button beside the
  frame's ✖ sets the note's length to what the file has left to play from that
  start, as closely as the boxes can name it *at the current tempo* — its hover
  text says how close, since a recording's length and a whole number of note
  values agree only by luck. Change the BPM and press it again. A length is the
  parts added together, so `1 whole + 1 / 8` is a whole note tied to an eighth,
  `+ 3 / 8` is a dotted quarter and `+ 1 / 3` a triplet half. Any denominator can
  be typed; a nominator of `0` is a length with no fractional part.
- **A space belongs to its note.** It has no delete button of its own: the ✖ on
  a note frame removes the note *and* its space, and there is no way to remove
  one without the other. Setting a space to `0 whole + 0` is the way to run
  two notes together — the frame stays as a placeholder, but adds no time.
- **Blocks — copying a section across every instrument at once.** Music repeats, and
  the second time round should not cost what the first did. Each note frame
  carries a select box (`☐`): click it to start a **block**, click another frame
  on the row to take everything between them, and click one that is already in
  the block to drop it. Shift-click moves the end last clicked, which is how a
  block shrinks. The **select all** box on the row head takes the whole row in
  one press, however long it is — a part of two hundred frames is otherwise a
  scroll to its end and back, and a whole part is the commonest block there is.
  Rows are selected independently, so a block spans as many of them as you like —
  or mark the phrase on the row you can hear it in and press **↔ Span Rows**,
  which takes *the same stretch of time* out of every other row. That is what
  makes a natural block one press: a section is a stretch of time, not a count of
  frames, and no two rows spend it on the same number of notes.
  - **🔁 Clone ×N** repeats the block in place — each copy directly behind the
    last, on every row at once, with everything that followed moved along by the
    same amount on every row. A copy occupies the block's **window** (from its
    first note to the end of its last space), not the frames' own extent, so the
    silence a row leaves at either end of the window is put back between one copy
    and the next and the instruments stay in step however many times it is pressed.
    The selection moves onto the copy, so pressing it again repeats the repeat.
  - **🗐 Copy** and **📋 Paste at End** carry a block, with the silence in front
    of each row's part of it, to the end of the sequence — every row padded up
    to the end of the longest one, so a block that sounded together is pasted
    sounding together. It goes back on the rows it came from: the frames name a
    pitch, a length and a place in a file that mean what they mean on the instrument
    they were written for.
  - A gap between copies is arithmetic, not a length anyone picked, so it is not
    always one the three length boxes can name — `255/256` needs 255 of the
    smallest fraction there is. What will not fit in one frame is laid out over
    frames of no length: silence, which is what a gap is, and which the transport
    skips. A rounded gap would be a copy that drifts further out of step with
    every repeat.
- **Time**: every row starts at zero, so two rows sound together exactly when the
  lengths in front of their frames add up the same — that is what makes a chord.
  The tempo control (BPM) sets what a beat is worth — a beat is a quarter note.
- **Repeat** loops the arrangement for as long as it is ticked, on the written
  length — the longest row, trailing silence included — rather than on the last
  note, so the bar structure survives the wrap and releases ring on across it.
  It can be switched on and off while the transport runs; unticking it makes the
  pass in flight the last one.
- **The loop can be trimmed at either end.** Two more note lengths, on the line
  under the transport, say where a repeat starts and where it stops — each
  counted *from the beginning of the sequence*, in the same three select
  boxes a note is written with. Both from the same end so the pair can be read
  against each other: `1/2` and `7/4` say which is in front of which at a
  glance. A stop left at zero means the end of the arrangement, and one asked
  for past the end is held there. Everything in front of the start still plays,
  once, on the way in; the line reads back what the two lengths came to in
  seconds, and says so when they leave nothing between them. The window is
  saved with the project.
- **A loop follows your edits.** While Repeat is on, changing notes — pitch,
  length, adding or deleting them — or a row's gain, the tempo or the repeat
  window is picked up at the next time round, without stopping. Adding a row, pointing one at another
  instrument, or switching one off is the exception: each changes which plugins are
  loaded, which cannot happen while the audio callback is running, so the panel
  says to press Play again.
- **Play / Stop** at the foot of the section renders the whole arrangement in
  real time, lighting up the frame each row is sounding as it goes: one output
  stream, one plugin instance per row, mixed with each row's own gain. The
  Sequencer loads its own instances, so a sequence plays whether or not the
  instruments' editors are open — and when one is open, its live grid is what plays.
  **Space** does the same: it plays when nothing is playing and stops whatever
  is (unless a text box is being typed in).
- **Export WAV** writes the whole sequence to a 16-bit `.wav` at the output
  device's rate and channel count. It renders offline, through the same voices,
  schedule and mix the transport uses, so the file is what Play sounds like; the
  render runs in the background and the panel stays usable while it works.

## Other features

- **Load Internal plugin** — loads the embedded LeSynth Fourier VST3
  (`internal_plugins/liblesynth_fourier.so`, committed precompiled) by its class
  ID. No separate plugin install required.
- **A wav instrument travels with the project.** Saving copies the file into the
  project folder's `audio/` and names it there, so a project keeps its
  recordings when `~/ReSynthRecordings` (or wherever the file came from) is
  cleaned out. A row playing one needs no plugin at all: the
  Sequencer decodes the file once, however many rows name it, and plays it under
  every note from that note's own start, cut to the note's length and faded a few
  milliseconds at each end so a cut cannot click. A start past the end of the
  file is simply silent. It is mixed through the same
  voices and the same per-row gain as everything else, so "Export WAV" writes
  exactly what the transport plays.
- **Plugin sounds are kept.** What a plugin is playing — the knobs set in
  its own editor — is taken from the instance when its editor closes, handed to
  the Sequencer's instances, and written into the project folder as a
  `.vststate` beside the manifest. It travels over `IComponent::getState`, the
  mechanism every VST3 has, rather than LeSynth's grid ABI, which only ours does.
  A **LeSynth** instrument keeps one too, beside its `.lsft`, and needs both: the
  `.lsft` is the harmonic grid, and the `.vststate` is every control that drew
  it — each harmonic's curve type, offset and granularity, and the 32
  nested-Fourier amplitude/phase sliders and base frequency under them, none of
  which is in the grid. Saving only the grid reloaded a correct picture over
  sliders that all read zero.
- **External plugins, in every common format** — "Create Custom Plugin Instrument"
  lists what is installed, filterable by format, and can browse for a bundle or
  a plugin file anywhere else. Every format plays notes, renders in the
  Sequencer and an exported `.wav`, keeps its knobs in the instrument (and in a saved
  project, as the plugin's own state beside the manifest), and shows its own
  editor in a native window (raw X11 on Linux, raw Win32 on Windows, an
  NSWindow on macOS):

  | Format | Linux | Windows | macOS | Looked for in |
  |---|---|---|---|---|
  | **VST3** | ✓ | ✓ | ✓ | `VST3_PATH`, `~/.vst3`, `/usr/lib/vst3`; `%CommonProgramFiles%\VST3`; `~/Library/Audio/Plug-Ins/VST3` |
  | **CLAP** | ✓ | ✓ | ✓ (no editor on macOS yet) | `CLAP_PATH`, `~/.clap`, `/usr/lib/clap`; `%CommonProgramFiles%\CLAP`; `~/Library/Audio/Plug-Ins/CLAP` |
  | **VST2** | ✓ | ✓ | ✓ | `VST_PATH`, `~/.vst`, `/usr/lib/vst`, `/usr/lib/lxvst`; `%ProgramFiles%\VSTPlugins`, `…\Steinberg\VSTPlugins`; `~/Library/Audio/Plug-Ins/VST` |
  | **LV2** | ✓ | ✓ | ✓ | `LV2_PATH`, `~/.lv2`, `/usr/lib/lv2`; `%APPDATA%\LV2`; `~/Library/Audio/Plug-Ins/LV2` |
  | **Audio Unit** (v2) | — | — | ✓ (no editor on macOS yet) | the system's component registry |

  Each format is in `src/plugin/` (VST3 in `src/vst/`), behind one
  `PluginInstance`, one `BlockProcessor` that drives any of them a block at a
  time, and one `PluginEditor` the editor window embeds. What each needed that
  is easy to get wrong:
  - **VST3** — a bundle is resolved to the library inside it, `ModuleEntry` is
    run, and the X11 window provides the `IPlugFrame` and `Linux::IRunLoop` a
    JUCE or Steinberg-SDK editor needs in order to draw at all.
  - **CLAP** — every instance has a main thread of its own: JUCE-built CLAPs
    (Surge XT) deadlock if a GUI call comes from any thread but the one that
    created them. That thread also fires the plugin's timers, watches its file
    descriptors and, on Windows, pumps its windows' messages.
  - **VST2** — no SDK: the `AEffect` ABI is declared clean-room. State is the
    plugin's chunk where it has one, every parameter's value otherwise.
  - **LV2** — no lilv: a small Turtle reader (`src/plugin/lv2/turtle.rs`) reads
    the bundle, so the Windows and macOS builds carry no C libraries. URID map,
    options, a worker and a log are provided; a plugin requiring anything else
    is refused by name. Editors are the plugin's own UI library (X11UI /
    WindowsUI), given instance access only when they ask for it.
  - **Audio Unit** — found by its type/subtype/manufacturer codes, which a
    instrument records as its plugin id (`au:aumu:dls :appl`); state is the unit's
    `ClassInfo` property list.

  Anything that is not loadable — a library with a missing dependency, a VST2
  shell, an LV2 needing an unsupported feature — is reported when the plugin is
  picked, not when its editor opens. `cargo run --bin plugin_probe -- <plugin>`
  says what the host sees in any plugin (`--list` prints everything the picker
  would offer); `tests/third_party_plugins.rs` loads and renders what is
  installed, in every format.
- **MIDI input** — pick a USB keyboard / port, connect, refresh; **Plays instrument**
  gives the keyboard to one instrument instead of every open one.
- **Logging** to `resynth-studio.log`.
- **Formerly Gemstone DAW.** Projects saved under the old name (`.gmstn`) still
  open; saving one writes a `.rsyn` beside it. Takes recorded into
  `~/GemstoneRecordings` are still found; new ones go to `~/ReSynthRecordings`.

## Project layout

```
Cargo.toml          # egui/eframe app crate
build.rs            # links the icon resource into the Windows .exe
Makefile            # builds the app; (re)builds + embeds the VST3 when its source is present
.cargo/config.toml  # Windows cross linker
packaging/macos/    # Info.plist template for the macOS app bundle
packaging/icon/     # the app icon (SVG, PNG, .ico, .icns) and make_icon.py, which draws it
packaging/windows/  # the .exe's icon resource, compiled by build.rs with windres
tools/zig-macos-cc.sh  # macOS cross linker (zig, honouring the minimum macOS)
tools/macos-sign*.sh   # sign + notarise the app on a Mac over ssh (make sign-macos)
tools/macos-release-zip.sh, tools/third-party-licenses.py  # make release
packaging/release/   # release README parts, licence-notice header, standard licence texts
internal_plugins/   # the embedded LeSynth Fourier VST3 (committed precompiled)
src/
  main.rs                                 # eframe entry point
  lib.rs
  plugin/                                 # one plugin, any format:
    mod.rs, processor.rs, editor.rs, scan.rs  #   instance, block driver, editor, scan
    clap.rs, vst2.rs, au.rs               #   CLAP, VST2, Audio Unit (macOS) hosting
    lv2/{mod,ui,turtle}.rs                #   LV2 hosting, its UIs, its Turtle reader
  vst/{host,module,editor,realtime,…}.rs  # VST3 hosting
  bin/plugin_probe.rs                     # `cargo run --bin plugin_probe -- <plugin>`:
                                          #   what the host sees in a plugin, and why
                                          #   one will not load
  audio/{engine,decode,mod}.rs            # cpal audio engine + audio file decoding
  midi/{input,mod}.rs                     # midir MIDI input
  analysis/{segmentation,mod}.rs          # host-side subtrack segmentation
  gui/
    app.rs            # top-level egui app
    registry.rs       # the shared list of Tracks every panel agrees on
    instrument.rs          # the Instruments panel
    sequencer/         # the Sequencer panel + its playback engine
    resynth.rs        # the Resynthesis panel
    editor_window/    # embedded plugin editor window (x11 / windows / fallback)
```

## Requirements

- A recent **Rust** toolchain.
- Linux: an X11 or Wayland display, plus GTK3 (used by the file-open dialog).

No copy of the Steinberg VST3 SDK is needed: the `vst3` crate ships pre-generated
bindings, so nothing is scraped from the SDK headers at build time.

## Build & run (Linux)

```sh
make run        # build and run the DAW
make build      # build only
make clean      # cargo clean (keeps the committed plugin binaries)
make clean-all  # also clean ../lesynth-fourier if it is checked out alongside
```

The binary lands at `target/release/resynth-studio`. The app is GUI-only and needs
a display. The embedded plugin is loaded from the committed
`internal_plugins/liblesynth_fourier.so`, so you do **not** need the
lesynth-fourier source tree just to run ReSynth Studio.

### Rebuilding the embedded plugin (developers)

If you have the [`lesynth-fourier`](https://github.com/hlavnjak/lesynth-fourier) source checked out next to this repo
(`../lesynth-fourier`), `make` automatically rebuilds it and refreshes
`internal_plugins/` when its sources change. Otherwise the committed binary is
used as-is.

```sh
make fourier  # build only the LeSynth Fourier VST3 (requires ../lesynth-fourier)
```

## Windows cross build (from Linux via mingw)

Requires the mingw toolchain (`x86_64-w64-mingw32-gcc`/`g++`) and
`rustup target add x86_64-pc-windows-gnu`.

```sh
make build-windows    # build the app + embed the VST3 .dll
make fourier-windows  # build only the VST3 plugin for Windows
```

Output: `target/x86_64-pc-windows-gnu/release/resynth-studio.exe`.

## macOS cross build (from Linux via zig)

Builds one **universal** binary — an `x86_64` slice for Intel Macs (macOS 10.13+)
and an `arm64` slice for Apple silicon (macOS 11+) — and wraps it in an app
bundle with the embedded plugin (also universal) in `Contents/PlugIns/`.

Requirements, none of which need root:

- [zig](https://ziglang.org/download/) on `PATH` (tested with 0.15.2) and
  `cargo install cargo-zigbuild` — zig is the Mach-O linker.
- `rustup target add x86_64-apple-darwin aarch64-apple-darwin`.
- `llvm-lipo` (ships with LLVM; override with `LIPO=...`).
- A macOS SDK for the framework headers and link stubs, unpacked at
  `~/.local/opt/MacOSX26.1.sdk` or passed as `MACOS_SDK=/path/to/MacOSX.sdk`.
  Apple's SDK licence allows its use only on Apple hardware; take it from your
  own Xcode where that matters.

```sh
make build-macos    # build both slices, lipo them, assemble the .app
make fourier-macos  # build only the universal VST3 plugin .dylib
```

Output: `target/universal-apple-darwin/release/ReSynth Studio.app` (and the bare
universal binary beside it). The arm64 slice is ad-hoc signed by the linker,
which is what Apple silicon requires to run it at all; that app is not notarised,
so a downloaded copy needs `xattr -cr "ReSynth Studio.app"` (or right-click → Open)
the first time. `make sign-macos` signs it with a Developer ID, notarises and
staples it on a Mac reached over ssh, and brings back a zip that opens with a
double-click. Plugin editors — LeSynth Fourier's and those of VST3, VST2 and LV2
plugins — open in a window of their own; CLAP and Audio Unit editors are not
supported on macOS yet, and the editor button says so instead of opening one.

## Release (`make release`)

`make release` writes `releases/<version>/` with one zip per platform —
`ReSynth-Studio-<version>-linux-x86_64.zip`, `-windows-x86_64.zip` and
`-macos-universal.zip` — and `SHA256SUMS`. Each zip is a single folder holding
the app, its embedded plugin, `LICENSE`, a `README.txt` for that platform and
`THIRD-PARTY-LICENSES.txt`, generated from `cargo metadata` for that target.

It needs the lesynth-fourier source beside this repo, `zip`, cargo-zigbuild,
and the Mac that `make sign-macos` uses. The Linux binaries are linked by zig
against glibc 2.31 (Ubuntu 20.04 / Debian 11 and later) rather than this
machine's glibc, and the macOS zip is the notarised app, re-zipped on the Mac.

## License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE).

Copyright 2026 Jakub Hlavnicka.
