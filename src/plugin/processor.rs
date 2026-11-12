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
//! Driving a plugin's audio, one block at a time, whatever its format.
//!
//! The instrument editor's audio stream and the Sequencer's mix both do the same
//! thing to a plugin every block: hand it the notes due in the block, let it
//! render into buffers laid out the way it negotiated, and read its main output
//! back. [`BlockProcessor`] is that, once. Everything it needs is allocated when
//! it is made — an audio callback allocates nothing — and the format-specific
//! part (a VST3 `ProcessData`, a CLAP event list, an LV2 atom sequence…) is a
//! [`RealtimeProcess`] the plugin's own module builds over the same buffers.

use std::sync::Arc;

use anyhow::Result;

use super::PluginInstance;

/// A MIDI message due `offset` frames into the block being processed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MidiEvent {
    pub offset: u32,
    pub data: [u8; 3],
}

/// Most notes one block will carry. A schedule denser than this in a single
/// block is not music; past it, events wait for the next block rather than
/// making the audio thread allocate.
pub const MAX_EVENTS_PER_BLOCK: usize = 512;

/// The format-specific half of a [`BlockProcessor`].
///
/// Built over a scratch whose pointer table never moves, so whatever a format
/// lays over it (bus descriptors, connected ports) stays valid for the
/// processor's life. `Send` because the processor is moved into the audio
/// callback; nothing else ever touches it.
pub(crate) trait RealtimeProcess: Send {
    /// Render `frames` frames: inputs are silence, outputs are written into the
    /// scratch's output channels, and `events` are the notes due in the block,
    /// in time order.
    fn process(&mut self, scratch: &mut AudioScratch, frames: usize, events: &[MidiEvent]);
}

/// A plugin, the buffers it renders into, and everything its format's
/// `process()` call needs — ready for an audio callback to drive.
pub struct BlockProcessor {
    /// Keeps the plugin (and its library) alive for as long as anything can
    /// call `process()` on it — the realtime half below points straight into it.
    _plugin: Arc<PluginInstance>,
    scratch: AudioScratch,
    /// Total channels across the audio input buses: where outputs start in
    /// `scratch`.
    in_channels: usize,
    /// Channels on the main output bus — what reaches the device.
    main_out: usize,
    /// The block size the plugin was set up for; it must never be handed more.
    max_block: usize,
    /// Notes waiting for the next block. Pre-allocated, never grown.
    events: Vec<MidiEvent>,
    realtime: Box<dyn RealtimeProcess>,
}

impl BlockProcessor {
    /// Lay out buffers for `plugin`'s negotiated bus layout and build its
    /// format's process state over them. The plugin must already be
    /// initialised.
    ///
    /// `fallback_block` is used only when the plugin reports no block size;
    /// a plugin with no output bus at all still gets a stereo one to write
    /// into, because the device is going to read something.
    pub fn new(plugin: Arc<PluginInstance>, fallback_block: usize) -> Result<Self> {
        let io = plugin.io();
        let out_buses: Vec<usize> = if io.outputs.is_empty() {
            vec![2]
        } else {
            io.outputs.clone()
        };
        let in_channels: usize = io.inputs.iter().sum();
        let out_channels: usize = out_buses.iter().sum();
        let max_block = if io.max_block > 0 {
            io.max_block
        } else {
            fallback_block.max(1)
        };
        let mut scratch = AudioScratch::new(in_channels + out_channels, max_block);
        let realtime = plugin.realtime(&mut scratch, &io.inputs, &out_buses, max_block)?;
        Ok(Self {
            _plugin: plugin,
            scratch,
            in_channels,
            main_out: out_buses[0],
            max_block,
            events: Vec::with_capacity(MAX_EVENTS_PER_BLOCK),
            realtime,
        })
    }

    /// The largest block [`Self::process`] will render.
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// Channels on the main output bus. Zero for a plugin with no output.
    pub fn main_out(&self) -> usize {
        self.main_out
    }

    /// Queue a MIDI message for the next block, `offset` frames into it.
    /// Dropped (and reported false) once the block is full.
    pub fn push_event(&mut self, offset: u32, data: [u8; 3]) -> bool {
        if self.events.len() >= MAX_EVENTS_PER_BLOCK {
            return false;
        }
        self.events.push(MidiEvent { offset, data });
        true
    }

    /// Render one block of up to `frames` frames — never more than the plugin
    /// was set up for — and return how many were rendered. The queued events
    /// are delivered and cleared; one past the end of the block lands on its
    /// last frame.
    pub fn process(&mut self, frames: usize) -> usize {
        let frames = frames.min(self.max_block);
        if frames == 0 {
            self.events.clear();
            return 0;
        }
        let last = frames as u32 - 1;
        for ev in &mut self.events {
            ev.offset = ev.offset.min(last);
        }
        // Stable, so two events on one frame keep the order they were sent in
        // — a note-off and the note-on that retriggers it must not swap.
        self.events.sort_by_key(|e| e.offset);
        self.scratch.reset(frames);
        self.realtime.process(&mut self.scratch, frames, &self.events);
        self.events.clear();
        frames
    }

    /// Main-bus channel `ch` of the block just rendered. A narrower plugin
    /// repeats its last channel, so a mono instrument fills a stereo device.
    pub fn output(&self, ch: usize) -> &[f32] {
        if self.main_out == 0 {
            return &[];
        }
        self.scratch
            .channel(self.in_channels + ch.min(self.main_out - 1))
    }
}

/// The buffers a plugin's `process()` writes through, and the pointer table it
/// reads them from.
///
/// Allocated once per stream and owned by whoever drives `process()` — the audio
/// callback must not allocate, and the pointer table the bus descriptors read
/// must not move. Raw pointers are not `Send` on their own; this whole set is,
/// because nothing but its owner ever touches it.
pub struct AudioScratch {
    planar: Vec<Vec<f32>>,
    ptrs: Vec<*mut f32>,
}

unsafe impl Send for AudioScratch {}

impl AudioScratch {
    /// `channels` buffers of `frames` samples each, plus the overrun pad.
    pub fn new(channels: usize, frames: usize) -> Self {
        let mut planar: Vec<Vec<f32>> = (0..channels.max(1))
            .map(|_| vec![0.0f32; frames + PROCESS_OVERRUN_PAD])
            .collect();
        let ptrs = planar.iter_mut().map(|v| v.as_mut_ptr()).collect();
        AudioScratch { planar, ptrs }
    }

    /// Silence `frames` samples (and the pad) on every channel, ready for a
    /// block: input buses have nothing to carry, and a plugin may add into its
    /// output rather than overwrite it. Also re-takes the channel pointers, which
    /// never change but must be derived afresh to stay valid to use.
    pub fn reset(&mut self, frames: usize) {
        for (slot, channel) in self.ptrs.iter_mut().zip(self.planar.iter_mut()) {
            let n = (frames + PROCESS_OVERRUN_PAD).min(channel.len());
            channel[..n].fill(0.0);
            *slot = channel.as_mut_ptr();
        }
    }

    /// The pointer table, to lay out as bus descriptors.
    pub fn ptrs_mut(&mut self) -> &mut [*mut f32] {
        &mut self.ptrs
    }

    /// One channel's samples, after a block has been processed.
    pub fn channel(&self, index: usize) -> &[f32] {
        &self.planar[index]
    }

    /// Frames each channel holds, not counting the overrun pad.
    pub fn frames(&self) -> usize {
        self.planar.first().map_or(0, |c| c.len() - PROCESS_OVERRUN_PAD)
    }
}

/// Slack allocated past the block on every channel buffer handed to a plugin.
///
/// Plenty of plugins process in a fixed internal block and round the host's
/// block *up* to it: Dexed works in 16 samples, Surge XT in 32. Ask either for
/// 4410 frames — which is exactly what this machine's device asks us for — and
/// the last internal block runs past the end of a buffer sized to the letter,
/// corrupting the heap. Hosts get away with tight buffers only because their
/// block sizes are powers of two; this pad is what makes any block size safe,
/// and it is far larger than any plausible internal block.
pub const PROCESS_OVERRUN_PAD: usize = 1024;
