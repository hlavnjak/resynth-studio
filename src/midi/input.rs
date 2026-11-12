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
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use anyhow::{Context, Result};
use midir::{MidiInput, MidiInputConnection};

pub type MidiEventQueue = Arc<Mutex<VecDeque<[u8; 3]>>>;

/// One message as a listener sees it: when it arrived, and what it was. The
/// stamp is taken in the MIDI callback, which is the only place the keyboard's
/// timing survives — a GUI thread that polls at 60 Hz can only say "some time in
/// the last 16 ms", which is a third of a 1/16 note at 120 BPM.
pub type TimedMessage = (Instant, [u8; 3]);

/// A listener on the keyboard: every message copied here, stamped.
///
/// A tap **copies**; it does not consume. The queue an open editor plays from is
/// drained destructively by its audio callback, so a recorder that read from it
/// would take notes out of the instrument's mouth — the user would hear half of
/// what they played and record the other half.
pub type MidiTap = Arc<Mutex<Vec<TimedMessage>>>;

/// The taps currently listening, held weakly: a recorder that goes away
/// unregisters itself simply by being dropped, and nothing has to remember to
/// take it off the list.
pub type MidiTaps = Arc<Mutex<Vec<Weak<Mutex<Vec<TimedMessage>>>>>>;

/// How many octaves the keyboard is transposed by on the way in, shared with the
/// input thread so the picker works on a live connection.
///
/// A small controller — a Keystation Mini 32 is 32 keys, C3 upwards — simply has
/// no low notes on it, so a bass line has to be played somewhere else and moved.
/// Doing it here rather than in each consumer means one implementation for the
/// instrument editors, the Sequencer's recording and anything added later: what is
/// heard and what is recorded cannot disagree about which note was played.
pub type OctaveShift = Arc<AtomicI32>;

/// How far the keyboard may be moved, in octaves either way. Four covers a
/// 32-key controller reaching either end of a piano.
pub const MAX_OCTAVE_SHIFT: i32 = 4;

/// No shift at all — the keyboard as it is played.
pub fn new_octave_shift() -> OctaveShift {
    Arc::new(AtomicI32::new(0))
}

/// Cap on messages held in one tap. A recording is a few hundred at most; this
/// only bounds a tap left listening and forgotten. Past it the tap stops
/// growing rather than dropping its oldest — a recording with a hole in the
/// middle is worse than one that ends early, and it can say where it stopped.
const MAX_TAPPED_EVENTS: usize = 65_536;

/// Cap on buffered MIDI messages. The queue is drained by an audio engine only
/// while an instrument/subtrack editor is open; without this bound, playing a
/// connected keyboard with no editor open would grow it without limit (and flood
/// on the next open). Dropping the oldest keeps at most a brief burst.
const MAX_QUEUED_EVENTS: usize = 1024;

/// Create a new empty MIDI event queue.
pub fn new_midi_queue() -> MidiEventQueue {
    Arc::new(Mutex::new(VecDeque::new()))
}

/// A fresh, empty tap list — nothing is listening yet.
pub fn new_midi_taps() -> MidiTaps {
    Arc::new(Mutex::new(Vec::new()))
}

/// Start listening. The returned tap fills with everything the keyboard sends
/// (alongside, not instead of, the main queue) until it is dropped.
pub fn add_midi_tap(taps: &MidiTaps) -> MidiTap {
    let tap: MidiTap = Arc::new(Mutex::new(Vec::new()));
    if let Ok(mut list) = taps.lock() {
        // Sweep out the taps whose owners have gone while we are here — this is
        // the only place the list is added to, so it is the natural place to
        // keep it from growing one dead entry per recording.
        list.retain(|t| t.strong_count() > 0);
        list.push(Arc::downgrade(&tap));
    }
    tap
}

/// List available MIDI input port names.
pub fn list_midi_ports() -> Result<Vec<String>> {
    let midi_in = MidiInput::new("resynth-studio-query")?;
    let ports = midi_in.ports();
    let names: Vec<String> = ports
        .iter()
        .map(|p| midi_in.port_name(p).unwrap_or_else(|_| "Unknown".into()))
        .collect();
    Ok(names)
}

/// List MIDI input ports that look like USB MIDI keyboards.
/// Filters out virtual/software ports (e.g. "Midi Through") and keeps
/// ports whose names suggest a USB hardware device.
pub fn list_usb_midi_keyboards() -> Result<Vec<String>> {
    let all = list_midi_ports()?;
    let filtered = all
        .into_iter()
        .filter(|name| {
            let lower = name.to_lowercase();
            // Exclude virtual/through ports
            if lower.contains("through") || lower.contains("virtual") || lower.contains("rtpmidi") {
                return false;
            }
            // Keep everything else — on ALSA, remaining ports are typically
            // hardware devices (USB MIDI keyboards, controllers, etc.)
            true
        })
        .collect();
    Ok(filtered)
}

/// Spawn a MIDI input thread that pushes raw 3-byte messages into the queue.
/// Connects to the first port whose name contains `device_filter`, or port 0 if None.
pub fn spawn_midi_thread(
    midi_events: MidiEventQueue,
    taps: MidiTaps,
    octave_shift: OctaveShift,
    device_filter: Option<&str>,
) -> Result<MidiInputConnection<()>> {
    let mut midi_in = MidiInput::new("resynth-studio-midi-in")?;
    midi_in.ignore(midir::Ignore::None);

    let ports = midi_in.ports();
    if ports.is_empty() {
        anyhow::bail!("No MIDI input ports found");
    }

    log::info!("Available MIDI input ports:");
    for (i, port) in ports.iter().enumerate() {
        let name = midi_in
            .port_name(port)
            .unwrap_or_else(|_| "Unknown".to_string());
        log::info!("  [{}] {}", i, name);
    }

    let selected_port = if let Some(filter) = device_filter {
        ports
            .iter()
            .enumerate()
            .find(|(_, p)| {
                midi_in
                    .port_name(p)
                    .unwrap_or_default()
                    .contains(filter)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        0
    };

    let port = &ports[selected_port];
    log::info!(
        "Connecting to MIDI device: {}",
        midi_in.port_name(port)?
    );

    // What each key that is *down* was transposed by. A key takes the shift as
    // it was when it was struck and keeps it until it is released: moving the
    // picker with a key held would otherwise send the note-off to a different
    // note and leave the first one sounding for good.
    let mut held_shift = [0i8; 128];
    let conn = midi_in
        .connect(
            port,
            "resynth-studio-midi-conn",
            move |_stamp, message, _| {
                if message.len() >= 3 {
                    let Some(msg) = shift_octaves(
                        [message[0], message[1], message[2]],
                        &octave_shift,
                        &mut held_shift,
                    ) else {
                        // Shifted off the end of MIDI — there is no such note to
                        // play, and its release will be dropped the same way.
                        return;
                    };
                    {
                        let mut queue = midi_events.lock().unwrap();
                        if queue.len() >= MAX_QUEUED_EVENTS {
                            queue.pop_front();
                        }
                        queue.push_back(msg);
                    }
                    // Listeners get their own stamped copy. `midir`'s own stamp
                    // is a device-relative microsecond count with no fixed
                    // origin; an `Instant` taken here can be compared with the
                    // one the transport started on, which is what a recording
                    // has to line up against.
                    let now = Instant::now();
                    if let Ok(list) = taps.lock() {
                        for tap in list.iter().filter_map(Weak::upgrade) {
                            if let Ok(mut events) = tap.lock() {
                                if events.len() < MAX_TAPPED_EVENTS {
                                    events.push((now, msg));
                                }
                            }
                        }
                    }
                }
            },
            (),
        )
        .map_err(|e| anyhow::anyhow!("Failed to connect to MIDI input: {}", e))?;

    log::info!("MIDI input connected.");
    Ok(conn)
}

/// Move a message by the current octave shift, or `None` if that would put it
/// off the end of MIDI.
///
/// Only the messages that name a key are moved — note on, note off, and the
/// per-key aftertouch that follows them. Everything else (the wheel, the
/// pedal, program changes) passes through untouched: they are not notes, and
/// transposing their data bytes would turn a modulation into nonsense.
fn shift_octaves(
    mut msg: [u8; 3],
    octave_shift: &OctaveShift,
    held_shift: &mut [i8; 128],
) -> Option<[u8; 3]> {
    let status = msg[0] & 0xF0;
    if !matches!(status, 0x80 | 0x90 | 0xA0) {
        return Some(msg);
    }
    let key = (msg[1] & 0x7F) as usize;
    let semitones = if status == 0x90 && msg[2] > 0 {
        // A key going down takes the shift as it stands, and is remembered by it.
        let shift = octave_shift.load(Ordering::Relaxed).clamp(
            -MAX_OCTAVE_SHIFT,
            MAX_OCTAVE_SHIFT,
        ) * 12;
        held_shift[key] = shift as i8;
        shift
    } else {
        // Its release — and its aftertouch — follow it wherever it went.
        held_shift[key] as i32
    };
    let shifted = key as i32 + semitones;
    if !(0..=127).contains(&shifted) {
        return None;
    }
    msg[1] = shifted as u8;
    Some(msg)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A router's insides, without a device: the routing is what has behaviour,
    /// and it is the same whether a real port or this filled the queue.
    fn router_inner(default_port: Option<&str>) -> RouterInner {
        RouterInner {
            default_port: default_port.map(str::to_string),
            default_target: KeyboardTarget::Everyone,
            default_held: [false; 128],
            open: HashMap::new(),
            subs: Vec::new(),
            next_id: 0,
        }
    }

    fn subscribe(inner: &mut RouterInner, want: Option<&str>) -> MidiEventQueue {
        let queue = new_midi_queue();
        let id = inner.next_id;
        inner.next_id += 1;
        inner.subs.push(Sub {
            id,
            want: want.map(str::to_string),
            queue: Arc::downgrade(&queue),
        });
        queue
    }

    fn drained(queue: &MidiEventQueue) -> Vec<[u8; 3]> {
        queue.lock().unwrap().drain(..).collect()
    }

    /// The MIDI panel's instrument picker: of the instances following the
    /// panel's port only the chosen one plays, an instance on a port of its
    /// own is left alone, and a key held across the switch is released where
    /// it was pressed.
    #[test]
    fn the_panels_keyboard_can_be_given_to_one_instance() {
        let mut inner = router_inner(Some("A"));
        let first = subscribe(&mut inner, None);
        let second = subscribe(&mut inner, None);
        let own_port = subscribe(&mut inner, Some("B"));

        inner.set_default_target(KeyboardTarget::Feed(1));
        inner.deliver("A", [0x90, 60, 100]);
        inner.deliver("B", [0x90, 40, 100]);
        assert!(drained(&first).is_empty());
        assert_eq!(drained(&second), vec![[0x90, 60, 100]]);
        assert_eq!(drained(&own_port), vec![[0x90, 40, 100]]);

        // Key 60 is still down on the second instance when the first takes over.
        inner.set_default_target(KeyboardTarget::Feed(0));
        assert_eq!(drained(&second), vec![[0x80, 60, 0]]);
        inner.deliver("A", [0x80, 60, 0]);
        inner.deliver("A", [0x90, 62, 100]);
        assert_eq!(drained(&first), vec![[0x80, 60, 0], [0x90, 62, 100]]);
        assert!(drained(&second).is_empty());

        // An instrument picked before its editor ever opened: silence, not everything.
        inner.set_default_target(KeyboardTarget::Nobody);
        inner.deliver("A", [0x90, 64, 100]);
        assert!(drained(&first).iter().all(|m| m[0] == 0x80));
        assert!(drained(&second).is_empty());

        inner.set_default_target(KeyboardTarget::Everyone);
        inner.deliver("A", [0x90, 65, 100]);
        assert_eq!(drained(&first), vec![[0x90, 65, 100]]);
        assert_eq!(drained(&second), vec![[0x90, 65, 100]]);
    }

    /// The defect the router exists to fix: two instances open at once each hear
    /// **everything** their keyboard sends. One shared queue was drained
    /// destructively, so they split the stream and neither played a tune.
    #[test]
    fn two_instances_on_one_keyboard_both_hear_all_of_it() {
        let mut inner = router_inner(Some("A"));
        let one = subscribe(&mut inner, None);
        let two = subscribe(&mut inner, Some("A"));

        inner.deliver("A", [0x90, 60, 100]);
        inner.deliver("A", [0x80, 60, 0]);
        assert_eq!(drained(&one), vec![[0x90, 60, 100], [0x80, 60, 0]]);
        assert_eq!(drained(&two), vec![[0x90, 60, 100], [0x80, 60, 0]]);
    }

    /// Two keyboards, two instruments: a message goes only to what asked for
    /// that port.
    #[test]
    fn each_instance_hears_only_its_own_keyboard() {
        let mut inner = router_inner(Some("A"));
        let follows_default = subscribe(&mut inner, None);
        let on_a = subscribe(&mut inner, Some("A"));
        let on_b = subscribe(&mut inner, Some("B"));

        inner.deliver("B", [0x90, 48, 100]);
        assert!(drained(&follows_default).is_empty());
        assert!(drained(&on_a).is_empty());
        assert_eq!(drained(&on_b), vec![[0x90, 48, 100]]);

        // Connecting the panel elsewhere moves everything that follows it, with
        // nothing re-subscribed.
        inner.default_port = Some("B".to_string());
        inner.deliver("B", [0x90, 50, 100]);
        assert_eq!(drained(&follows_default), vec![[0x90, 50, 100]]);
        assert_eq!(drained(&on_b), vec![[0x90, 50, 100]]);
        assert!(drained(&on_a).is_empty());
    }

    /// An instance that has gone unsubscribes itself, and its keyboard is then
    /// held by nothing — a device kept open by a closed editor is one no other
    /// program can have.
    #[test]
    fn a_dropped_instance_releases_its_keyboard() {
        let mut inner = router_inner(Some("A"));
        let gone = subscribe(&mut inner, Some("B"));
        assert!(port_is_wanted("B", Some("A"), &inner.subs));

        drop(gone);
        // Delivering is what notices: the queue's owner is gone, so the
        // subscription goes with it.
        inner.deliver("B", [0x90, 60, 100]);
        assert!(inner.subs.is_empty(), "a dead subscription was kept");
        assert!(!port_is_wanted("B", Some("A"), &inner.subs));
        // The panel's own port stays open whether or not an instrument asked for it.
        assert!(port_is_wanted("A", Some("A"), &inner.subs));
    }

    /// The shift moves notes by whole octaves, and leaves everything that is
    /// not a note alone.
    #[test]
    fn the_shift_moves_notes_and_nothing_else() {
        let shift = new_octave_shift();
        let mut held = [0i8; 128];
        shift.store(-2, Ordering::Relaxed);

        // Middle C down two octaves.
        assert_eq!(
            shift_octaves([0x90, 60, 100], &shift, &mut held),
            Some([0x90, 36, 100])
        );
        // A control change carries a controller number, not a note: moving it
        // would turn the modulation wheel into some other control.
        assert_eq!(
            shift_octaves([0xB0, 60, 100], &shift, &mut held),
            Some([0xB0, 60, 100])
        );
        // Off the bottom of MIDI — there is no such note to play.
        assert_eq!(shift_octaves([0x90, 12, 100], &shift, &mut held), None);
    }

    /// The classic stuck note: the picker is moved while a key is held, so the
    /// release names a different note than the press did. A key keeps the shift
    /// it was struck with until it comes back up.
    #[test]
    fn moving_the_shift_under_a_held_key_does_not_strand_the_note() {
        let shift = new_octave_shift();
        let mut held = [0i8; 128];

        shift.store(-1, Ordering::Relaxed);
        let on = shift_octaves([0x90, 60, 100], &shift, &mut held).unwrap();
        assert_eq!(on, [0x90, 48, 100]);

        // …and now the user moves the picker, still holding the key.
        shift.store(2, Ordering::Relaxed);
        // Both spellings of a release follow the note where it went.
        assert_eq!(
            shift_octaves([0x80, 60, 0], &shift, &mut held),
            Some([0x80, 48, 0])
        );
        assert_eq!(
            shift_octaves([0x90, 60, 0], &shift, &mut held),
            Some([0x90, 48, 0])
        );
        // Aftertouch on the same key goes with it too.
        assert_eq!(
            shift_octaves([0xA0, 60, 90], &shift, &mut held),
            Some([0xA0, 48, 90])
        );
        // The next press takes the shift as it stands now.
        assert_eq!(
            shift_octaves([0x90, 60, 100], &shift, &mut held),
            Some([0x90, 84, 100])
        );
    }
}

// ── Routing one keyboard per instance ───────────────────────────────────────

/// Which keyboard feeds one plugin instance.
///
/// **Why a router at all.** There used to be one connection and one queue, and
/// every open editor drained it — destructively, so two editors open at once
/// each got about half of what was played and neither played a recognisable
/// note. And a machine with two keyboards on it could only ever use one of
/// them.
///
/// A subscription is a queue of its own plus the port it wants, so:
///
/// * two instances open at once each hear **everything**, rather than splitting
///   the stream between them;
/// * each can be pointed at a different keyboard;
/// * a port is opened once however many instances listen to it, and closed
///   again when the last one goes.
///
/// `None` for a wanted port means "whatever the MIDI panel is connected to",
/// which is what an instrument uses until it is told otherwise. That is resolved at
/// delivery, not at subscription, so changing the panel's port moves every
/// instrument that follows it without re-subscribing anything.
#[derive(Clone)]
pub struct MidiRouter {
    inner: Arc<Mutex<RouterInner>>,
    /// Applied once, here, so every subscriber sees the same note — see
    /// [`shift_octaves`].
    octave_shift: OctaveShift,
    /// The Sequencer's recorders, which listen to every port at once: a take is
    /// "what was played", not "what one instance heard".
    taps: MidiTaps,
}

struct RouterInner {
    /// The port a subscriber gets when it has not asked for one of its own.
    default_port: Option<String>,
    /// Which of the subscribers following that port actually hear it.
    default_target: KeyboardTarget,
    /// Keys down on the default port right now, so that moving the target
    /// mid-note can release them where they were pressed.
    default_held: [bool; 128],
    /// Open connections, by port name. Dropping one closes the device.
    open: HashMap<String, MidiInputConnection<()>>,
    subs: Vec<Sub>,
    next_id: u64,
}

/// One instance's feed.
struct Sub {
    id: u64,
    /// The port it asked for, or `None` to follow the panel's.
    want: Option<String>,
    /// Weak, so an instance that goes away unsubscribes by being dropped. The
    /// queue is kept alive by the instrument that owns it and the audio engine
    /// draining it.
    queue: Weak<Mutex<VecDeque<[u8; 3]>>>,
}

/// Who hears the MIDI panel's port, among the instances that follow it.
///
/// With several editors open, every one of them following the panel's port
/// plays every key — which is right for layering two sounds and wrong for
/// playing one of them. This is the MIDI panel's instrument picker. An
/// instance pointed at a port of its own is not following the panel and is
/// not affected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KeyboardTarget {
    /// Every instance following the panel's port.
    #[default]
    Everyone,
    /// Only the instance with this feed id.
    Feed(u64),
    /// One instrument was picked, but it has no feed yet (its editor has never
    /// been opened): nothing plays rather than everything.
    Nobody,
}

impl KeyboardTarget {
    fn includes(self, feed_id: u64) -> bool {
        match self {
            Self::Everyone => true,
            Self::Feed(id) => id == feed_id,
            Self::Nobody => false,
        }
    }
}

/// A subscription handed to one instance: the queue its audio engine drains, and
/// the id its source is changed by.
pub struct MidiFeed {
    pub queue: MidiEventQueue,
    pub id: u64,
}

impl MidiRouter {
    pub fn new(octave_shift: OctaveShift, taps: MidiTaps) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RouterInner {
                default_port: None,
                default_target: KeyboardTarget::Everyone,
                default_held: [false; 128],
                open: HashMap::new(),
                subs: Vec::new(),
                next_id: 0,
            })),
            octave_shift,
            taps,
        }
    }

    /// The port everything that has not chosen one of its own listens to.
    pub fn default_port(&self) -> Option<String> {
        self.inner.lock().ok()?.default_port.clone()
    }

    /// Connect the panel's own port — the default feed. Opening it is what
    /// "Connect" does; every instrument following the default moves with it.
    pub fn set_default_port(&self, port: Option<String>) -> Result<()> {
        if let Some(name) = &port {
            self.open_port(name)?;
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.default_port = port;
            inner.close_unused();
        }
        Ok(())
    }

    /// Choose which of the instances following the panel's port hear it — see
    /// [`KeyboardTarget`]. Cheap when nothing changes, so it can be called
    /// every frame.
    pub fn set_default_target(&self, target: KeyboardTarget) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.set_default_target(target);
        }
    }

    /// Start feeding a new instance. `want` is the port it should listen to, or
    /// `None` to follow the panel's.
    pub fn subscribe(&self, want: Option<String>) -> Result<MidiFeed> {
        if let Some(name) = &want {
            self.open_port(name)?;
        }
        let queue = new_midi_queue();
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("the MIDI router was poisoned"))?;
        let id = inner.next_id;
        inner.next_id += 1;
        inner.subs.retain(|s| s.queue.strong_count() > 0);
        inner.subs.push(Sub {
            id,
            want,
            queue: Arc::downgrade(&queue),
        });
        Ok(MidiFeed { queue, id })
    }

    /// Point an existing feed at another keyboard. Takes effect on the next
    /// message: nothing is re-opened on the instance's side, and a note already
    /// held is released by whichever port sent it (see [`shift_octaves`] for the
    /// same idea applied to the octave picker).
    pub fn set_source(&self, feed_id: u64, want: Option<String>) -> Result<()> {
        if let Some(name) = &want {
            self.open_port(name)?;
        }
        if let Ok(mut inner) = self.inner.lock() {
            if let Some(sub) = inner.subs.iter_mut().find(|s| s.id == feed_id) {
                sub.want = want;
            }
            inner.close_unused();
        }
        Ok(())
    }

    /// Drop the connections nothing is listening to any more — an editor closed,
    /// an instrument removed. Cheap, and the only thing that hands a device back.
    pub fn release_unused(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.subs.retain(|s| s.queue.strong_count() > 0);
            inner.close_unused();
        }
    }

    /// Open `name` if it is not open already, and start delivering from it.
    fn open_port(&self, name: &str) -> Result<()> {
        {
            let inner = self
                .inner
                .lock()
                .map_err(|_| anyhow::anyhow!("the MIDI router was poisoned"))?;
            if inner.open.contains_key(name) {
                return Ok(());
            }
        }
        // Outside the lock: connecting talks to the driver, and the callback it
        // installs takes the same lock.
        let conn = self.connect(name)?;
        if let Ok(mut inner) = self.inner.lock() {
            inner.open.insert(name.to_string(), conn);
        }
        Ok(())
    }

    /// The connection itself: one callback per port, delivering to whoever wants
    /// that port when the message arrives.
    fn connect(&self, name: &str) -> Result<MidiInputConnection<()>> {
        let mut midi_in = MidiInput::new("resynth-studio-midi-in")?;
        midi_in.ignore(midir::Ignore::None);
        let port = midi_in
            .ports()
            .into_iter()
            .find(|p| midi_in.port_name(p).is_ok_and(|n| n == name))
            .with_context(|| format!("MIDI port '{name}' is not there any more"))?;

        let inner = self.inner.clone();
        let octave_shift = self.octave_shift.clone();
        let taps = self.taps.clone();
        let port_name = name.to_string();
        // See `spawn_midi_thread`: a key keeps the shift it was pressed with.
        let mut held_shift = [0i8; 128];
        let conn = midi_in
            .connect(
                &port,
                "resynth-studio-midi-conn",
                move |_stamp, message, _| {
                    if message.len() < 3 {
                        return;
                    }
                    let Some(msg) = shift_octaves(
                        [message[0], message[1], message[2]],
                        &octave_shift,
                        &mut held_shift,
                    ) else {
                        return;
                    };
                    if let Ok(mut inner) = inner.lock() {
                        inner.deliver(&port_name, msg);
                    }
                    tap(&taps, msg);
                },
                (),
            )
            .map_err(|e| anyhow::anyhow!("could not open MIDI port '{name}': {e}"))?;
        log::info!("MIDI input connected: {name}");
        Ok(conn)
    }
}

impl RouterInner {
    /// Push one message to every instance listening to `port`, dropping the
    /// subscriptions whose owner has gone.
    fn deliver(&mut self, port: &str, msg: [u8; 3]) {
        let follows_default = self.default_port.as_deref() == Some(port);
        if follows_default {
            let key = (msg[1] & 0x7F) as usize;
            match msg[0] & 0xF0 {
                0x90 => self.default_held[key] = msg[2] > 0,
                0x80 => self.default_held[key] = false,
                _ => {}
            }
        }
        let target = self.default_target;
        self.subs.retain(|sub| {
            let Some(queue) = sub.queue.upgrade() else {
                return false;
            };
            let wanted = match &sub.want {
                Some(name) => name == port,
                None => follows_default && target.includes(sub.id),
            };
            if wanted {
                if let Ok(mut q) = queue.lock() {
                    if q.len() >= MAX_QUEUED_EVENTS {
                        q.pop_front();
                    }
                    q.push_back(msg);
                }
            }
            true
        });
    }

    /// Move the panel's port to another target. An instance that stops hearing
    /// it is sent a note-off for every key still down: the release that is
    /// coming will go to the new target, and without this the old one would
    /// hold the note for good.
    fn set_default_target(&mut self, target: KeyboardTarget) {
        let old = self.default_target;
        if old == target {
            return;
        }
        self.default_target = target;
        let held = self.default_held;
        for sub in &self.subs {
            if sub.want.is_some() || !old.includes(sub.id) || target.includes(sub.id) {
                continue;
            }
            let Some(queue) = sub.queue.upgrade() else { continue };
            if let Ok(mut q) = queue.lock() {
                for key in (0..128u8).filter(|&k| held[k as usize]) {
                    q.push_back([0x80, key, 0]);
                }
            };
        }
    }

    /// Close the ports nothing listens to. A device held open by a closed editor
    /// is a device another program cannot have.
    fn close_unused(&mut self) {
        let default = self.default_port.clone();
        let subs = std::mem::take(&mut self.subs);
        self.open
            .retain(|name, _| port_is_wanted(name, default.as_deref(), &subs));
        self.subs = subs;
    }
}

/// Whether a port has anything listening to it: the panel is connected to it, or
/// an instance asked for it by name.
fn port_is_wanted(name: &str, default: Option<&str>, subs: &[Sub]) -> bool {
    Some(name) == default || subs.iter().any(|s| s.want.as_deref() == Some(name))
}

/// Copy one message to every listening recorder.
fn tap(taps: &MidiTaps, msg: [u8; 3]) {
    let now = Instant::now();
    if let Ok(list) = taps.lock() {
        for tap in list.iter().filter_map(Weak::upgrade) {
            if let Ok(mut events) = tap.lock() {
                if events.len() < MAX_TAPPED_EVENTS {
                    events.push((now, msg));
                }
            }
        }
    }
}
