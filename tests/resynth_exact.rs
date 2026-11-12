//! The requirement, stated directly: analysis followed by its inverse must
//! return the source. Not "correlates well", not "matches per period" — return
//! it, for every file, whatever its pitch or format.

use std::path::{Path, PathBuf};

use resynth_studio::analysis;
use resynth_studio::audio::decode_audio_file;
use resynth_studio::vst::{class_ids, Vst3Instance};

const NUM_HARMONICS: usize = 256;

fn load_plugin() -> Vst3Instance {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("internal_plugins")
        .join("liblesynth_fourier.so");
    Vst3Instance::load(&path, Some(&class_ids::FOURIER_SYNTH), None).expect("load plugin")
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

#[test]
fn resynthesis_reproduces_every_source_file() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let plugin = load_plugin();
    let mut checked = 0;

    for (label, path) in [
        ("D5.wav", root.join("D5.wav")),
        ("D5 -> AAC 48kbps", PathBuf::from(std::env::var("D5_AAC").unwrap_or_default())),
        ("my_voice.m4a", root.join("my_voice.m4a")),
    ] {
        if !path.exists() {
            eprintln!("skipping {label}");
            continue;
        }
        let audio = decode_audio_file(&path).expect("decode");
        let sr = audio.sample_rate;
        let sub = analysis::segment(&audio.samples, sr)
            .into_iter()
            .filter(|s| s.is_reasonable(sr))
            .max_by_key(|s| s.end - s.start)
            .expect("usable subtrack");
        let end = sub.end.min(audio.samples.len());
        let original = &audio.samples[sub.start..end];
        let contour = analysis::build_contour(&sub);
        let grid = plugin
            .analyze_full(original, sr, sub.base_freq, &contour, 0, NUM_HARMONICS)
            .expect("analyze_full");
        let rec = plugin.resynthesize_exact(&grid, true, grid.sample_rate).expect("resynthesize_exact");

        let n = original.len().min(rec.len());
        let peak = original.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-9);
        let worst = (0..n).fold(0.0f32, |m, i| m.max((original[i] - rec[i]).abs())) / peak;
        let rms = ((0..n)
            .map(|i| {
                let d = (original[i] - rec[i]) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64)
            .sqrt() as f32
            / peak;

        println!(
            "{label:<20} f0 {:>6.1} Hz  {:>5} buckets  {} of {} samples  \
             peak err {:>7.1} dB  rms err {:>7.1} dB",
            sub.base_freq,
            grid.num_buckets,
            rec.len(),
            original.len(),
            db(worst),
            db(rms),
        );
        assert_eq!(
            rec.len(),
            original.len(),
            "{label}: reconstruction must cover the whole subtrack"
        );
        assert!(
            db(worst) < -80.0,
            "{label}: reconstruction is not exact — peak error {:.1} dB",
            db(worst)
        );
        checked += 1;
    }
    assert!(checked >= 2, "not enough source files present to prove anything");
}

/// A saved instrument must reload with its exact inverse intact — what `.lsft` v3
/// exists for. The grid alone cannot be inverted (bucket lengths are not
/// derivable from `pitch_ratio`, and DC/Nyquist are not harmonics), so before v3
/// an instrument sounded worse after saving it than before.
#[test]
fn a_saved_instrument_reloads_with_its_exact_inverse() {
    use resynth_studio::track_format::TrackState;

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let wav = root.join("D5.wav");
    if !wav.exists() {
        eprintln!("skipping: D5.wav not present");
        return;
    }
    let plugin = load_plugin();
    let audio = decode_audio_file(&wav).expect("decode");
    let sr = audio.sample_rate;
    let sub = analysis::segment(&audio.samples, sr)
        .into_iter()
        .filter(|s| s.is_reasonable(sr))
        .max_by_key(|s| s.end - s.start)
        .expect("usable subtrack");
    let end = sub.end.min(audio.samples.len());
    let original = &audio.samples[sub.start..end];
    let contour = analysis::build_contour(&sub);
    let grid = plugin
        .analyze_full(original, sr, sub.base_freq, &contour, 0, NUM_HARMONICS)
        .expect("analyze_full");

    // Round-trip the grid through the file format exactly as a save/load does.
    let state = TrackState {
        num_harmonics: grid.num_harmonics,
        num_buckets: grid.num_buckets,
        base_freq: grid.base_freq,
        duration_secs: grid.duration_secs,
        sample_rate: grid.sample_rate,
        display_gain: grid.display_gain,
        amplitude: grid.amplitude.clone(),
        phase: grid.phase.clone(),
        pitch_ratio: grid.pitch_ratio.clone(),
        bucket_lengths: grid.bucket_periods.iter().map(|&p| p as u32).collect(),
        dc: grid.dc.clone(),
        nyquist: grid.nyquist.clone(),
        // A grid straight from the analyser: no editor, no harmonics switched off.
        amp_enabled: Vec::new(),
        phase_enabled: Vec::new(),
    };
    state.validate().expect("validate");
    assert!(
        state.supports_exact_inverse(),
        "an analysed instrument must save with its exact inverse"
    );
    let reloaded = TrackState::from_bytes(&state.to_bytes()).expect("reparse");
    assert!(reloaded.supports_exact_inverse());

    // Invert the *reloaded* grid and compare against the source file.
    let restored = resynth_studio::vst::AnalysisGrid {
        num_harmonics: reloaded.num_harmonics,
        num_buckets: reloaded.num_buckets,
        amplitude: reloaded.amplitude.clone(),
        phase: reloaded.phase.clone(),
        pitch_ratio: reloaded.pitch_ratio.clone(),
        bucket_periods: reloaded.bucket_lengths.iter().map(|&n| n as f32).collect(),
        dc: reloaded.dc.clone(),
        nyquist: reloaded.nyquist.clone(),
        display_gain: reloaded.display_gain,
        base_freq: reloaded.base_freq,
        sample_rate: reloaded.sample_rate,
        duration_secs: reloaded.duration_secs,
    };
    let rec = plugin
        .resynthesize_exact(&restored, true, restored.sample_rate)
        .expect("resynthesize_exact");

    let n = original.len().min(rec.len());
    let peak = original.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-9);
    let worst = (0..n).fold(0.0f32, |m, i| m.max((original[i] - rec[i]).abs())) / peak;
    println!("after save+load, peak error {:.1} dB", db(worst));
    assert_eq!(rec.len(), original.len());
    assert!(
        db(worst) < -80.0,
        "a reloaded instrument no longer reproduces its source: {:.1} dB",
        db(worst)
    );
}

/// The audition renders into the playback stream, at the *device's* rate.
/// Bucket lengths are in source samples, so unscaled they played `my_voice.m4a`
/// (24 kHz) nearly an octave sharp and half as long on a 48 kHz device. Pitch
/// and duration must survive the rate change for every file.
#[test]
fn the_audition_keeps_pitch_and_duration_at_any_device_rate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let plugin = load_plugin();
    let mut checked = 0;

    for (label, path) in [
        ("D5.wav", root.join("D5.wav")),
        ("my_voice.m4a", root.join("my_voice.m4a")),
    ] {
        if !path.exists() {
            continue;
        }
        let audio = decode_audio_file(&path).expect("decode");
        let sr = audio.sample_rate;
        let sub = analysis::segment(&audio.samples, sr)
            .into_iter()
            .filter(|s| s.is_reasonable(sr))
            .max_by_key(|s| s.end - s.start)
            .expect("usable subtrack");
        let end = sub.end.min(audio.samples.len());
        let original = &audio.samples[sub.start..end];
        let contour = analysis::build_contour(&sub);
        let grid = plugin
            .analyze_full(original, sr, sub.base_freq, &contour, 0, NUM_HARMONICS)
            .expect("analyze_full");
        let src_secs = original.len() as f32 / sr;

        for device_sr in [44_100.0f32, 48_000.0] {
            let rec = plugin
                .resynthesize_exact(&grid, true, device_sr)
                .expect("resynthesize_exact");
            let secs = rec.len() as f32 / device_sr;
            // Pitch, measured independently of the analysis: autocorrelation via
            // the host's own segmenter, run on the rendered audio.
            let got = analysis::segment(&rec, device_sr)
                .into_iter()
                .filter(|s| s.is_reasonable(device_sr))
                .max_by_key(|s| s.end - s.start)
                .map(|s| s.base_freq)
                .expect("rendered audio must still be pitched");
            let cents = 1200.0 * (got / sub.base_freq).log2();
            println!(
                "{label:<14} {sr:>6.0} Hz -> {device_sr:>6.0} Hz: {secs:.3}s vs {src_secs:.3}s, \
                 f0 {got:.1} vs {:.1} Hz ({cents:+.1} cents)",
                sub.base_freq
            );
            assert!(
                (secs - src_secs).abs() < 0.01,
                "{label} at {device_sr} Hz lasts {secs:.3}s, source is {src_secs:.3}s"
            );
            assert!(
                cents.abs() < 15.0,
                "{label} at {device_sr} Hz plays at {got:.1} Hz, source is {:.1} ({cents:+.1} cents)",
                sub.base_freq
            );
            checked += 1;
        }
    }
    assert!(checked >= 2, "no source files to check");
}
