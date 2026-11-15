//! Night breathing rate from heart beats, for nights the ring sends no breath of its own.
//!
//! The ring's own per-window estimate (`sleep_period_information_2.breath`) is the
//! first choice, but it is not always there: one Ring 4 (fw 2.12.13) sent 819 such
//! windows in a single batch and then none for a month, while the beat stream kept
//! coming every night. Oura's app does not read breathing off the ring either; it
//! derives it on the phone from inter-beat intervals (`breathing_rate_calculate_averages`),
//! whose resampling kernel is not recovered. This is the textbook version of the same
//! idea, ported from `tools/respiratory_rate.py`:
//!
//! Heart rate rises on the in-breath and falls on the out-breath (respiratory sinus
//! arrhythmia), so the beat-to-beat interval carries a ripple at the breathing
//! frequency. Per one-minute window: resample the intervals to 4 Hz, remove the trend,
//! and take the strongest spectral peak in the breathing band. A window counts only
//! when that peak clearly stands above the rest of the band, which drops movement and
//! poor contact on its own. The night's rate is the median of the windows that count.
//!
//! It is an estimate, not the ring's number: checked here on synthetic beats only.

use std::sync::OnceLock;

/// One usable beat: seconds from the night's start, and the interval that ended there.
pub type Beat = (f32, u16);

/// A night's breathing rate from beats, and how many one-minute windows backed it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub rate: f64,
    pub windows: usize,
}

/// Plausible inter-beat interval, ms (30–200 bpm): the range the ring's decoders use.
const IBI_MS: std::ops::RangeInclusive<i64> = 300..=2000;
const WINDOW_S: f64 = 60.0;
/// Resample rate, Hz: ecore's grid.
const FS: f64 = 4.0;
const SAMPLES: usize = (WINDOW_S * FS) as usize;
/// Breathing band, Hz: 9 to 25 breaths a minute. The floor sits above the slow
/// blood-pressure rhythm near 0.1 Hz, which is not breathing.
const BAND_LO: f64 = 0.15;
const BAND_HI: f64 = 0.42;
/// Spectrum step, Hz (0.3 breaths a minute); the peak is then refined between steps.
const BAND_STEP: f64 = 0.005;
/// A window needs beats across this much of it, and no hole longer than `MAX_GAP_S`.
const MIN_COVERAGE: f64 = 0.8;
const MAX_GAP_S: f64 = 4.0;
/// Beats this far from the window's median interval are ectopic or mis-detected.
const MAX_DEVIATION: f64 = 0.30;
/// The peak must be this many times the band's median power. Pure beat noise clears 4
/// in most windows and 10 in about one in a hundred.
const MIN_PEAK_RATIO: f64 = 10.0;
/// And the peak with its neighbours within this many Hz must hold this share of the
/// band's power.
const PEAK_HALF_WIDTH: f64 = 0.025;
const MIN_CONCENTRATION: f64 = 0.6;
/// The middle half of a night's windows must agree within this many breaths a minute:
/// sleeping breath is steady, and windows scattered across the band are not breathing.
const MAX_SPREAD: f64 = 3.0;
/// And at least this share of the windows that had the beats must show a rhythm. Noise
/// alone gets a few percent of windows past the peak tests, bunched low in the band.
const MIN_SHARE: f64 = 0.10;
/// Fewer windows than this (20 minutes) is not a night's worth.
pub const MIN_WINDOWS: usize = 20;

/// The usable beats of one packet, placed in time. The ring stamps the packet, not each
/// beat: the stamp is the last beat and earlier ones walk back by the intervals (the
/// rule `research_export` uses). Beats outside the plausible range, or that the ring's
/// quality flag rejects (`quality` is 1 for a good beat), still advance the walk but
/// are not returned.
pub fn packet_beats(packet_s: f64, ibi_ms: &[i64], quality: Option<&[i64]>) -> Vec<Beat> {
    let mut out = Vec::with_capacity(ibi_ms.len());
    let mut back = 0.0;
    for i in (0..ibi_ms.len()).rev() {
        let ibi = ibi_ms[i];
        let good = quality.map_or(true, |q| q.get(i) == Some(&1));
        let at = packet_s - back;
        if good && IBI_MS.contains(&ibi) && at >= 0.0 {
            out.push((at as f32, ibi as u16));
        }
        back += ibi.max(0) as f64 / 1000.0;
    }
    out.reverse();
    out
}

/// The night's breathing rate, breaths per minute, from its beats; `None` when fewer
/// than [`MIN_WINDOWS`] one-minute windows show a clear breathing rhythm, or when those
/// windows disagree too much to be one sleeper's breathing.
pub fn nightly_rate(beats: &[Beat]) -> Option<Estimate> {
    let mut beats: Vec<(f64, f64)> = beats
        .iter()
        .map(|&(t, ibi)| (t as f64, ibi as f64))
        .collect();
    beats.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (first, last) = (beats.first()?.0, beats.last()?.0);

    let mut rates = Vec::new();
    let mut usable = 0;
    let mut from = 0;
    let mut start = first;
    while start + WINDOW_S <= last {
        while from < beats.len() && beats[from].0 < start {
            from += 1;
        }
        let to = from + beats[from..].partition_point(|b| b.0 < start + WINDOW_S);
        match window_rate(&beats[from..to], start) {
            Window::Rate(rate) => {
                usable += 1;
                rates.push(rate);
            }
            Window::NoRhythm => usable += 1,
            Window::TooFewBeats => {}
        }
        start += WINDOW_S;
    }
    if rates.len() < MIN_WINDOWS || (rates.len() as f64) < MIN_SHARE * usable as f64 {
        return None;
    }
    rates.sort_by(f64::total_cmp);
    if rates[rates.len() * 3 / 4] - rates[rates.len() / 4] > MAX_SPREAD {
        return None;
    }
    let mid = rates.len() / 2;
    let rate = if rates.len() % 2 == 0 {
        (rates[mid - 1] + rates[mid]) / 2.0
    } else {
        rates[mid]
    };
    Some(Estimate {
        rate,
        windows: rates.len(),
    })
}

enum Window {
    /// Not enough clean beats across the minute to look for a rhythm.
    TooFewBeats,
    /// The beats were there, but no clear breathing peak.
    NoRhythm,
    /// Breaths per minute.
    Rate(f64),
}

/// One window starting at `start`.
fn window_rate(beats: &[(f64, f64)], start: f64) -> Window {
    if beats.len() < 30 {
        return Window::TooFewBeats;
    }
    let mut sorted: Vec<f64> = beats.iter().map(|b| b.1).collect();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let clean: Vec<(f64, f64)> = beats
        .iter()
        .copied()
        .filter(|b| (b.1 - median).abs() <= MAX_DEVIATION * median)
        .collect();
    if clean.len() < 30 {
        return Window::TooFewBeats;
    }
    // Enough of the window covered, and no hole a breath could hide in.
    let covered: f64 = clean.iter().map(|b| b.1 / 1000.0).sum();
    let end = start + WINDOW_S;
    let mut prev = start;
    for b in &clean {
        if b.0 - prev > MAX_GAP_S {
            return Window::TooFewBeats;
        }
        prev = b.0;
    }
    if covered < MIN_COVERAGE * WINDOW_S || end - prev > MAX_GAP_S {
        return Window::TooFewBeats;
    }

    // Intervals on an even 4 Hz grid (linear between beats, held flat at the edges).
    let mut x = [0.0f64; SAMPLES];
    let mut j = 0;
    for (i, slot) in x.iter_mut().enumerate() {
        let t = start + i as f64 / FS;
        while j + 1 < clean.len() && clean[j + 1].0 <= t {
            j += 1;
        }
        *slot = if t <= clean[0].0 {
            clean[0].1
        } else if j + 1 >= clean.len() {
            clean[clean.len() - 1].1
        } else {
            let (t0, v0) = clean[j];
            let (t1, v1) = clean[j + 1];
            if t1 > t0 {
                v0 + (v1 - v0) * (t - t0) / (t1 - t0)
            } else {
                v0
            }
        };
    }
    // Remove the straight-line trend, then taper (Hann) so the window's edges don't leak.
    let n = SAMPLES as f64;
    let mean_i = (n - 1.0) / 2.0;
    let mean_x = x.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, v) in x.iter().enumerate() {
        let di = i as f64 - mean_i;
        sxy += di * (v - mean_x);
        sxx += di * di;
    }
    let slope = sxy / sxx;
    let table = table();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean_x - slope * (i as f64 - mean_i)) * table.hann[i];
    }

    // Beats sample the rhythm once each, so nothing faster than half the heart rate
    // can be seen: cap the band there.
    let beat_hz = 1000.0 / median;
    let hi = BAND_HI.min(beat_hz / 2.0);
    let bins = table.freqs.iter().take_while(|&&f| f <= hi).count();
    if bins < 5 {
        return Window::NoRhythm;
    }
    let power: Vec<f64> = (0..bins)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            let (cos, sin) = (&table.cos[k], &table.sin[k]);
            for i in 0..SAMPLES {
                re += x[i] * cos[i];
                im += x[i] * sin[i];
            }
            re * re + im * im
        })
        .collect();
    let Some(peak) = (0..bins).max_by(|&a, &b| power[a].total_cmp(&power[b])) else {
        return Window::NoRhythm;
    };
    // A maximum on the band's edge is the shoulder of something outside it.
    if peak == 0 || peak == bins - 1 {
        return Window::NoRhythm;
    }
    let mut ranked = power.clone();
    ranked.sort_by(f64::total_cmp);
    let floor = ranked[bins / 2];
    if power[peak] <= 0.0 || power[peak] < MIN_PEAK_RATIO * floor {
        return Window::NoRhythm;
    }
    // A breathing rhythm puts most of the band's power in one place; noise spreads it.
    let near = (PEAK_HALF_WIDTH / BAND_STEP).round() as usize;
    let lobe: f64 = power[peak.saturating_sub(near)..(peak + near + 1).min(bins)]
        .iter()
        .sum();
    if lobe < MIN_CONCENTRATION * power.iter().sum::<f64>() {
        return Window::NoRhythm;
    }
    // Refine between steps with a parabola through the peak and its neighbours.
    let (a, b, c) = (power[peak - 1], power[peak], power[peak + 1]);
    let bend = a - 2.0 * b + c;
    let shift = if bend < 0.0 {
        (0.5 * (a - c) / bend).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Window::Rate((table.freqs[peak] + shift * BAND_STEP) * 60.0)
}

struct Table {
    freqs: Vec<f64>,
    hann: Vec<f64>,
    cos: Vec<Vec<f64>>,
    sin: Vec<Vec<f64>>,
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| {
        let steps = ((BAND_HI - BAND_LO) / BAND_STEP).round() as usize;
        let freqs: Vec<f64> = (0..=steps)
            .map(|k| BAND_LO + k as f64 * BAND_STEP)
            .collect();
        let tau = std::f64::consts::TAU;
        let wave = |f: f64, trig: fn(f64) -> f64| -> Vec<f64> {
            (0..SAMPLES)
                .map(|i| trig(tau * f * i as f64 / FS))
                .collect()
        };
        Table {
            hann: (0..SAMPLES)
                .map(|i| 0.5 - 0.5 * (tau * i as f64 / (SAMPLES as f64 - 1.0)).cos())
                .collect(),
            cos: freqs.iter().map(|&f| wave(f, f64::cos)).collect(),
            sin: freqs.iter().map(|&f| wave(f, f64::sin)).collect(),
            freqs,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise in -1..1 (no rand dependency).
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }

    /// Beats for `minutes` at `bpm`, breathing `breaths` a minute with a ripple of
    /// `depth_ms`, a slow 0.1 Hz wave of `slow_ms`, and `jitter_ms` of noise per beat.
    fn beats(
        minutes: f64,
        bpm: f64,
        breaths: f64,
        depth_ms: f64,
        slow_ms: f64,
        jitter_ms: f64,
    ) -> Vec<Beat> {
        let tau = std::f64::consts::TAU;
        let mut seed = 7;
        let mut out = Vec::new();
        let mut t = 0.0;
        while t < minutes * 60.0 {
            let ibi = 60_000.0 / bpm
                + depth_ms * (tau * breaths / 60.0 * t).sin()
                + slow_ms * (tau * 0.1 * t).sin()
                + jitter_ms * noise(&mut seed);
            t += ibi / 1000.0;
            out.push((t as f32, ibi.round() as u16));
        }
        out
    }

    #[test]
    fn finds_the_breathing_rate_in_clean_beats() {
        for breaths in [11.0, 13.5, 15.0, 18.0, 21.0] {
            let got = nightly_rate(&beats(60.0, 58.0, breaths, 40.0, 0.0, 5.0)).unwrap();
            assert!((got.rate - breaths).abs() < 0.3, "{breaths}: {got:?}");
            assert!(got.windows >= 55, "{got:?}");
        }
    }

    #[test]
    fn a_faint_rhythm_under_noise_is_still_read() {
        // A ripple well under half the beat noise: few windows pass, but they agree.
        let got = nightly_rate(&beats(480.0, 60.0, 16.0, 8.0, 0.0, 20.0)).unwrap();
        assert!((got.rate - 16.0).abs() < 0.5, "{got:?}");
    }

    #[test]
    fn holds_with_noise_and_a_stronger_slow_wave() {
        // A shallow ripple, beat noise of the same size, and a 0.1 Hz wave twice as deep.
        let got = nightly_rate(&beats(120.0, 52.0, 14.0, 20.0, 40.0, 20.0)).unwrap();
        assert!((got.rate - 14.0).abs() < 0.5, "{got:?}");
    }

    #[test]
    fn slow_hearts_cap_the_band_but_still_read_slow_breathing() {
        let got = nightly_rate(&beats(60.0, 42.0, 12.0, 50.0, 0.0, 5.0)).unwrap();
        assert!((got.rate - 12.0).abs() < 0.3, "{got:?}");
    }

    #[test]
    fn no_rhythm_gives_no_rate() {
        assert_eq!(
            nightly_rate(&beats(480.0, 60.0, 15.0, 0.0, 0.0, 30.0)),
            None
        );
        // Only the slow wave: it sits below the band and must not be read as breathing.
        assert_eq!(
            nightly_rate(&beats(480.0, 60.0, 15.0, 0.0, 40.0, 15.0)),
            None
        );
        for bpm in [45.0, 60.0, 75.0] {
            assert_eq!(
                nightly_rate(&beats(480.0, bpm, 15.0, 0.0, 0.0, 15.0)),
                None,
                "{bpm}"
            );
        }
        assert_eq!(nightly_rate(&[]), None);
    }

    #[test]
    fn too_little_data_gives_no_rate() {
        assert_eq!(nightly_rate(&beats(15.0, 58.0, 15.0, 40.0, 0.0, 5.0)), None);
    }

    #[test]
    fn holes_drop_their_windows_only() {
        // Every third minute has no beats at all.
        let all = beats(90.0, 58.0, 16.0, 40.0, 0.0, 5.0);
        let holed: Vec<Beat> = all
            .into_iter()
            .filter(|b| (b.0 as f64 / 60.0) as i64 % 3 != 1)
            .collect();
        let got = nightly_rate(&holed).unwrap();
        assert!((got.rate - 16.0).abs() < 0.3, "{got:?}");
        assert!(got.windows <= 60, "{got:?}");
    }

    #[test]
    fn ectopic_beats_are_ignored() {
        let mut b = beats(60.0, 58.0, 15.0, 40.0, 0.0, 5.0);
        for i in (0..b.len()).step_by(40) {
            b[i].1 = 1900;
        }
        let got = nightly_rate(&b).unwrap();
        assert!((got.rate - 15.0).abs() < 0.3, "{got:?}");
    }

    #[test]
    fn packet_beats_walk_back_from_the_stamp_and_skip_bad_ones() {
        let got = packet_beats(100.0, &[1000, 2500, 900, 800], None);
        assert_eq!(got, vec![(95.8, 1000), (99.2, 900), (100.0, 800)]);
        // Quality 1 is a good beat; anything else is dropped but still takes its time.
        let got = packet_beats(100.0, &[1000, 900, 800], Some(&[1, 0, 1]));
        assert_eq!(got, vec![(98.3, 1000), (100.0, 800)]);
        // A beat before the night's start is dropped.
        assert_eq!(packet_beats(0.5, &[1000, 800], None), vec![(0.5, 800)]);
    }
}
