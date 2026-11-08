//! Readiness — an Oura-like 0–100 morning score, one per main sleep.
//!
//! The contributors are the ones Oura publishes, and the weights are the combiner
//! weights open_oura recovered by regressing a year of Oura's own Readiness scores on
//! their contributor sub-scores (open_oura `docs/algorithms/score-weights.md`,
//! R² 0.97). The sub-score curves are NOT Oura's: Oura doesn't publish them and the
//! ecore tables don't read back, so each curve below is this crate's own, written to
//! follow Oura's description of the contributor:
//!
//! * a deviation from your own baseline gets full marks within half a personal
//!   standard deviation (the "smallest worthwhile change" of HRV-guided training,
//!   Plews/Buchheit) and loses [`POINTS_PER_SD`] points per further SD;
//! * a ratio to your own activity baseline gets full marks inside a band and loses
//!   [`POINTS_PER_DOUBLING`] points per doubling (or halving) beyond it.
//!
//! Every input is a nightly or daily value the ring measured; nights and days from a
//! different ring than the one scored never enter its baselines (fit and sensor differ
//! between rings). A score whose ring has fewer than [`MATURE_NIGHTS`] earlier nights
//! is `provisional`, as Oura's is for its first two weeks.

use serde_json::{json, Value};

/// Oura's combiner weights as recovered by regression (they sum to 99).
const W_RESTING_HR: f64 = 17.0;
const W_PREVIOUS_NIGHT: f64 = 15.0;
const W_HRV_BALANCE: f64 = 15.0;
const W_TEMPERATURE: f64 = 13.0;
const W_SLEEP_BALANCE: f64 = 12.0;
const W_PREVIOUS_DAY_ACTIVITY: f64 = 10.0;
const W_RECOVERY_INDEX: f64 = 10.0;
const W_ACTIVITY_BALANCE: f64 = 7.0;

/// Earlier nights of the same ring before a score stops being provisional.
pub const MATURE_NIGHTS: usize = 14;
/// Fewest earlier values a baseline contributor needs before it is scored at all.
const MIN_BASELINE_VALUES: usize = 3;
/// A sleep this long or longer is scored; shorter ones are naps.
pub const MIN_MAIN_SLEEP_S: f64 = 3.0 * 3600.0;
/// No score when the contributors present carry less than this share of the weight.
const MIN_WEIGHT_SHARE: f64 = 0.5;

const POINTS_PER_SD: f64 = 15.0;
const POINTS_PER_DOUBLING: f64 = 40.0;
/// Points lost per hour short of the time-in-bed target.
const POINTS_PER_HOUR_SHORT: f64 = 20.0;
/// Time in bed that earns full marks for Previous Night and Sleep Balance. Oura's
/// sleep-need default is 8 h of sleep; with no sleep stages from the ring this is
/// judged on time in bed, so the same 8 h is a slightly easier bar.
const IN_BED_TARGET_H: f64 = 8.0;
/// Hours between the night's lowest heart rate and waking that earn full Recovery
/// Index marks: Oura's stated optimum is the low point at least 6 h before you wake.
const RECOVERY_FULL_H: f64 = 6.0;
/// A day needs this many minutes of activity readings to count (12 h worn).
pub const FULL_DAY_MINUTES: u32 = 12 * 60;

/// One measured night, as `build_summary` saw it.
#[derive(Clone, Debug, Default)]
pub struct NightIn {
    pub serial: Option<String>,
    /// Local calendar day you woke (days since 1970-01-01).
    pub wake_day: i64,
    pub start_unix: f64,
    pub end_unix: f64,
    /// Lowest 5-minute heart rate, bpm.
    pub rhr: Option<f64>,
    /// When that lowest heart rate was measured.
    pub lowest_hr_unix: Option<f64>,
    /// Mean nightly RMSSD, ms.
    pub hrv_ms: Option<f64>,
    /// Nightly skin temperature, °C.
    pub skin_temp: Option<f64>,
    /// The night's sleep score, when the ring staged the night.
    pub sleep_score: Option<f64>,
}

impl NightIn {
    fn in_bed_h(&self) -> f64 {
        (self.end_unix - self.start_unix) / 3600.0
    }
    fn is_main(&self) -> bool {
        self.end_unix - self.start_unix >= MIN_MAIN_SLEEP_S
    }
}

/// One local day of activity from one ring: active MET-minutes (Σ max(MET − 1, 0)
/// over its minutes) and how many minutes had a reading.
#[derive(Clone, Debug)]
pub struct DayIn {
    pub serial: String,
    pub day: i64,
    pub active_met_min: f64,
    pub minutes: u32,
}

/// Full marks within half an SD; `POINTS_PER_SD` off for each SD beyond.
fn deviation_score(z: f64) -> f64 {
    (100.0 - POINTS_PER_SD * (z - 0.5).max(0.0)).clamp(0.0, 100.0)
}

/// Full marks for a ratio within `band`× either way; `POINTS_PER_DOUBLING` off per
/// doubling or halving beyond it.
fn ratio_score(ratio: f64, band: f64) -> f64 {
    if ratio <= 0.0 {
        return 0.0;
    }
    let beyond = (ratio.ln().abs() - band.ln()).max(0.0) / std::f64::consts::LN_2;
    (100.0 - POINTS_PER_DOUBLING * beyond).clamp(0.0, 100.0)
}

fn shortfall_score(hours: f64) -> f64 {
    (100.0 - POINTS_PER_HOUR_SHORT * (IN_BED_TARGET_H - hours).max(0.0)).clamp(0.0, 100.0)
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// Sample standard deviation, floored at `floor` so a very steady baseline doesn't
/// turn a trivial wobble into many SDs.
fn sd(v: &[f64], floor: f64) -> f64 {
    let Some(m) = mean(v) else { return floor };
    if v.len() < 2 {
        return floor;
    }
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt().max(floor)
}

/// Mean weighted by `decay^(days before `today`)`, so recent days count more.
fn recent_weighted(points: &[(i64, f64)], today: i64, decay: f64) -> Option<f64> {
    let (mut sum, mut weight) = (0.0, 0.0);
    for &(day, v) in points {
        let w = decay.powi((today - day).max(0) as i32);
        sum += w * v;
        weight += w;
    }
    (weight > 0.0).then(|| sum / weight)
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

struct Contributor {
    key: &'static str,
    label: &'static str,
    weight: f64,
    score: f64,
    value: Option<f64>,
    baseline: Option<f64>,
    unit: &'static str,
    basis: &'static str,
}

impl Contributor {
    fn to_json(&self) -> Value {
        json!({
            "key": self.key,
            "label": self.label,
            "weight": self.weight,
            "score": self.score.round(),
            "value": self.value.map(r1),
            "baseline": self.baseline.map(r1),
            "unit": self.unit,
            "basis": self.basis,
        })
    }
}

/// Readiness for `nights[index]`, from it and the earlier nights and days of the same
/// ring. `nights` is oldest first. `None` for a nap or when too little is known.
pub fn readiness_for(nights: &[NightIn], index: usize, days: &[DayIn]) -> Option<Value> {
    let night = &nights[index];
    if !night.is_main() {
        return None;
    }
    let same_ring = |serial: &Option<String>| serial == &night.serial;
    let earlier: Vec<&NightIn> = nights[..index]
        .iter()
        .filter(|n| n.is_main() && same_ring(&n.serial) && n.end_unix <= night.start_unix)
        .collect();
    let within = |days_back: i64| {
        earlier
            .iter()
            .copied()
            .filter(move |n| night.wake_day - n.wake_day <= days_back)
    };
    let mut parts: Vec<Contributor> = Vec::new();

    // Resting heart rate: last night's lowest against your ~2-month average. Higher is worse.
    if let Some(rhr) = night.rhr {
        let prior: Vec<f64> = within(60).filter_map(|n| n.rhr).collect();
        if prior.len() >= MIN_BASELINE_VALUES {
            let base = mean(&prior).unwrap_or(rhr);
            let z = (rhr - base) / sd(&prior, 1.0);
            parts.push(Contributor {
                key: "resting_hr",
                label: "Resting heart rate",
                weight: W_RESTING_HR,
                score: deviation_score(z),
                value: Some(rhr),
                baseline: Some(base),
                unit: "bpm",
                basis: "lowest heart rate vs your 60-day average",
            });
        }
    }

    // HRV balance: the last 14 days (recent nights weighted more) against ~3 months.
    if night.hrv_ms.is_some() {
        let mut series: Vec<(i64, f64)> = within(90)
            .filter_map(|n| n.hrv_ms.map(|v| (n.wake_day, v)))
            .collect();
        series.extend(night.hrv_ms.map(|v| (night.wake_day, v)));
        if series.len() > MIN_BASELINE_VALUES {
            let all: Vec<f64> = series.iter().map(|p| p.1).collect();
            let recent_pts: Vec<(i64, f64)> = series
                .iter()
                .copied()
                .filter(|p| night.wake_day - p.0 < 14)
                .collect();
            let recent = recent_weighted(&recent_pts, night.wake_day, 0.8);
            let base = mean(&all);
            if let (Some(recent), Some(base)) = (recent, base) {
                let z = (base - recent) / sd(&all, 2.0);
                parts.push(Contributor {
                    key: "hrv_balance",
                    label: "HRV balance",
                    weight: W_HRV_BALANCE,
                    score: deviation_score(z),
                    value: Some(recent),
                    baseline: Some(base),
                    unit: "ms",
                    basis: "14-day weighted HRV vs your 90-day average",
                });
            }
        }
    }

    // Body temperature: last night against your ~2-month average. Either way is off.
    if let Some(t) = night.skin_temp {
        let prior: Vec<f64> = within(60).filter_map(|n| n.skin_temp).collect();
        if prior.len() >= MIN_BASELINE_VALUES {
            let base = mean(&prior).unwrap_or(t);
            let z = (t - base).abs() / sd(&prior, 0.15);
            parts.push(Contributor {
                key: "temperature",
                label: "Body temperature",
                weight: W_TEMPERATURE,
                score: deviation_score(z),
                value: Some(t - base),
                baseline: Some(base),
                unit: "°C",
                basis: "difference from your 60-day average",
            });
        }
    }

    // Recovery index: how long before waking your heart rate bottomed out.
    if let Some(low_at) = night.lowest_hr_unix {
        let hours = ((night.end_unix - low_at) / 3600.0).max(0.0);
        parts.push(Contributor {
            key: "recovery_index",
            label: "Recovery index",
            weight: W_RECOVERY_INDEX,
            score: (100.0 * hours / RECOVERY_FULL_H).clamp(0.0, 100.0),
            value: Some(hours),
            baseline: None,
            unit: "h",
            basis: "hours from the night's lowest heart rate to waking",
        });
    }

    // Previous night: the sleep score when the ring staged the night, else time in bed.
    parts.push(match night.sleep_score {
        Some(score) => Contributor {
            key: "previous_night",
            label: "Previous night",
            weight: W_PREVIOUS_NIGHT,
            score: score.clamp(0.0, 100.0),
            value: Some(score),
            baseline: None,
            unit: "",
            basis: "the night's sleep score",
        },
        None => Contributor {
            key: "previous_night",
            label: "Previous night",
            weight: W_PREVIOUS_NIGHT,
            score: shortfall_score(night.in_bed_h()),
            value: Some(night.in_bed_h()),
            baseline: Some(IN_BED_TARGET_H),
            unit: "h",
            basis: "time in bed vs 8 h (the ring reports no sleep stages)",
        },
    });

    // Sleep balance: average time in bed over the last 14 days against the same target.
    let fortnight: Vec<f64> = within(13)
        .map(NightIn::in_bed_h)
        .chain(std::iter::once(night.in_bed_h()))
        .collect();
    if fortnight.len() >= MIN_BASELINE_VALUES {
        let avg = mean(&fortnight).unwrap_or(0.0);
        parts.push(Contributor {
            key: "sleep_balance",
            label: "Sleep balance",
            weight: W_SLEEP_BALANCE,
            score: shortfall_score(avg),
            value: Some(avg),
            baseline: Some(IN_BED_TARGET_H),
            unit: "h",
            basis: "average time in bed over 14 days vs 8 h",
        });
    }

    // Activity: full days from this ring only.
    let full: Vec<(i64, f64)> = days
        .iter()
        .filter(|d| Some(&d.serial) == night.serial.as_ref() && d.minutes >= FULL_DAY_MINUTES)
        .map(|d| (d.day, d.active_met_min))
        .collect();
    let yesterday = night.wake_day - 1;

    // Previous day activity: yesterday against your 30-day average. Far more or far
    // less than usual both count against you.
    if let Some(&(_, y)) = full.iter().find(|d| d.0 == yesterday) {
        let prior: Vec<f64> = full
            .iter()
            .filter(|d| d.0 < yesterday && yesterday - d.0 <= 30)
            .map(|d| d.1)
            .collect();
        if prior.len() >= MIN_BASELINE_VALUES {
            if let Some(base) = mean(&prior).filter(|b| *b > 0.0) {
                parts.push(Contributor {
                    key: "previous_day_activity",
                    label: "Previous day activity",
                    weight: W_PREVIOUS_DAY_ACTIVITY,
                    score: ratio_score(y / base, 1.5),
                    value: Some(y),
                    baseline: Some(base),
                    unit: "MET-min",
                    basis: "yesterday's activity vs your 30-day average",
                });
            }
        }
    }

    // Activity balance: the last 14 days (recent weighted more) against ~2 months.
    let upto: Vec<(i64, f64)> = full
        .iter()
        .copied()
        .filter(|d| d.0 <= yesterday && yesterday - d.0 < 60)
        .collect();
    let recent: Vec<(i64, f64)> = upto
        .iter()
        .copied()
        .filter(|d| yesterday - d.0 < 14)
        .collect();
    if upto.len() >= 7 && !recent.is_empty() {
        let all: Vec<f64> = upto.iter().map(|d| d.1).collect();
        if let (Some(r), Some(base)) = (
            recent_weighted(&recent, yesterday, 0.8),
            mean(&all).filter(|b| *b > 0.0),
        ) {
            parts.push(Contributor {
                key: "activity_balance",
                label: "Activity balance",
                weight: W_ACTIVITY_BALANCE,
                score: ratio_score(r / base, 1.25),
                value: Some(r),
                baseline: Some(base),
                unit: "MET-min",
                basis: "14-day weighted activity vs your 60-day average",
            });
        }
    }

    let total = W_RESTING_HR
        + W_PREVIOUS_NIGHT
        + W_HRV_BALANCE
        + W_TEMPERATURE
        + W_SLEEP_BALANCE
        + W_PREVIOUS_DAY_ACTIVITY
        + W_RECOVERY_INDEX
        + W_ACTIVITY_BALANCE;
    let present: f64 = parts.iter().map(|c| c.weight).sum();
    let score = (present / total >= MIN_WEIGHT_SHARE)
        .then(|| (parts.iter().map(|c| c.weight * c.score).sum::<f64>() / present).round());
    Some(json!({
        "score": score,
        "provisional": earlier.len() < MATURE_NIGHTS,
        "history_nights": earlier.len(),
        "weight_present": present,
        "contributors": parts.iter().map(Contributor::to_json).collect::<Vec<_>>(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: f64 = 86_400.0;

    /// A main night ending on `wake_day` at 07:00 UTC after 8 h in bed.
    fn night(wake_day: i64, rhr: f64, hrv: f64, temp: f64) -> NightIn {
        let end = wake_day as f64 * DAY + 7.0 * 3600.0;
        NightIn {
            serial: Some("B".into()),
            wake_day,
            start_unix: end - 8.0 * 3600.0,
            end_unix: end,
            rhr: Some(rhr),
            lowest_hr_unix: Some(end - 6.5 * 3600.0),
            hrv_ms: Some(hrv),
            skin_temp: Some(temp),
            sleep_score: None,
        }
    }

    #[test]
    fn curves() {
        assert_eq!(deviation_score(0.0), 100.0);
        assert_eq!(deviation_score(0.5), 100.0);
        assert_eq!(deviation_score(1.5), 85.0);
        assert_eq!(deviation_score(20.0), 0.0);
        assert_eq!(ratio_score(1.4, 1.5), 100.0);
        assert!((ratio_score(3.0, 1.5) - 60.0).abs() < 1e-9);
        assert!((ratio_score(1.0 / 3.0, 1.5) - 60.0).abs() < 1e-9);
        assert_eq!(shortfall_score(8.5), 100.0);
        assert_eq!(shortfall_score(6.5), 70.0);
    }

    #[test]
    fn a_steady_fortnight_scores_full_and_matures() {
        let nights: Vec<NightIn> = (0..15)
            .map(|d| night(100 + d, 55.0 + (d % 3) as f64, 50.0 + (d % 4) as f64, 35.0))
            .collect();
        let r = readiness_for(&nights, 14, &[]).unwrap();
        assert_eq!(r["provisional"], false);
        assert_eq!(r["history_nights"], 14);
        // no activity data: the score is renormalised over what was measured
        assert_eq!(r["weight_present"], 82.0);
        assert!(r["score"].as_f64().unwrap() >= 95.0, "{r}");
    }

    #[test]
    fn a_rough_night_pulls_the_score_down() {
        let mut nights: Vec<NightIn> = (0..15)
            .map(|d| night(100 + d, 55.0 + (d % 3) as f64, 50.0 + (d % 4) as f64, 35.0))
            .collect();
        let last = nights.last_mut().unwrap();
        last.rhr = Some(64.0);
        last.hrv_ms = Some(30.0);
        last.skin_temp = Some(35.8);
        last.lowest_hr_unix = Some(last.end_unix - 1.0 * 3600.0);
        let r = readiness_for(&nights, 14, &[]).unwrap();
        let by_key = |k: &str| {
            r["contributors"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["key"] == k)
                .unwrap()["score"]
                .as_f64()
                .unwrap()
        };
        assert!(by_key("resting_hr") < 50.0);
        assert!(by_key("temperature") < 50.0);
        assert!(by_key("recovery_index") < 20.0);
        assert!(r["score"].as_f64().unwrap() < 75.0, "{r}");
    }

    #[test]
    fn another_rings_nights_never_enter_the_baseline() {
        let mut nights: Vec<NightIn> = (0..4).map(|d| night(100 + d, 40.0, 90.0, 33.0)).collect();
        for n in &mut nights[..3] {
            n.serial = Some("A".into());
        }
        let r = readiness_for(&nights, 3, &[]).unwrap();
        assert_eq!(r["history_nights"], 0);
        assert_eq!(r["provisional"], true);
        let keys: Vec<&str> = r["contributors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["key"].as_str().unwrap())
            .collect();
        assert!(!keys.contains(&"resting_hr"), "{keys:?}");
    }

    #[test]
    fn naps_get_no_score_and_too_little_data_gives_no_number() {
        let mut nap = night(100, 50.0, 50.0, 35.0);
        nap.start_unix = nap.end_unix - 3600.0;
        assert!(readiness_for(&[nap], 0, &[]).is_none());
        // a first night alone: previous night + recovery index are 25 of 99 weight
        let r = readiness_for(&[night(100, 50.0, 50.0, 35.0)], 0, &[]).unwrap();
        assert!(r["score"].is_null(), "{r}");
    }

    #[test]
    fn activity_compares_full_days_of_the_same_ring() {
        let nights: Vec<NightIn> = (0..10).map(|d| night(100 + d, 55.0, 50.0, 35.0)).collect();
        let day = |day: i64, met: f64, minutes: u32, serial: &str| DayIn {
            serial: serial.into(),
            day,
            active_met_min: met,
            minutes,
        };
        let mut days: Vec<DayIn> = (100..108).map(|d| day(d, 200.0, 1_000, "B")).collect();
        days.push(day(108, 800.0, 1_000, "B")); // yesterday: 4× usual
        days.push(day(99, 5_000.0, 1_000, "A")); // another ring's day is ignored
        days.push(day(98, 5_000.0, 100, "B")); // a part-worn day is ignored
        let r = readiness_for(&nights, 9, &days).unwrap();
        let c = r["contributors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["key"] == "previous_day_activity")
            .unwrap()
            .clone();
        assert_eq!(c["baseline"], 200.0);
        // 4× is log2(4 / 1.5) ≈ 1.4 doublings past the band edge, 40 points each
        assert!((c["score"].as_f64().unwrap() - 43.0).abs() <= 1.0, "{c}");
        assert!(r["contributors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["key"] == "activity_balance"));
    }
}
