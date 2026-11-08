//! Research export: the ring's raw per-beat and per-window streams as CSV files, on
//! the corrected wall clock, for a chosen time window.
//!
//! One implementation for every client: the iOS app (via `oura-core`), the web
//! dashboard (`/api/research-export`) and the CLI (`oura research-export`) all call
//! [`build`] and [`to_zip`], so the same window gives byte-identical files.
//!
//! Nothing here is a ring pre-computation (no sleep stages, no HRV averages): each
//! row is one value the ring recorded. Rows whose boot cannot be tied to real time
//! (`ClockSource::Undated`) are left out and counted in `manifest.json`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use oura_store::Store;
use serde_json::{json, Value};

use crate::ring_time::{ClockSource, RingClock};
use crate::{civil, OffsetHours};

/// `readings.kind` of a researcher's "mark now" event (see [`add_marker`]).
pub const MARKER_KIND: &str = "research_marker";

/// Bumped whenever a column or a rule changes, and written into `manifest.json`.
pub const EXPORT_VERSION: u32 = 1;

/// The streams a researcher can pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stream {
    /// Every heart beat: `ibi_and_amplitude_event` (0x60).
    RrIntervals,
    /// Beats with the ring's quality flag: `green_ibi_quality_event` (0x80).
    RrQuality,
    /// 30 s accelerometer statistics during sleep: `sleep_acm_period` (0x72).
    SleepMovement,
    /// 30 s accelerometer summary, day and night: `motion_event` (0x47).
    Motion,
    /// Raw SpO2 ratio and perfusion index: `spo2_r_pi_event`.
    Spo2Raw,
    /// Skin temperature: `temp_event` and `sleep_temp_event`.
    SkinTemperature,
}

impl Stream {
    pub const ALL: [Stream; 6] = [
        Stream::RrIntervals,
        Stream::RrQuality,
        Stream::SleepMovement,
        Stream::Motion,
        Stream::Spo2Raw,
        Stream::SkinTemperature,
    ];

    /// Stable id used by the CLI flag, the web query string and the iOS bridge.
    pub fn id(self) -> &'static str {
        match self {
            Stream::RrIntervals => "rr_intervals",
            Stream::RrQuality => "rr_quality",
            Stream::SleepMovement => "sleep_movement_30s",
            Stream::Motion => "motion_30s",
            Stream::Spo2Raw => "spo2_raw",
            Stream::SkinTemperature => "skin_temperature",
        }
    }

    pub fn from_id(id: &str) -> Option<Stream> {
        Stream::ALL.into_iter().find(|s| s.id() == id.trim())
    }

    pub fn file_name(self) -> String {
        format!("{}.csv", self.id())
    }

    fn header(self) -> &'static str {
        match self {
            Stream::RrIntervals => "utc_time,local_time,unix_s,rr_ms,hr_bpm,amplitude,artifact,packet_unix_s,beat_in_packet,clock_source",
            Stream::RrQuality => "utc_time,local_time,unix_s,rr_ms,hr_bpm,quality,packet_unix_s,beat_in_packet,clock_source",
            Stream::SleepMovement => "utc_time,local_time,unix_s,mad_mean_g,mad_max_g,mad_iqr_g,finger_angle_mean,finger_angle_max,finger_angle_iqr,clock_source",
            Stream::Motion => "utc_time,local_time,unix_s,avg_x,avg_y,avg_z,motion_seconds,low_intensity,high_intensity,orientation,clock_source",
            Stream::Spo2Raw => "utc_time,local_time,unix_s,sample_in_packet,r_ratio,perfusion_index,clock_source",
            Stream::SkinTemperature => "utc_time,local_time,unix_s,source,sample_in_packet,temp_c,clock_source",
        }
    }
}

/// What to export: `[start_unix, end_unix)` in UTC seconds.
#[derive(Clone, Debug)]
pub struct Request {
    pub start_unix: f64,
    pub end_unix: f64,
    pub streams: Vec<Stream>,
    /// Fixed UTC offset for the `local_time` column (the UTC column is authoritative).
    pub tz_offset_hours: f64,
}

/// The finished export: files in a fixed order (`README.md`, `manifest.json`, then
/// one CSV per requested stream).
pub struct Export {
    pub files: Vec<(String, String)>,
    pub manifest: Value,
}

// RR outside this range is not a heart beat; 2000 ms is the decoder's ceiling and
// marks a missed beat rather than a 30 bpm one.
const RR_MIN_MS: i64 = 300;
const RR_MAX_MS: i64 = 2000;
/// A beat further than this from the median of its neighbours is flagged `jump`.
const JUMP_FRACTION: f64 = 0.25;
const JUMP_NEIGHBOURS: usize = 10;

struct Row {
    unix: f64,
    source: ClockSource,
    /// Columns after `unix_s` and before `clock_source`.
    cols: String,
}

#[derive(Default)]
struct Beat {
    unix: f64,
    packet_unix: f64,
    rr: i64,
    amplitude: i64,
    index: usize,
    source: Option<ClockSource>,
}

pub fn build(db: &Path, req: &Request) -> Result<Export> {
    anyhow::ensure!(req.end_unix > req.start_unix, "the end of the window must be after its start");
    anyhow::ensure!(
        req.tz_offset_hours.is_finite() && (-24.0..=24.0).contains(&req.tz_offset_hours),
        "invalid UTC offset"
    );
    let mut streams = req.streams.clone();
    streams.sort();
    streams.dedup();
    anyhow::ensure!(!streams.is_empty(), "choose at least one data stream");

    let store = Store::open_read_only(db).context("opening DB")?;
    let events = store.decoded_events().context("reading events")?;
    let device = store.device_info().ok().flatten();
    let clock = (!events.is_empty()).then(|| RingClock::from_events(&events));

    let mut rows: BTreeMap<Stream, Vec<Row>> = streams.iter().map(|s| (*s, Vec::new())).collect();
    let mut beats: Vec<Beat> = Vec::new();
    let mut undated: BTreeMap<&'static str, usize> = BTreeMap::new();
    // Undated rows are only worth reporting when they would have fallen in the window
    // on the counter's best guess.
    let in_window = |t: f64| t >= req.start_unix && t < req.end_unix;

    if let Some(clock) = &clock {
        for (ds, tag, jstr, captured) in &events {
            let name = oura_protocol::events::event_name(*tag);
            let Some(stream) = stream_of(name) else { continue };
            if !rows.contains_key(&stream) {
                continue;
            }
            let at = clock.resolve(*ds, *captured);
            if !in_window(at.unix) && !in_window(at.unix - 60.0) && !in_window(at.unix + 60.0) {
                continue;
            }
            if !at.source.is_dated() && at.source != ClockSource::Fallback {
                if in_window(at.unix) {
                    *undated.entry(stream.id()).or_default() += 1;
                }
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(jstr) else { continue };
            match stream {
                Stream::RrIntervals => {
                    let ibi = ints(&v["ibi_ms"]);
                    let amp = ints(&v["amplitude"]);
                    for (i, (rr, unix)) in beat_times(&ibi, at.unix).into_iter().enumerate() {
                        beats.push(Beat {
                            unix,
                            packet_unix: at.unix,
                            rr,
                            amplitude: amp.get(i).copied().unwrap_or(0),
                            index: i,
                            source: Some(at.source),
                        });
                    }
                }
                Stream::RrQuality => {
                    let ibi = ints(&v["ibi_ms"]);
                    let q = ints(&v["quality"]);
                    for (i, (rr, unix)) in beat_times(&ibi, at.unix).into_iter().enumerate() {
                        push(&mut rows, stream, unix, at.source, format!(
                            "{rr},{},{},{},{i}",
                            hr(rr),
                            q.get(i).map(|x| x.to_string()).unwrap_or_default(),
                            fmt_unix(at.unix),
                        ));
                    }
                }
                Stream::SleepMovement => {
                    // Field order checked against Oura's parser (`SleepAcmPeriod`
                    // proto fields 2-7) and the data (max >= mean on every row).
                    let m = floats(&v["acm_mad"]);
                    if m.len() == 6 {
                        push(&mut rows, stream, at.unix, at.source, format!(
                            "{},{},{},{},{},{}",
                            m[3], m[4], m[5], m[0], m[1], m[2]
                        ));
                    }
                }
                Stream::Motion => {
                    let f = |k: &str| v[k].as_i64().map(|x| x.to_string()).unwrap_or_default();
                    push(&mut rows, stream, at.unix, at.source, format!(
                        "{},{},{},{},{},{},{}",
                        f("avg_x"), f("avg_y"), f("avg_z"), f("motion_seconds"),
                        f("low_intensity"), f("high_intensity"), f("orientation")
                    ));
                }
                Stream::Spo2Raw => {
                    let r = floats(&v["r"]);
                    let pi = floats(&v["perfusion_index"]);
                    for (i, (r, pi)) in r.iter().zip(&pi).enumerate() {
                        push(&mut rows, stream, at.unix, at.source, format!("{i},{r},{pi}"));
                    }
                }
                Stream::SkinTemperature => {
                    for (i, t) in floats(&v["temps_c"]).iter().enumerate() {
                        push(&mut rows, stream, at.unix, at.source, format!("{name},{i},{t}"));
                    }
                }
            }
        }
    }

    if rows.contains_key(&Stream::RrIntervals) {
        beats.sort_by(|a, b| a.unix.total_cmp(&b.unix));
        let flags = artifact_flags(&beats);
        let out = rows.get_mut(&Stream::RrIntervals).unwrap();
        for (b, flag) in beats.iter().zip(flags) {
            out.push(Row {
                unix: b.unix,
                source: b.source.unwrap_or(ClockSource::Undated),
                cols: format!(
                    "{},{},{},{flag},{},{}",
                    b.rr,
                    hr(b.rr),
                    b.amplitude,
                    fmt_unix(b.packet_unix),
                    b.index
                ),
            });
        }
    }

    let markers = read_markers(db, req)?;

    let tz = req.tz_offset_hours;
    let mut files = Vec::new();
    let mut counts = serde_json::Map::new();
    let mut clock_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut csvs = Vec::new();
    for stream in &streams {
        let mut list = rows.remove(stream).unwrap_or_default();
        list.retain(|r| in_window(r.unix));
        // Stable sort: rows of one packet keep their order.
        list.sort_by(|a, b| a.unix.total_cmp(&b.unix));
        let mut csv = String::with_capacity(list.len() * 80);
        csv.push_str(stream.header());
        csv.push('\n');
        for r in &list {
            *clock_counts.entry(r.source.label()).or_default() += 1;
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                iso(r.unix, 0.0),
                iso(r.unix, tz),
                fmt_unix(r.unix),
                r.cols,
                r.source.label()
            ));
        }
        counts.insert(stream.id().into(), json!(list.len()));
        csvs.push((stream.file_name(), csv));
    }

    if !markers.is_empty() {
        let mut csv = String::from("utc_time,local_time,unix_s,label,clock_source\n");
        for (unix, label) in &markers {
            csv.push_str(&format!("{},{},{},{},phone\n", iso(*unix, 0.0), iso(*unix, tz), fmt_unix(*unix), csv_text(label)));
        }
        counts.insert("markers".into(), json!(markers.len()));
        csvs.push(("markers.csv".to_string(), csv));
    }

    let manifest = json!({
        "export_version": EXPORT_VERSION,
        "window": {
            "start_utc": iso(req.start_unix, 0.0),
            "end_utc": iso(req.end_unix, 0.0),
            "start_unix": req.start_unix,
            "end_unix": req.end_unix,
            "tz_offset_hours": tz,
        },
        "ring": device.map(|d| json!({ "serial": d.0, "hardware": d.1, "firmware": d.2 })),
        "rows": counts,
        "rows_by_clock_source": clock_counts,
        "excluded_undated_rows": undated,
    });
    files.push(("README.md".to_string(), readme(&manifest)));
    files.push(("manifest.json".to_string(), serde_json::to_string_pretty(&manifest)? + "\n"));
    files.extend(csvs);
    Ok(Export { files, manifest })
}

/// Record a "mark now" event at `unix` (phone clock) with a free-text label, so a
/// researcher can line the ring data up with another recording (start of an audio
/// recording, a noise played on purpose). Exported as `markers.csv`.
pub fn add_marker(db: &Path, unix: f64, label: &str) -> Result<()> {
    let store = Store::open(db).context("opening DB")?;
    let serial = store.device_serials().ok().and_then(|s| s.into_iter().next()).unwrap_or_default();
    let label: String = label.chars().filter(|c| !c.is_control()).take(200).collect();
    store.insert_reading(&serial, MARKER_KIND, unix, label.trim())?;
    Ok(())
}

fn read_markers(db: &Path, req: &Request) -> Result<Vec<(f64, String)>> {
    let conn = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut stmt = conn.prepare(
        "SELECT value, COALESCE(unit, '') FROM readings WHERE kind = ?1 AND value >= ?2 AND value < ?3 ORDER BY value, id",
    )?;
    let rows = stmt
        .query_map(rusqlite::params![MARKER_KIND, req.start_unix, req.end_unix], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// A CSV field: quoted when it holds a comma, a quote or a line break.
fn csv_text(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// A ZIP of the export (deflated), the form the apps hand to the share sheet or
/// the browser. Entries carry a fixed timestamp so equal exports are equal bytes.
pub fn to_zip(export: &Export, folder: &str) -> Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .last_modified_time(zip::DateTime::default());
        for (name, body) in &export.files {
            zip.start_file(format!("{folder}/{name}"), opts)?;
            zip.write_all(body.as_bytes())?;
        }
        zip.finish()?;
    }
    Ok(buf.into_inner())
}

/// Folder/file stem for a window, e.g. `oura-research-20261015-2000`.
pub fn bundle_name(start_unix: f64, tz_offset_hours: f64) -> String {
    let local = iso(start_unix, tz_offset_hours);
    let digits: String = local.chars().filter(|c| c.is_ascii_digit()).take(12).collect();
    format!("oura-research-{}-{}", &digits[..8], &digits[8..12])
}

fn stream_of(name: &str) -> Option<Stream> {
    Some(match name {
        "ibi_and_amplitude_event" => Stream::RrIntervals,
        "green_ibi_quality_event" => Stream::RrQuality,
        "sleep_acm_period" => Stream::SleepMovement,
        "motion_event" => Stream::Motion,
        "spo2_r_pi_event" => Stream::Spo2Raw,
        "temp_event" | "sleep_temp_event" => Stream::SkinTemperature,
        _ => return None,
    })
}

/// Beat times inside one packet. The ring stamps the packet, not each beat; we take
/// the stamp as the last beat and walk back by the intervals. Missed-beat markers
/// (outside 300–2000 ms) still advance the walk by their value, as the ring counted
/// that time.
fn beat_times(ibi: &[i64], packet_unix: f64) -> Vec<(i64, f64)> {
    let mut out = vec![(0, 0.0); ibi.len()];
    let mut back = 0.0;
    for i in (0..ibi.len()).rev() {
        out[i] = (ibi[i], packet_unix - back);
        back += ibi[i] as f64 / 1000.0;
    }
    out
}

/// `range`: not a heart beat (outside 300–2000 ms). `jump`: more than 25 % away from
/// the median of the 10 valid beats on each side. Empty: plausible beat.
fn artifact_flags(beats: &[Beat]) -> Vec<&'static str> {
    let valid: Vec<bool> = beats.iter().map(|b| b.rr >= RR_MIN_MS && b.rr < RR_MAX_MS).collect();
    let idx: Vec<usize> = (0..beats.len()).filter(|i| valid[*i]).collect();
    let mut flags = vec!["range"; beats.len()];
    for (k, &i) in idx.iter().enumerate() {
        let lo = k.saturating_sub(JUMP_NEIGHBOURS);
        let hi = (k + JUMP_NEIGHBOURS + 1).min(idx.len());
        let mut w: Vec<i64> = idx[lo..hi].iter().filter(|&&j| j != i).map(|&j| beats[j].rr).collect();
        if w.is_empty() {
            flags[i] = "";
            continue;
        }
        w.sort_unstable();
        let med = w[w.len() / 2] as f64;
        flags[i] = if (beats[i].rr as f64 - med).abs() / med > JUMP_FRACTION { "jump" } else { "" };
    }
    flags
}

fn push(rows: &mut BTreeMap<Stream, Vec<Row>>, s: Stream, unix: f64, source: ClockSource, cols: String) {
    rows.get_mut(&s).unwrap().push(Row { unix, source, cols });
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default()
}
fn floats(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default()
}
fn hr(rr: i64) -> String {
    if rr > 0 {
        format!("{:.1}", 60_000.0 / rr as f64)
    } else {
        String::new()
    }
}
fn fmt_unix(t: f64) -> String {
    format!("{t:.3}")
}

/// ISO 8601 with milliseconds and the offset, e.g. `2026-10-16T03:05:48.120+01:00`.
pub fn iso(unix: f64, tz_offset_hours: f64) -> String {
    let off = (OffsetHours(tz_offset_hours).0 * 3600.0).round() as i64;
    let ms_total = (unix * 1000.0).round() as i64 + off * 1000;
    let secs = ms_total.div_euclid(1000);
    let ms = ms_total.rem_euclid(1000);
    let (y, mo, d) = civil(secs.div_euclid(86_400));
    let sod = secs.rem_euclid(86_400);
    let zone = if off == 0 {
        "Z".to_string()
    } else {
        let sign = if off < 0 { '-' } else { '+' };
        format!("{sign}{:02}:{:02}", off.abs() / 3600, off.abs() % 3600 / 60)
    };
    format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}.{ms:03}{zone}",
        sod / 3600,
        sod % 3600 / 60,
        sod % 60
    )
}

fn readme(manifest: &Value) -> String {
    let w = &manifest["window"];
    format!(
        "# Oura ring research export\n\n\
Window: {start} to {end} (UTC). Export format version {ver}.\n\n\
Every row is a value the ring recorded; nothing is averaged or scored.\n\n\
## Time columns (all files)\n\n\
- `utc_time`: ISO 8601 UTC with milliseconds. Use this one to align with other recordings.\n\
- `local_time`: same instant at the fixed offset {tz:+} h chosen at export.\n\
- `unix_s`: seconds since 1970-01-01 UTC.\n\
- `clock_source`: how the ring's internal counter was converted to real time.\n\
  `anchor` = between two time synchronisations with the phone (accurate to ~1 s);\n\
  `projected` = from the nearest plausible synchronisation; `download_time` =\n\
  estimated from the download time, can be off by the gap between two syncs.\n\
  Rows the app cannot date at all are excluded and counted in `manifest.json`.\n\n\
## rr_intervals.csv: every heart beat (event 0x60)\n\n\
The ring sends beats in packets of 6 with one timestamp. `packet_unix_s` is that\n\
timestamp. `unix_s` is our estimate of each beat: the packet time is taken as the\n\
last beat and earlier beats are placed by walking back the intervals. Consecutive\n\
packets do not always join up exactly, so treat per-beat times as +/- 1 s.\n\n\
- `rr_ms`: interval since the previous beat, ms. `hr_bpm` = 60000 / rr_ms.\n\
- `amplitude`: optical pulse amplitude (arbitrary units, low = weak signal).\n\
- `artifact`: empty = plausible beat; `range` = outside 300-2000 ms (2000 means a\n\
  missed beat); `jump` = more than 25 % from the median of the 10 valid beats on\n\
  each side. Nothing is removed: filter on this column.\n\n\
## rr_quality.csv: beats with a quality flag (event 0x80)\n\n\
Same timing rule, packets of 7, recorded intermittently. `quality` is the ring's flag\n\
(1 = good).\n\n\
## sleep_movement_30s.csv: accelerometer statistics (event 0x72)\n\n\
One row per 30 s window, during sleep only. `mad_*_g`: mean, maximum and\n\
interquartile range of the accelerometer's mean amplitude deviation, in g (higher\n\
= more movement). `finger_angle_*`: mean, maximum and IQR of the finger's angle\n\
change in the window (ring units). Field order follows Oura's own parser.\n\n\
## motion_30s.csv: accelerometer summary (event 0x47)\n\n\
One row per 30 s, day and night. `avg_x/y/z`: mean acceleration per axis (about\n\
1/1000 g, so the magnitude is ~1000 at rest); gives the hand's orientation.\n\
`motion_seconds`: seconds of the 30 with movement. `low_intensity`,\n\
`high_intensity`: ring movement counters. `orientation`: ring orientation code.\n\n\
## spo2_raw.csv (event spo2_r_pi)\n\n\
`r_ratio`: red/infrared ratio that Oura's SpO2 curve uses (not a percentage).\n\
`perfusion_index`: optical perfusion. Several samples share one packet timestamp.\n\n\
## skin_temperature.csv\n\n\
`source` = `temp_event` (about every minute, several sensors per row as\n\
`sample_in_packet`) or `sleep_temp_event` (during sleep). Degrees Celsius.\n\n\
## markers.csv (only when markers were placed)\n\n\
Moments marked with \"Mark now\" in the app, on the phone's clock (`clock_source` =\n\
`phone`), with the label typed at the time. Use them to line the ring data up with\n\
another recording.\n\n\
## Not available\n\n\
The raw accelerometer at 10-50 Hz and the raw PPG waveform are not stored by the\n\
ring overnight on consumer firmware; 30 s windows are the finest movement data.\n",
        start = w["start_utc"].as_str().unwrap_or(""),
        end = w["end_utc"].as_str().unwrap_or(""),
        ver = manifest["export_version"],
        tz = w["tz_offset_hours"].as_f64().unwrap_or(0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_formats_utc_and_offsets() {
        assert_eq!(iso(1_760_576_748.12, 0.0), "2025-10-16T01:05:48.120Z");
        assert_eq!(iso(1_760_576_748.12, 1.0), "2025-10-16T02:05:48.120+01:00");
        assert_eq!(iso(0.0, -5.5), "1969-12-31T18:30:00.000-05:30");
    }

    #[test]
    fn beats_walk_back_from_the_packet_stamp() {
        let t = beat_times(&[1000, 1200, 800], 100.0);
        assert_eq!(t, vec![(1000, 98.0), (1200, 99.2), (800, 100.0)]);
    }

    #[test]
    fn artifacts_flag_range_and_jumps() {
        let rr = [1000, 1010, 990, 400, 1005, 2000, 995, 1000];
        let beats: Vec<Beat> = rr
            .iter()
            .enumerate()
            .map(|(i, &rr)| Beat { unix: i as f64, rr, ..Default::default() })
            .collect();
        assert_eq!(artifact_flags(&beats), vec!["", "", "", "jump", "", "range", "", ""]);
    }

    #[test]
    fn markers_round_trip_into_the_export() {
        let dir = std::env::temp_dir().join(format!("rx-marker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("t.db");
        drop(Store::open(&db).unwrap());
        add_marker(&db, 1000.25, "noise, 60 dB").unwrap();
        add_marker(&db, 5000.0, "outside").unwrap();
        let req = Request { start_unix: 0.0, end_unix: 2000.0, streams: vec![Stream::Motion], tz_offset_hours: 0.0 };
        let export = build(&db, &req).unwrap();
        let (_, csv) = export.files.iter().find(|(n, _)| n == "markers.csv").unwrap();
        assert_eq!(csv, "utc_time,local_time,unix_s,label,clock_source\n1970-01-01T00:16:40.250Z,1970-01-01T00:16:40.250Z,1000.250,\"noise, 60 dB\",phone\n");
        assert_eq!(export.manifest["rows"]["markers"], 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn stream_ids_round_trip() {
        for s in Stream::ALL {
            assert_eq!(Stream::from_id(s.id()), Some(s));
        }
        assert_eq!(bundle_name(1_760_576_748.0, 1.0), "oura-research-20251016-0205");
    }
}
