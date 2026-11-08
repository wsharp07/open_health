"""Inputs for Oura's NSSA sleep stager (sleepstaging_2_6_0), built the way the
official Android app builds them. Pure Python (no torch) and mirrored line by line
in apps/ios/OuraApp/SleepStaging.swift; tests live in tools/test_nssa_inputs.py.

Official pipeline (decompiled 7.18.0, see docs/algorithms/sleep-staging.md):

* bedtime = the ring's own `bedtime_period` (0x76), the latest event for that start;
* ACM = `sleep_acm_period` (0x72), ONE 30 s window per event with six float32
  statistics. The model wants them as
  [madIqr, madMax, madTrimmedMean, fingerAngleIqr, fingerAngleMax, fingerAngleMean];
* IBI = `ibi_and_amplitude_event` (0x60) only (never 0x80), every beat stamped at
  the START of its interval (event time minus the IBIs that follow it), clamped to
  be non-decreasing, then corrected by ecore's IBI correction (validity flag);
* temperature = `sleep_temp_event` (0x75), one float32 sample per 30 s, the last
  sample at the event time;
* each stream keeps events whose ring time falls inside the bedtime and then the
  samples from the first to the last one inside [start, end] (inclusive).
"""

import numpy as _np  # float32 arithmetic for bit-exact decoding

TAG_BEDTIME = 0x76
TAG_ACM = 0x72
TAG_IBI = 0x60
TAG_TEMP = 0x75

TEMP_SAMPLE_MS = 30000


def decode_acm_period(body):
    """`EventParser::parse_api_sleep_acm_period` (libringeventparser 0x2bf39c): bytes
    0..5 are three `int + frac/255` finger angles (tmean, max, iqr), bytes 6..11 three
    `12-bit/4095 + high nibble` MAD statistics (tmean, max, iqr), all float32. Returns
    them in the NSSA model's column order or None for a malformed body."""
    if body is None or len(body) != 12:
        return None
    f = _np.float32
    angle = [f(body[2 * i]) / f(255.0) + f(body[2 * i + 1]) for i in range(3)]
    mad = []
    for i in range(3):
        lo, hi = body[6 + 2 * i], body[7 + 2 * i]
        mad.append(f(lo | ((hi & 0x0F) << 8)) / f(4095.0) + f(hi >> 4))
    fa_mean, fa_max, fa_iqr = angle
    mad_mean, mad_max, mad_iqr = mad
    return [float(v) for v in (mad_iqr, mad_max, mad_mean, fa_iqr, fa_max, fa_mean)]


def expand_temp_event(body, event_ms):
    """`parse_api_sleep_temp_event` (0x2be9e0): an even-length 2..14 byte body of
    little-endian centi-degrees; sample i of n is stamped event - (n-1-i)*30 s."""
    if body is None or len(body) < 2 or len(body) > 15 or len(body) % 2:
        return []
    n = len(body) // 2
    out = []
    for i in range(n):
        raw = body[2 * i] | (body[2 * i + 1] << 8)
        out.append((event_ms - (n - 1 - i) * TEMP_SAMPLE_MS, float(_np.float32(raw) / _np.float32(100.0))))
    return out


def expand_ibi_event(ibis, amps, event_ms, last_ms):
    """`parse_api_ibi_and_amplitude_event` (0x2be468): beat k starts at event time
    minus ibi[k..5]; each stamp is clamped to the previous one (the parser's session
    keeps the last stamp across events). Returns (beats, new_last)."""
    beats = []
    stamp = event_ms - sum(ibis)
    for i, ibi in enumerate(ibis):
        clamped = stamp if stamp > last_ms else last_ms
        amp = amps[i] if i < len(amps) else 0
        beats.append((clamped, int(ibi), int(amp)))
        last_ms = clamped
        stamp += ibi
    return beats, last_ms


def inclusive_window(items, start_ms, end_ms):
    """`RingEventParserExtKt.indexOfFirstAndLastOrNull` + `subListInclusive`: keep
    everything from the first to the last item stamped inside [start, end]."""
    first = next((i for i, item in enumerate(items) if start_ms <= item[0] <= end_ms), None)
    if first is None:
        return []
    last = next(i for i in range(len(items) - 1, -1, -1) if start_ms <= items[i][0] <= end_ms)
    return items[first:last + 1]


def collect_nssa_inputs(rows, start_ds, end_ds, captured_unix, unix_ms, correct_ibis):
    """Assemble one bedtime's NSSA inputs.

    rows: (ds, tag, decoded_json dict, captured_unix, body bytes) in DB order.
    unix_ms(ds, captured) -> epoch ms via the shared RingClock.
    correct_ibis(list of (ts, ibi, amp)) -> list of (ts, ibi, amp, validity).
    Returns dict(bedtime=(start_ms, end_ms), acm=[(ts, [6 floats])],
    ibi=[(ts, ibi, amp, validity)], temp=[(ts, value)]).
    """
    start_ms, end_ms = unix_ms(start_ds, captured_unix), unix_ms(end_ds, captured_unix)
    acm_events, ibi_events, temp_events = [], [], []
    for ds, tag, value, captured, body in rows:
        if tag not in (TAG_ACM, TAG_IBI, TAG_TEMP):
            continue
        # ds restarts after a reboot: reject rows from another boot epoch whose
        # relative counter happens to overlap this night's range.
        if not start_ds - 6000 <= ds <= end_ds + 6000:
            continue
        event_ms = unix_ms(ds, captured)
        if abs(event_ms - (start_ms + (ds - start_ds) * 100)) > 300000:
            continue
        # The app queries DbRawEvents by event time inside the bedtime.
        if not start_ms <= event_ms <= end_ms:
            continue
        if tag == TAG_ACM:
            stats = decode_acm_period(body)
            if stats is not None:
                acm_events.append((event_ms, stats))
        elif tag == TAG_TEMP:
            temp_events.append((event_ms, body))
        else:
            ibis = value.get("ibi_ms") or []
            if len(ibis) == 6:
                ibi_events.append((event_ms, ibis, value.get("amplitude") or []))
    acm_events.sort(key=lambda e: e[0])
    temp_events.sort(key=lambda e: e[0])
    ibi_events.sort(key=lambda e: e[0])

    acm = inclusive_window(acm_events, start_ms, end_ms)
    temp = []
    for event_ms, body in temp_events:
        temp.extend(expand_temp_event(body, event_ms))
    temp = inclusive_window(temp, start_ms, end_ms)
    beats, last = [], 0
    for event_ms, ibis, amps in ibi_events:
        expanded, last = expand_ibi_event(ibis, amps, event_ms, last)
        beats.extend(expanded)
    beats = inclusive_window(beats, start_ms, end_ms)
    ibi = correct_ibis(beats) if beats else []
    return {"bedtime": (start_ms, end_ms), "acm": acm, "ibi": ibi, "temp": temp}
