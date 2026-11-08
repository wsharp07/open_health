#!/usr/bin/env python3
"""Write the synthetic NSSA parity fixture shared by the Python and Swift tests
(apps/ios/OuraApp/Tests/NssaParityFixture.json). Everything here is generated from
a fixed seed; no personal ring data is involved.

    python tools/make_nssa_fixture.py          # rewrite the fixture
    python tools/make_nssa_fixture.py --check  # exit 1 if it is stale
"""
import json
import random
import sys
from pathlib import Path

from ibi_correction import correct_ibis
from nssa_inputs import decode_acm_period, expand_ibi_event, expand_temp_event, inclusive_window

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "apps" / "ios" / "OuraApp" / "Tests" / "NssaParityFixture.json"


def aggregate_5min(stages):
    """Same rule as oura-summary aggregate_30s_to_5min / Sleep.aggregate5Min."""
    codes = [c for c in stages if c in (1, 2, 3, 4)]
    out = []
    for i in range(0, len(codes), 10):
        w = codes[i:i + 10]
        if len(w) < 5:
            continue
        counts = {c: w.count(c) for c in (1, 2, 3, 4)}
        if counts[4] >= 4:
            out.append(4)
            continue
        top = max(counts.values())
        out.append(next(c for c in (4, 1, 3, 2) if counts[c] >= top))
    return out


def ibi_stream(rng, n, t0):
    """Plausible sleep beats with ectopics, dropouts, fast tails and gaps."""
    out, t, base = [], t0, rng.randint(850, 1150)
    for i in range(n):
        ibi = int(base + rng.gauss(0, 35))
        r = rng.random()
        if r < 0.03:
            ibi = rng.randint(300, 520)          # fast tail / ectopic
        elif r < 0.05:
            ibi = rng.randint(1600, 2400)        # missed beat
        elif r < 0.055:
            ibi = 0                              # empty slot
        if rng.random() < 0.004:
            t += rng.randint(5000, 40000)        # sensor gap
        amp = rng.choice([0, rng.randint(100, 3000)])
        t += max(ibi, 1)
        out.append([t, ibi, amp])
        if i % 300 == 0:
            base = rng.randint(800, 1200)
    return out


def build():
    rng = random.Random(20261016)
    acm = []
    for _ in range(40):
        body = [rng.randrange(256) for _ in range(12)]
        acm.append({"body": body, "values": decode_acm_period(bytes(body))})
    temps = []
    for n in (1, 3, 7):
        body = []
        for _ in range(n):
            v = rng.randint(2500, 3800)
            body += [v & 0xFF, v >> 8]
        t = 1792100000000 + rng.randrange(10 ** 6)
        temps.append({"body": body, "event_ms": t, "samples": expand_temp_event(bytes(body), t)})
    ibi_events, last, beats = [], 0, []
    t = 1792100000000
    for k in range(12):
        ibis = [rng.randint(300, 1900) for _ in range(6)]
        amps = [rng.randint(0, 4000) for _ in range(6)]
        t += rng.choice([sum(ibis), sum(ibis) - 2000, sum(ibis) + 900])  # overlapping packets clamp
        out, last = expand_ibi_event(ibis, amps, t, last)
        ibi_events.append({"ibis": ibis, "amps": amps, "event_ms": t, "beats": out})
        beats += out
    windows = []
    for lo, hi in ((beats[3][0], beats[40][0]), (beats[0][0] - 1, beats[-1][0] + 1), (0, 1)):
        windows.append({"start": lo, "end": hi, "kept": len(inclusive_window(beats, lo, hi)),
                        "first": (inclusive_window(beats, lo, hi) or [[None]])[0][0]})
    corrections = []
    for s in range(6):
        stream = ibi_stream(rng, 700 + 150 * s, 1792100000000 + s * 10 ** 7)
        corrections.append({"raw": stream, "corrected": [list(b) for b in correct_ibis(stream)]})
    stage_runs = []
    for _ in range(8):
        n = rng.randint(0, 64)
        st = [rng.choice([1, 2, 2, 2, 3, 4, 0]) for _ in range(n)]
        stage_runs.append({"stages": st, "five_min": aggregate_5min(st)})
    return {"acm": acm, "temp": temps, "ibi_events": ibi_events, "ibi_windows": windows,
            "ibi_correction": corrections, "aggregate_5min": stage_runs}


def main():
    data = json.dumps(build(), separators=(",", ":"), sort_keys=True)
    if "--check" in sys.argv:
        sys.exit(0 if FIXTURE.is_file() and FIXTURE.read_text() == data else 1)
    FIXTURE.write_text(data)
    print(f"wrote {FIXTURE} ({len(data) // 1024} KiB)")


if __name__ == "__main__":
    main()
