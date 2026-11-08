import struct
import unittest

import numpy as np

from ibi_correction import correct_ibis
from nssa_inputs import (collect_nssa_inputs, decode_acm_period, expand_ibi_event, expand_temp_event,
                         inclusive_window)
import make_nssa_fixture


class AcmPeriodTests(unittest.TestCase):
    def test_finger_angles_then_mad_in_model_order(self):
        # tmean 19 + 72/255, max 27 + 208/255, iqr 16 + 45/255 (finger angle, degrees);
        # MAD tmean/max/iqr = 12-bit/4095 + high nibble.
        body = bytes.fromhex("4813D01B2D103B08EF175608")
        mad_iqr, mad_max, mad_tmean, fa_iqr, fa_max, fa_mean = decode_acm_period(body)
        f = np.float32
        self.assertEqual(fa_mean, float(f(0x48) / f(255) + f(0x13)))
        self.assertEqual(fa_max, float(f(0xD0) / f(255) + f(0x1B)))
        self.assertEqual(fa_iqr, float(f(0x2D) / f(255) + f(0x10)))
        self.assertEqual(mad_tmean, float(f(0x3B | (0x08 & 0xF) << 8) / f(4095) + f(0)))
        self.assertEqual(mad_max, float(f(0xEF | (0x17 & 0xF) << 8) / f(4095) + f(1)))
        self.assertEqual(mad_iqr, float(f(0x56 | (0x08 & 0xF) << 8) / f(4095) + f(0)))
        self.assertAlmostEqual(fa_mean, 19.2824, places=4)
        self.assertAlmostEqual(mad_iqr, 0.5211, places=4)

    def test_malformed_bodies_are_skipped(self):
        self.assertIsNone(decode_acm_period(b"\x00" * 11))
        self.assertIsNone(decode_acm_period(None))


class StreamTests(unittest.TestCase):
    def test_sleep_temp_samples_are_30s_apart_ending_at_the_event(self):
        body = struct.pack("<3H", 3536, 3538, 3402)
        self.assertEqual(expand_temp_event(body, 100000),
                         [(40000, float(np.float32(35.36))), (70000, float(np.float32(35.38))),
                          (100000, float(np.float32(34.02)))])
        self.assertEqual(expand_temp_event(b"\x01\x02\x03", 0), [])

    def test_beats_start_their_interval_and_never_go_back(self):
        beats, last = expand_ibi_event([1000, 900, 800, 700, 600, 500], [1, 2, 3, 4, 5, 6], 10000, 0)
        self.assertEqual([b[0] for b in beats], [5500, 6500, 7400, 8200, 8900, 9500])
        self.assertEqual(last, 9500)
        # an overlapping next packet is clamped to the previous stamp
        beats, last = expand_ibi_event([1000] * 6, [0] * 6, 13000, last)
        self.assertEqual([b[0] for b in beats], [9500, 9500, 9500, 10000, 11000, 12000])

    def test_inclusive_window_keeps_first_to_last_inside(self):
        items = [(1, "a"), (5, "b"), (3, "c"), (9, "d")]
        self.assertEqual(inclusive_window(items, 3, 5), [(5, "b"), (3, "c")])
        self.assertEqual(inclusive_window(items, 10, 20), [])

    def test_collect_uses_only_0x60_0x72_0x75_inside_the_bedtime(self):
        acm = bytes(12)
        temp = struct.pack("<2H", 3500, 3510)
        rows = [
            (100, 0x72, {}, 0, acm),
            (200, 0x75, {}, 0, temp),
            (300, 0x60, {"ibi_ms": [1000] * 6, "amplitude": [500] * 6}, 0, None),
            (360, 0x60, {"ibi_ms": [1000] * 6, "amplitude": [500] * 6}, 0, None),
            (420, 0x60, {"ibi_ms": [1000] * 6, "amplitude": [500] * 6}, 0, None),
            (310, 0x80, {"ibi_ms": [900] * 3, "quality": [1] * 3}, 0, None),   # Ring 5 green IBI: not used
            (320, 0x47, {"motion_seconds": 3}, 0, None),                      # motion: not an NSSA input
            (5000, 0x72, {}, 0, acm),                                         # after the bedtime
        ]
        inputs = collect_nssa_inputs(rows, 0, 1000, 0, lambda ds, cu: ds * 100, correct_ibis)
        self.assertEqual(inputs["bedtime"], (0, 100000))
        self.assertEqual(len(inputs["acm"]), 1)
        self.assertEqual([t for t, _ in inputs["temp"]], [20000])  # the first sample precedes the bedtime
        # ecore's correction emits with a short delay: the first beat (validity 0)
        # starts at event time minus its packet, the rest follow 1 s apart.
        self.assertEqual(inputs["ibi"][0], (24000, 1000, 500, 0))
        self.assertTrue(all(v == 1 for _, _, _, v in inputs["ibi"][1:]))


class FixtureTests(unittest.TestCase):
    def test_shared_swift_fixture_is_current(self):
        fixture = make_nssa_fixture.FIXTURE.read_text()
        import json
        self.assertEqual(json.loads(fixture), json.loads(json.dumps(make_nssa_fixture.build())))


if __name__ == "__main__":
    unittest.main()
