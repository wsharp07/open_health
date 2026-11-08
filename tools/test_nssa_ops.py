"""The C ops (apps/ios/OuraApp/NssaOps.c, shared with iOS) must equal the Python
reference bit for bit. Skips the native half when no C compiler is available."""
import random
import unittest

import numpy as np
import torch

import nssa_ops
from nssa_ops import py_biquad_cascade, py_create_windows, py_find_indices, py_find_peaks


@unittest.skipIf(nssa_ops.native() is None, "no C compiler for NssaOps.c")
class NativeMatchesReference(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)

    def test_create_windows(self):
        for _ in range(200):
            n = self.rng.randint(1, 400)
            t = 1792100000000 + self.rng.randint(-10 ** 6, 10 ** 6)
            ts = []
            for _ in range(n):
                t += self.rng.choice([0, 1, 250, 999, 30000, 29999, 61000, 400000])
                ts.append(t)
            w = self.rng.choice([30.0, 60.0, 0.5, 300.0])
            got_ts, got_rows = nssa_ops.create_windows(torch.tensor(ts).reshape(-1, 1), w)
            ref_ts, ref_rows = py_create_windows(ts, w)
            self.assertEqual(got_ts.reshape(-1).tolist(), ref_ts)
            self.assertEqual(got_rows.tolist(), ref_rows)

    def test_find_indices(self):
        for _ in range(300):
            x = sorted(self.rng.randint(0, 5000) for _ in range(self.rng.randint(2, 60)))
            vs = sorted(self.rng.randint(x[0], x[-1]) for _ in range(self.rng.randint(0, 80)))
            try:
                ref = py_find_indices(x, vs)
            except IndexError:
                with self.assertRaises(IndexError):
                    nssa_ops.find_indices(torch.tensor(x), torch.tensor(vs, dtype=torch.long))
                continue
            got = nssa_ops.find_indices(torch.tensor(x), torch.tensor(vs, dtype=torch.long))
            self.assertEqual(got.tolist(), ref)

    def test_biquad_cascade_bits(self):
        rng = np.random.default_rng(3)
        for _ in range(60):
            k = int(rng.integers(1, 5))
            sos = rng.normal(size=(k, 6)).astype(np.float32)
            sos[:, 3] = rng.uniform(0.5, 2.0, k).astype(np.float32)
            sos[:, 4:] *= np.float32(0.3)
            x = (rng.normal(size=int(rng.integers(1, 400))) * 10 ** rng.uniform(-3, 3)).astype(np.float32)
            got = nssa_ops.biquad_cascade(torch.from_numpy(sos), torch.from_numpy(x)).numpy()
            ref = np.array(py_biquad_cascade(sos.tolist(), x.tolist()), dtype=np.float32)
            # unstable random filters may blow up to NaN, whose payload is not meaningful
            nan = np.isnan(ref)
            self.assertTrue(np.array_equal(nan, np.isnan(got)))
            self.assertTrue(np.array_equal(got[~nan].view(np.uint32), ref[~nan].view(np.uint32)))

    def test_find_peaks(self):
        for _ in range(400):
            v = [float(np.float32(self.rng.choice([0, 1, 1, 2, 3, self.rng.random(), float("nan")])))
                 for _ in range(self.rng.randint(0, 40))]
            got = nssa_ops.find_peaks(torch.tensor(v, dtype=torch.float32)).tolist()
            self.assertEqual(got, py_find_peaks([float(np.float32(a)) for a in v]))


class ClassifierTests(unittest.TestCase):
    def test_native_and_numpy_forest_agree(self):
        try:
            forest = nssa_ops.forest()
        except RuntimeError:
            self.skipTest("classifier not extracted (private Oura weights)")
        rng = np.random.default_rng(11)
        f = (rng.normal(size=(300, 182)) * 3).astype(np.float32)
        f[rng.random(f.shape) < 0.1] = np.nan
        ref = forest.predict(f)
        got = nssa_ops.sleep_classifier(torch.from_numpy(f)).numpy()
        self.assertTrue(np.array_equal(got.view(np.uint32), ref.view(np.uint32)))
        self.assertTrue(np.allclose(ref.sum(1), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
