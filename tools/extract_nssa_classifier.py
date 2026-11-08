#!/usr/bin/env python3
"""Extract Oura's NSSA sleep classifier from the official app's libalgos.so.

The sleep stager's last step (`oura_ops::oura_sleep_classifier`) is not in the
TorchScript file: it is a treelite-compiled forest (2000 trees, 4 classes, 182
features) baked into native arm64 code. This lifts that code symbolically
(tools/nssa_extract/lift.py: every branch, fcsel/ccmp trick and fadd order), turns
it into an explicit forest (export.py) and writes the flat table
`sleepstaging_2_6_0_classifier.bin` read by tools/nssa_classifier.py and
apps/ios/OuraApp/NssaOps.c. The output is Oura's model: keep it out of git, next to
the other decrypted models.

    pip install capstone pyelftools            # lifting
    pip install unicorn                        # only for --verify
    python tools/extract_nssa_classifier.py config.arm64_v8a.apk [--verify 2000]

The input may be the arm64 split APK (lib/arm64-v8a/libalgos.so inside) or the .so.
--verify runs the native predict_multiclass under unicorn on N random rows and
requires bit-identical probabilities.
"""
import argparse
import os
import pickle
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="libalgos.so or the arm64 split APK containing it")
    ap.add_argument("--out", help="output file (default: <models dir>/sleepstaging_2_6_0_classifier.bin)")
    ap.add_argument("--verify", type=int, default=0, metavar="N", help="compare N rows with the emulated native code")
    args = ap.parse_args()

    from _common import resolve_models_dir
    from nssa_classifier import CLASSIFIER_NAME, Forest, write_forest
    out = Path(args.out) if args.out else resolve_models_dir(REPO, "sleepstaging_2_6_0.pt") / CLASSIFIER_NAME

    with tempfile.TemporaryDirectory() as tmp:
        so = Path(args.source)
        if zipfile.is_zipfile(so):
            with zipfile.ZipFile(so) as z:
                so = Path(z.extract("lib/arm64-v8a/libalgos.so", tmp))
        os.environ["NSSA_LIBALGOS"] = str(so)
        from nssa_extract import export, lift

        t = time.time()
        units = lift.unit_symbols()
        lifted = {}
        for i in sorted(units):
            lifted[i] = lift.Unit(i, *units[i]).lift()
        print(f"lifted {len(lifted)} units in {time.time() - t:.0f}s ({len(lift.NODES)} expression nodes)")
        pkl, js = Path(tmp) / "lifted.pkl", Path(tmp) / "trees.json"
        pickle.dump({"nodes": lift.NODES, "units": lifted}, open(pkl, "wb"))
        export.main(str(pkl), str(js))
        import json
        write_forest(json.load(open(js)), out)
        print(f"wrote {out}")

        if args.verify:
            import numpy as np
            from nssa_extract import emu
            native = emu.Native()
            forest = Forest(out)
            rng = np.random.default_rng(7)
            rows = (rng.normal(size=(args.verify, 182)) * 10.0 ** rng.integers(-2, 3, size=(args.verify, 182)))
            rows[rng.random(rows.shape) < 0.1] = np.nan
            rows = rows.astype(np.float32)
            ours = forest.softmax(forest.margins(rows))
            bad = sum(1 for r in range(len(rows))
                      if list(ours[r].view(np.uint64)) != native.predict(emu.entries_from_row(rows[r])))
            print(f"verify: {len(rows) - bad}/{len(rows)} rows bit-identical to the native predict_multiclass")
            if bad:
                sys.exit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
