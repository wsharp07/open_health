"""Oura's NSSA TorchScript custom ops (libalgos.so, namespace ``oura_ops``) so that
``sleepstaging_2_6_0.pt`` loads and runs with stock torch.

Two implementations of each op, kept bit-identical by tools/test_nssa_ops.py:
* a pure-Python reference (below), transcribed from the arm64 code;
* apps/ios/OuraApp/NssaOps.c — the same C the iOS app compiles — built on first use
  with the system C compiler and loaded through ctypes. Set OURA_NSSA_PURE=1 to force
  the Python path (it is ~50x slower on the biquad).
Semantics and addresses: docs/algorithms/sleep-staging.md.

Import this module before ``torch.jit.load``.
"""
import ctypes
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from _common import resolve_models_dir
from nssa_classifier import CLASSIFIER_NAME, Forest

REPO = Path(__file__).resolve().parent.parent
C_SOURCE = REPO / "apps" / "ios" / "OuraApp" / "NssaOps.c"

_LIB = torch.library.Library("oura_ops", "DEF")
_LIB.define("oura_create_windows(Tensor timestamps, float window_s) -> (Tensor, Tensor)")
_LIB.define("oura_find_indices(Tensor x, Tensor xnew) -> Tensor")
_LIB.define("oura_biquad_cascade(Tensor sos, Tensor x) -> Tensor")
_LIB.define("oura_find_peaks(Tensor x) -> Tensor")
_LIB.define("oura_sleep_classifier(Tensor features) -> Tensor")


# ------------------------------------------------------------------ native loader
_native = None


def native():
    """ctypes handle on NssaOps.c compiled for this machine, or None."""
    global _native
    if _native is not None:
        return _native or None
    _native = False
    if os.environ.get("OURA_NSSA_PURE") or not C_SOURCE.is_file():
        return None
    src = C_SOURCE.read_bytes() + (C_SOURCE.with_suffix(".h")).read_bytes()
    tag = hashlib.sha256(src).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "open_health"
    lib = cache / f"nssa_ops_{tag}{'.dylib' if os.uname().sysname == 'Darwin' else '.so'}"
    if not lib.is_file():
        try:
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / lib.name
                subprocess.run([os.environ.get("CC", "cc"), "-O2", "-ffp-contract=off", "-shared", "-fPIC",
                                str(C_SOURCE), "-o", str(out), "-lm"],
                               check=True, capture_output=True, timeout=120)
                os.replace(out, lib)
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        h = ctypes.CDLL(str(lib))
    except OSError:
        return None
    p64, pf, pu8 = ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)
    h.nssa_create_windows.restype = ctypes.c_int64
    h.nssa_create_windows.argtypes = [p64, ctypes.c_int64, ctypes.c_double, p64, p64]
    h.nssa_find_indices.restype = ctypes.c_int
    h.nssa_find_indices.argtypes = [p64, ctypes.c_int64, p64, ctypes.c_int64, p64]
    h.nssa_biquad_cascade.restype = None
    h.nssa_biquad_cascade.argtypes = [pf, ctypes.c_int64, pf, ctypes.c_int64, pf]
    h.nssa_find_peaks.restype = ctypes.c_int64
    h.nssa_find_peaks.argtypes = [pf, ctypes.c_int64, p64]
    h.nssa_classifier_predict.restype = ctypes.c_int
    h.nssa_classifier_predict.argtypes = [pu8, ctypes.c_int64, pf, ctypes.c_int64, ctypes.c_int64, pf]
    _native = h
    return h


def _ptr(a, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


# ------------------------------------------------------------------ pure Python
def _i32(v):
    """Truncate to a signed 32-bit int (the native code keeps indices in w regs)."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _cdiv(a, b):
    """C99 integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _crem(a, b):
    return a - _cdiv(a, b) * b


def py_create_windows(a, window_s):
    """nssaops::create_windows @0x5602fc over a list of N ms stamps. Returns
    (window end stamps, rows of [5-min lookback, 1-min lookback, start, end))."""
    size = len(a)
    W = int(math.trunc(float(window_s) * 1000.0))
    first, last = a[0], a[size - 1]
    start = first - _crem(_crem(first, W) + W, W)
    end = last + W - _crem(last, W)
    n = _i32(_cdiv(end - start, W))
    rows = [[-1, -1, -1, -1] for _ in range(max(n, 0))]
    wts = [0] * max(n, 0)
    if end <= start:
        return wts, rows
    p_start = p_end = p_1 = p_5 = 0
    cur, k = start, 0
    while True:
        while p_start < size and a[p_start] < cur:
            p_start += 1
        while p_end < size and a[p_end] < cur + W - 1:
            p_end += 1
        while p_1 < size and a[p_1] < cur + W - 60000:
            p_1 += 1
        while p_5 < size and a[p_5] < cur + W - 300000:
            p_5 += 1
        nxt = cur + W
        if k < n:
            row = [p_5, p_1, p_start, p_end]
            if nxt - 300000 < start:
                row[0] = -1
            if nxt - 60000 < start:
                row[1] = row[2] = row[3] = -1
            rows[k] = row
            wts[k] = nxt
        k += 1
        cur = nxt
        if not cur < end:
            break
    return wts, rows


def py_find_indices(xs, vs):
    """nssaops::find_indices @0x561248: running int32 "low" index for interp1d; the
    first probe of each value is x[j+1]. Raises where the native code reads past x."""
    out, j, nx = [], 0, len(xs)
    for v in vs:
        m = j + 1
        while True:
            if m >= nx:
                raise IndexError("oura_find_indices: native code would read past x")
            if v > xs[m]:
                m += 1
                continue
            break
        j = _i32(m - 1)
        out.append(j)
    return out


def _r32(v):
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float32(v))


def _fma32(a, b, c):
    """Correctly rounded float32 fma(a, b, c) for float32 inputs (AArch64 fmadd s)."""
    p = a * b  # exact in binary64 (24x24 bit product)
    s = p + c
    if not (math.isfinite(p) and math.isfinite(c) and math.isfinite(s)):
        return _r32(s)
    bb = s - p
    e = (p - (s - bb)) + (c - bb)  # TwoSum: p + c == s + e exactly
    r = _r32(s)
    if e != 0.0 and r != s:
        if r < s:
            lo, hi = r, float(np.nextafter(np.float32(r), np.float32(math.inf)))
        else:
            lo, hi = float(np.nextafter(np.float32(r), np.float32(-math.inf))), r
        if s - lo == hi - s:  # s sits on a float32 midpoint: the exact sum decides
            r = hi if e > 0 else lo
    return r


def py_biquad_cascade(sos_rows, vals):
    """nssaops::biquad_cascade @0x55e860 + Biquad<float>::step @0x564234: TDF-II,
    coefficients divided by a0 in float32, zero state, float32 fmadd:
    y = fma(x,b0,z1); z1 = z2 + fma(x,b1,-(a1*y)); z2 = fma(x,b2,-(a2*y))."""
    secs = [[_r32(b0 / a0), _r32(b1 / a0), _r32(b2 / a0), _r32(a1 / a0), _r32(a2 / a0), 0.0, 0.0]
            for b0, b1, b2, a0, a1, a2 in sos_rows]
    out = []
    for v in vals:
        for sec in secs:
            b0, b1, b2, a1, a2, z1, z2 = sec
            y = _fma32(v, b0, z1)
            sec[5] = _r32(z2 + _fma32(v, b1, -_r32(a1 * y)))
            sec[6] = _fma32(v, b2, -_r32(a2 * y))
            v = y
        out.append(v)
    return out


def py_find_peaks(v):
    """nssaops::find_peaks @0x55ee1c == scipy _local_maxima_1d (float32 compares):
    strict rise, flat plateau allowed, strict fall; plateau midpoints."""
    i_max = _i32(len(v)) - 1
    mids, i = [], 1
    if i_max >= 2:
        while i < i_max:
            if v[i - 1] < v[i]:
                ia = i + 1
                while ia < i_max and v[ia] == v[i]:
                    ia += 1
                if v[ia] < v[i]:
                    mids.append((i + ia - 1) // 2)
                    i = ia
            i += 1
    return mids


# ------------------------------------------------------------------ torch ops
def create_windows(ts, window_s):
    a = np.ascontiguousarray(ts.reshape(-1).numpy(), dtype=np.int64)
    h = native()
    if h is not None and len(a):
        n = h.nssa_create_windows(_ptr(a, ctypes.c_int64), len(a), float(window_s), None, None)
        wts = np.zeros(max(n, 0), np.int64)
        rows = np.full((max(n, 0), 4), -1, np.int64)
        h.nssa_create_windows(_ptr(a, ctypes.c_int64), len(a), float(window_s),
                              _ptr(wts, ctypes.c_int64), _ptr(rows, ctypes.c_int64))
        return torch.from_numpy(wts).reshape(-1, 1), torch.from_numpy(rows)
    wts, rows = py_create_windows(a.tolist(), window_s)
    return (torch.tensor(wts, dtype=torch.long).reshape(-1, 1),
            torch.tensor(rows, dtype=torch.long).reshape(-1, 4))


def find_indices(x, xnew):
    xs = np.ascontiguousarray(x.reshape(-1).numpy(), dtype=np.int64)
    vs = np.ascontiguousarray(xnew.reshape(-1).numpy(), dtype=np.int64)
    h = native()
    if h is not None:
        out = np.zeros(len(vs), np.int64)
        if h.nssa_find_indices(_ptr(xs, ctypes.c_int64), len(xs), _ptr(vs, ctypes.c_int64), len(vs),
                               _ptr(out, ctypes.c_int64)) != 0:
            raise IndexError("oura_find_indices: native code would read past x")
        return torch.from_numpy(out).reshape(xnew.shape)
    return torch.tensor(py_find_indices(xs.tolist(), vs.tolist()), dtype=torch.long).reshape(xnew.shape)


def biquad_cascade(sos, x):
    s = np.ascontiguousarray(sos.to(torch.float32).numpy())
    v = np.ascontiguousarray(x.to(torch.float32).reshape(-1).numpy())
    h = native()
    if h is not None and s.shape[0] <= 64:
        out = np.empty_like(v)
        h.nssa_biquad_cascade(_ptr(s, ctypes.c_float), s.shape[0], _ptr(v, ctypes.c_float), len(v),
                              _ptr(out, ctypes.c_float))
        return torch.from_numpy(out).reshape(x.shape).to(x.dtype)
    out = py_biquad_cascade(s.tolist(), v.tolist())
    return torch.tensor(out, dtype=torch.float32).reshape(x.shape).to(x.dtype)


def find_peaks(x):
    v = np.ascontiguousarray(x.to(torch.float32).reshape(-1).numpy())
    h = native()
    if h is not None:
        out = np.zeros(max(len(v) // 2, 1), np.int64)
        m = h.nssa_find_peaks(_ptr(v, ctypes.c_float), len(v), _ptr(out, ctypes.c_int64))
        return torch.from_numpy(out[:m].copy())
    return torch.tensor(py_find_peaks(v.tolist()), dtype=torch.long)


_forest = None


def forest():
    global _forest
    if _forest is None:
        path = resolve_models_dir(REPO, CLASSIFIER_NAME) / CLASSIFIER_NAME
        if not path.is_file():
            raise RuntimeError(f"missing {path}: run tools/extract_nssa_classifier.py on Oura's libalgos.so")
        _forest = Forest(path)
    return _forest


def sleep_classifier(features):
    """nssaops::sleep_classifier: float32 [N,182] -> float32 [N,4] class probabilities."""
    f = np.ascontiguousarray(features.to(torch.float32).numpy())
    if f.ndim != 2 or f.shape[1] != 182:
        raise RuntimeError(f"oura_sleep_classifier expects [N,182], got {tuple(f.shape)}")
    fo = forest()
    h = native()
    if h is not None and len(f):
        blob = np.frombuffer(fo.blob, np.uint8)
        out = np.empty((len(f), 4), np.float32)
        if h.nssa_classifier_predict(_ptr(blob, ctypes.c_uint8), len(blob), _ptr(f, ctypes.c_float),
                                     len(f), f.shape[1], _ptr(out, ctypes.c_float)) == 0:
            return torch.from_numpy(out)
    return torch.from_numpy(fo.predict(f))


_LIB.impl("oura_create_windows", create_windows, "CompositeExplicitAutograd")
_LIB.impl("oura_find_indices", find_indices, "CompositeExplicitAutograd")
_LIB.impl("oura_biquad_cascade", biquad_cascade, "CompositeExplicitAutograd")
_LIB.impl("oura_find_peaks", find_peaks, "CompositeExplicitAutograd")
_LIB.impl("oura_sleep_classifier", sleep_classifier, "CompositeExplicitAutograd")
