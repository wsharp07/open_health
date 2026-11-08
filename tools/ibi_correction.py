"""Exact port of Oura's on-phone IBI correction (official Android app).

Source of truth: libappecore.so (the library EcoreWrapper's JNI binds to; NOT the
older copy in libecore.so, which differs):
  ibi_correction_init_cpp(ctx, cb) -> rr_correction_init(state, {process_corrected_ibi},
      max_out=4, max_gap_ms=5000)  -> rr_correction_init_ex(..., interp_ts=false,
      hr_mode=true, max_hr=150.0)  -> median_error_init_ex(me, 150.0, true)
  ibi_correction_cpp(ibi, amp, ts): state.now_ts = ts; rr_correction_run_ex(state, ibi, amp, extra=0)
  process_corrected_ibi(ibi, raw_amp, amp, validity, ts, extra) -> Kotlin
      IbiAndAmplitudeEvent.Corrected(timestamp=ts, ibi, amplitude=amp, validity)

Validity codes (rr_validity_indicator_t as emitted):
  0 = first beat of the stream (passed through unchecked)
  1 = valid beat (ibi/amp unchanged)
  2 = invalid beat (ibi kept as-is, amplitude forced to 0) — also the median_error
      post-check demotes 1 -> 2
  3 = interpolated / re-distributed beat filling a gap between two valid beats
      (ibi replaced, amplitude 0, timestamp re-computed)
Output is delayed (beats come out 3+3 calls later) and at most 4 beats are emitted per
input call; beats still buffered at the end of the stream are never emitted.

All arithmetic mirrors the arm64 code: integer fields are u16/u8/u32/u64 and every
floating-point operation is IEEE float32 (numpy.float32 here, Float in Swift).
Comparisons are written so NaN behaves like the native branch conditions.
"""
import math
import numpy as np

F = np.float32
CAP = 64
K60000 = F(60000.0)


def f32(x):
    return F(x)


def fcvtas(x):
    """float32 -> int, round to nearest, ties away from zero (arm64 FCVTAS)."""
    x = float(x)
    if math.isnan(x):
        return 0
    r = math.floor(abs(x) + 0.5)
    return int(r if x >= 0 else -r)


def fcvtzs(x):
    x = float(x)
    if math.isnan(x):
        return 0
    return int(math.trunc(x))


class Entry:
    """s_ibi_data_t (32 bytes)."""
    __slots__ = ('ts', 'ts2', 'ibi', 'amp', 'ibi_out', 'amp_out', 'validity', 'ok', 'extra')

    def __init__(self, ts=0, ts2=0, ibi=0, amp=0, ibi_out=0, amp_out=0, validity=0, ok=0, extra=0):
        self.ts, self.ts2, self.ibi, self.amp = ts, ts2, ibi, amp
        self.ibi_out, self.amp_out, self.validity, self.ok, self.extra = ibi_out, amp_out, validity, ok, extra

    def copy(self):
        return Entry(self.ts, self.ts2, self.ibi, self.amp, self.ibi_out, self.amp_out,
                     self.validity, self.ok, self.extra)


class RingBuf:
    """ibirbuffer_t: fixed 64 slots (never cleared), head = next write, tail = oldest."""

    def __init__(self):
        self.slots = [Entry() for _ in range(CAP)]
        self.head = 0
        self.tail = 0
        self.full = False

    def push(self, e):
        self.slots[self.head] = e.copy()
        if self.full:
            self.tail = 0 if self.tail + 1 == CAP else self.tail + 1
        self.head = 0 if self.head + 1 == CAP else self.head + 1
        self.full = self.head == self.tail

    def is_empty(self):
        return (not self.full) and self.head == self.tail

    def pop(self):
        if self.is_empty():
            return None
        e = self.slots[self.tail].copy()
        self.full = False
        self.tail = 0 if self.tail + 1 == CAP else self.tail + 1
        return e

    def count(self):
        if self.full:
            return CAP
        return (self.head + CAP - self.tail) % CAP

    def rfh(self, i):
        """read_from_head(signed char i): 0 = newest, -1 = previous ... (returns the slot itself)."""
        i = ((i + 128) & 0xff) - 128  # sxtb
        if self.is_empty() or abs(i) >= CAP:
            raise RuntimeError('native abort in ibirbuffer_read_from_head')
        idx = i + CAP + self.head - 1
        if idx < 0:
            raise RuntimeError('native abort in ibirbuffer_read_from_head')
        return self.slots[idx % CAP]


def median7_u16(v):
    """Transliteration of median7_u16 (values u16; 0xffff treated as 'missing')."""
    M = 0xffff
    w8, w9, w12 = v[0], v[1], v[2]
    w10 = w9 if w9 > w8 else w8
    w13 = w9 if w9 < w8 else w8
    if w9 == M:
        w10, w9 = M, w8  # csinv -> all ones (only ever compared/used as u16)
    else:
        w9 = w13
    # third
    if w12 == M:
        w8, w10 = w10, M
    elif w12 >= w10:
        w8, w10 = w10, w12
    else:
        w8 = w12 if w12 > w9 else w9
        w9 = w12 if w12 < w9 else w9
    # fourth
    w12 = v[3]
    if w12 == M:
        w11, w8, w10 = w8, w10, M
    elif w12 >= w10:
        w11, w8, w10 = w8, w10, w12
    elif w12 >= w8:
        w11, w8 = w8, w12
    else:
        w11 = w12 if w12 > w9 else w9
        w9 = w12 if w12 < w9 else w9
    # fifth
    w12 = v[4]
    if w12 >= w10:
        w9, w8 = w8, w10
    elif w12 >= w8:
        w9, w8 = w8, w12
    else:
        w10_ = w11
        t = w12 if w12 > w9 else w9
        w11 = t if w12 < w10_ else w11
        w9 = w12 if w12 > w10_ else w10_
    # sixth
    w10 = v[5]
    if w10 < w8:
        a8, a11 = w11, w9
        t = w10 if w10 > a8 else a8
        w9 = t if w10 < a11 else w9
        w8 = w10 if w10 > a11 else a11
    # seventh
    w10 = v[6]
    t = w10 if w10 > w9 else w9
    return t if w10 < w8 else w8


def _s16(x):
    x &= 0xffff
    return x - 0x10000 if x & 0x8000 else x


def median7_s16(v):
    """Transliteration of median7(short*) (signed; 0x7fff treated as 'missing')."""
    M = 0x7fff
    w8, w9r, w12 = _s16(v[0]), _s16(v[1]), _s16(v[2])
    w10u = w9r & 0xffff  # and w10, w9, #0xffff
    w13 = w9r if w9r > w8 else w8
    w14 = w9r if w9r < w8 else w8
    if w10u == M:
        w9, w10 = w10u, w8
    else:
        w9, w10 = w13, w14
    if w12 == M:
        w8, w9 = w9, M
    elif w12 >= _s16(w9):
        w8, w9 = w9, w12
    else:
        w10s = _s16(w10)
        w8 = w12 if w12 > w10s else w10s
        w10 = w12 if w12 < w10s else w10s
    w12 = _s16(v[3])
    if w12 == M:
        w11, w8, w9 = w8, w9, M
    elif w12 >= _s16(w9):
        w11, w8, w9 = w8, w9, w12
    elif w12 >= _s16(w8):
        w11, w8 = w8, w12
    else:
        w10s = _s16(w10)
        w11 = w12 if w12 > w10s else w10s
        w10 = w12 if w12 < w10s else w10s
    w12 = _s16(v[4])
    if w12 >= _s16(w9):
        w10, w8 = w8, w9
    elif w12 >= _s16(w8):
        w10, w8 = w8, w12
    else:
        w9s, w10s = _s16(w10), _s16(w11)
        t = w12 if w12 > w9s else w9s
        w11 = t if w12 < w10s else w11
        w10 = w12 if w12 > w10s else w10s
    w9 = _s16(v[5])
    if w9 < _s16(w8):
        a8, a11 = _s16(w11), _s16(w10)
        t = w9 if w9 > a8 else a8
        w10 = t if w9 < a11 else w10
        w8 = w9 if w9 > a11 else a11
    w9 = _s16(v[6])
    w10s = _s16(w10)
    t = w9 if w9 > w10s else w10s
    return t if w9 < _s16(w8) else w8


def _hr(ibi):
    return K60000 / F(ibi)


class IbiCorrection:
    def __init__(self, max_out=4, max_gap=5000, interp_ts=False, hr_mode=True, max_hr=150.0):
        self.buf1 = RingBuf()           # raw beats
        self.buf2 = RingBuf()           # pending outputs
        self.count = 0                  # i32 @0x1028
        self.max_out = max_out          # u16 @0x102c
        self.max_gap = max_gap          # u16 @0x102e
        self.prev_ibi = 1000            # u16 @0x1030
        self.n_out = 0                  # u16 @0x1032
        self.n_valid = 0                # u16 @0x1034
        self.interp_ts = interp_ts      # u8 @0x1036
        self.hr_mode = hr_mode          # u8 @0x1037
        self.prev_hr = F(60.0)          # f32 @0x1038
        self.max_hr = F(max_hr)         # f32 @0x103c
        self.bad_run = 0                # u8 @0x1040
        self.saved_bad_run = 0          # u8 @0x1041
        z = [0] * 7
        self.dl_ibi, self.dl_raw_amp, self.dl_amp = list(z), list(z), list(z)
        self.dl_val, self.dl_ts, self.dl_extra = list(z), list(z), list(z)
        # median_error state @0x10c8
        self.me_prev_hr = F(0.0)
        self.me_max_hr = F(max_hr)
        self.me_flag = bool(hr_mode)
        self.me_prev_ibi = 1000
        self.now_ts = 0                 # u64 @0x10d8
        self.ms_flag = True             # u8 @0x10e0
        self.out = []

    # --- helpers --------------------------------------------------------
    def _var7_buf1(self):
        """0x1770dc: variance of HR over buf1[-6..0] (no zero guard)."""
        hs = [_hr(self.buf1.rfh(i).ibi) for i in range(-6, 1)]
        s = hs[0]
        for h in hs[1:]:
            s = F(s + h)
        mean = F(s / F(7.0))
        acc = None
        for h in hs:
            d = F(h - mean)
            d2 = F(d * d)
            acc = d2 if acc is None else F(acc + d2)
        return F(acc / F(6.0))

    def _median_error(self):
        A, V = self.dl_ibi, self.dl_val
        if V[3] != 1:
            return
        w22 = A[3]
        s8 = _hr(w22)
        if self.me_prev_hr == F(0.0):
            self.me_prev_hr = s8
        med = _s16(median7_s16(A))
        tot = sum(_s16(a) for a in A)
        mean = F(F(tot) / F(7.0))
        s1 = F(K60000 / F(med))
        s0 = F(K60000 / F(fcvtzs(mean)))
        bad = False
        if self.me_flag:
            if s8 <= F(30.0) or s8 > self.me_max_hr or abs(F(s8 - s1)) > F(18.0):
                bad = True
            elif s8 > F(F(self.me_prev_hr + F(23.0)) + F(4.0)):
                bad = True
            elif s8 < F(self.me_prev_hr * F(0.625)):
                bad = True
            elif s8 < F(F(self.me_prev_hr + F(-16.0)) + F(-4.0)):
                bad = True
            else:
                d = F(s0 - s8)
                if F(d * d) > F(self._var7_dl() * F(5.0)):
                    bad = True
        else:
            if s8 <= F(30.0) or s8 > self.me_max_hr or abs(F(s8 - s1)) > F(18.0):
                bad = True
            elif abs(w22 - self.me_prev_ibi) * 100 > self.me_prev_ibi * 25:
                bad = True
            else:
                d = F(s0 - s8)
                if F(d * d) > F(self._var7_dl() * F(5.0)):
                    bad = True
                elif not (self.dl_extra[3] < 7):
                    bad = True
        if bad:
            V[3] = 2
        self.me_prev_hr = s8
        self.me_prev_ibi = w22

    def _var7_dl(self):
        """median_error_ex helper: HR variance over the 7-beat delay line (0 for ibi==0)."""
        A = self.dl_ibi
        hs = [(_hr(a) if a else F(0.0)) for a in A]
        s = F(0.0) if A[0] == 0 else hs[0]
        for i in range(1, 7):
            if A[i]:
                s = F(s + hs[i])
        mean = F(s / F(7.0))
        acc = None
        for h in hs:
            d = F(h - mean)
            d2 = F(d * d)
            acc = d2 if acc is None else F(acc + d2)
        return F(acc / F(6.0))

    def _flush(self, start, n, dec):
        w20, w23 = start, n
        while True:
            e = self.buf1.rfh(w20)
            o = e.copy()
            o.ibi_out, o.amp_out, o.validity, o.ts2 = e.ibi, 0, 2, e.ts
            self.buf2.push(o)
            e.ibi = 0
            e.amp = 0
            e.validity = 0
            self.buf1.pop()
            if dec:
                self.bad_run = (self.bad_run - 1) & 0xff
            w23 = (w23 - 1) & 0xffffffff
            w20 += 1
            if (w23 & 0xff) == 0:
                break

    def _after_gap(self, w25):
        """L870 (w25 > -64 always here) -> flush long bad runs, then L924."""
        if self.bad_run > 60:
            self._flush(-63, self.bad_run, True)
        self._emit_last()

    def _emit_last(self):
        e = self.buf1.rfh(-3)
        if e.ok == 1:
            o = e.copy()
            o.validity, o.ibi_out, o.amp_out, o.ts2 = 1, e.ibi, e.amp, e.ts
            self.buf2.push(o)

    # --- main -------------------------------------------------------------
    def run(self, ibi, amp, ts, extra=0):
        ibi &= 0xffff
        amp &= 0xffff
        self.now_ts = ts & 0xffffffffffffffff
        if ibi != 0:
            self._ingest(ibi, amp, extra)
        self._output()

    def _ingest(self, ibi, amp, extra):
        self.count += 1
        new = Entry(ts=self.now_ts, ibi=ibi, amp=amp, ok=1, extra=extra)
        self.buf1.push(new)
        if self.count == 1:
            o = new.copy()
            o.ibi_out, o.amp_out, o.validity, o.ts2 = ibi, amp, 0, self.now_ts
            if ((new.ts - 1) & 0xffffffffffffffff) <= 99999999998:
                self.ms_flag = False
            self.buf2.push(o)
            return
        if self.count < 5:
            return
        e = self.buf1.rfh(-3)
        s8 = _hr(e.ibi) if e.ibi else K60000
        n = self.buf1.count()
        if n <= 6:
            bad = False
            if self.hr_mode:
                if s8 <= F(30.0) or s8 > self.max_hr:
                    bad = True
                elif s8 < F(self.prev_hr * F(0.625)) or s8 > F(self.prev_hr + F(18.0)):
                    bad = True
                elif s8 < F(self.prev_hr + F(-25.0)):
                    bad = True
            else:
                if s8 <= F(30.0) or s8 > self.max_hr:
                    bad = True
                elif abs(e.ibi - self.prev_ibi) * 100 > self.prev_ibi * 20:
                    bad = True
                elif not (e.extra < 7):
                    bad = True
            if bad:
                e.ok = 0
                self.bad_run = (self.bad_run + 1) & 0xff
                update = True
            else:
                e.ok = 1
                self.saved_bad_run = self.bad_run
                self.bad_run = 0
                update = True
        else:
            m = median7_u16([self.buf1.rfh(i).ibi for i in range(-6, 1)])
            mm = m if (m & 0xfffe) else 1
            s9 = F(K60000 / F(mm))
            tot = sum(self.buf1.rfh(i).ibi for i in range(-6, 1))
            mean = F(F(tot) / F(7.0))
            s0 = F(K60000 / mean) if mean != F(0.0) else F(0.0)
            s0 = F(s0 - s8)
            bad = False
            if s8 <= F(30.0) or s8 > self.max_hr or abs(F(s8 - s9)) > F(14.0):
                bad = True
            elif self.hr_mode:
                if s8 < F(self.prev_hr * F(0.625)) or s8 > F(self.prev_hr + F(23.0)):
                    bad = True
                elif s8 < F(self.prev_hr + F(-16.0)):
                    bad = True
                elif F(s0 * s0) > F(self._var7_buf1() * F(1.5)):
                    bad = True
            else:
                if abs(e.ibi - self.prev_ibi) * 100 > self.prev_ibi * 20:
                    bad = True
                elif F(s0 * s0) > F(self._var7_buf1() * F(1.5)):
                    bad = True
                elif not (e.extra < 7):
                    bad = True
            if bad:
                e.ok = 0
                self.bad_run = (self.bad_run + 1) & 0xff
                update = self.bad_run >= 4
            else:
                e.ok = 1
                self.saved_bad_run = self.bad_run
                self.bad_run = 0
                update = True
        if update:
            self.prev_hr = s8
            self.prev_ibi = e.ibi
        a, b = self.buf1.rfh(-3), self.buf1.rfh(-4)
        if a.ok == 1 and b.ok == 0:
            self._gap()
            return
        if self.bad_run > 60:
            self._flush(-63, self.bad_run, True)
        self._emit_last()

    def _gap(self):
        w24 = 0
        while True:
            g = w24 - 4
            if self.buf1.rfh(g).ok:
                break
            w8 = w24 - 5
            w24 -= 1
            if not (-63 <= w8 <= -1):
                # no earlier valid beat in the buffer
                if self.bad_run > 60:
                    self._flush(-63, self.bad_run, True)
                elif self.saved_bad_run:
                    self._flush(-3 - self.saved_bad_run, self.saved_bad_run, False)
                self._emit_last()
                return
        w21 = w24 - 3
        ge = self.buf1.rfh(g)
        gi, gts = ge.ibi, ge.ts
        le = self.buf1.rfh(-3)
        li, lts = le.ibi, le.ts
        sumbad = 0
        if g <= -5:
            for k in range(w21, -3):
                sumbad += self.buf1.rfh(k).ibi
        w11 = li + gi
        nfill = fcvtas(F(F(sumbad) / F(w11 >> 1)))
        if nfill <= 1:
            nfill = 1
        nbad = -w24
        same = nfill == nbad
        if sumbad > self.max_gap or (1 - w24) > self.max_out:
            last_ts = 0
            e = self.buf1.rfh(w21)
            if not e.ok:
                while True:
                    e = self.buf1.rfh(w21)
                    o = e.copy()
                    o.ibi_out, o.amp_out, o.validity, o.ts2 = e.ibi, 0, 2, e.ts
                    self.buf2.push(o)
                    w21 += 1
                    last_ts = e.ts
                    if self.buf1.rfh(w21).ok:
                        break
            if self.max_gap != 0:
                self.buf2.push(Entry(ts2=last_ts, validity=2))
            self._after_gap(g)
            return
        # fill / redistribute the gap
        step = fcvtas(F(F(li - gi) / F(nfill + 1)))
        end = nfill + w24 - 4
        cur = step + gi
        cum = 0
        last_ts = 0
        w20 = g
        while True:
            w22 = w20 + 1
            e = self.buf1.rfh(w22)
            o = e.copy()
            if nbad >= nfill or w22 < -3:
                last_ts = e.ts
            else:
                o = Entry(ts=last_ts)
            o.ibi_out = cur & 0xffff
            o.amp_out = 0
            o.validity = 3
            self.buf2.push(o)
            w20 += 1
            cum += cur
            cur += step
            if w20 >= end:
                break
        if self.interp_ts and ((w11 & 0xffffffff) >= 2):
            keep = same  # (the interp-ts branch; unused by the app: interp_ts=false)
            self._restore_or_validate(keep, nfill, g, end, w21)
            return
        diff = sumbad - cum
        q = F(F(diff) / F(nfill))
        q = F(math.ceil(q)) if diff < 0 else F(math.floor(q))
        w11q = fcvtzs(q)
        scale = F(1000.0) if not self.ms_flag else F(1.0)
        r = diff - nfill * w11q
        w23 = 1 - nfill
        keep = same
        cts = 0
        w24b = g
        while True:
            e2 = self.buf2.rfh(w23)
            if r == 0:
                v = e2.ibi_out + w11q
            elif r < 0:
                v = (w11q - 1) + e2.ibi_out
                r += 1
            else:
                v = (w11q + 1) + e2.ibi_out
                r -= 1
            v &= 0xffffffff
            e2.ibi_out = v & 0xffff
            cts = (cts + (v & 0xffff)) & 0xffffffffffffffff
            t = (fcvtas(F(F(cts) / scale)) + gts) & 0xffffffffffffffff
            e2.ts2 = lts if t > lts else t
            if keep:
                d = abs(self.buf1.rfh(end).ibi - e2.ibi_out)
                keep = d < 126
            else:
                keep = False
            w24b += 1
            w23 += 1
            if w24b >= end:
                break
        if keep:
            self._restore_or_validate(True, nfill, g, end, w21)
            return
        self._after_gap(g)

    def _restore_or_validate(self, keep, nfill, g, end, w21):
        if not keep:
            self._after_gap(g)
            return
        w23, w24 = g, 1 - nfill
        while True:
            w23 += 1
            v = self.buf1.rfh(w23).ibi
            self.buf2.rfh(w24).ibi_out = v
            w24 += 1
            if not (w23 < end):
                break
        li = self.buf1.rfh(-3).ibi
        gi = self.buf1.rfh(g).ibi
        s9 = F(F(K60000 / F(min(li, gi))) + F(34.5))
        mn, mx = 0xffff, 0
        if w21 <= -4:
            for i in range(w21, -3):
                v = self.buf1.rfh(i).ibi
                if mn >= v:
                    mn = v
                if mx < v:
                    mx = v
        else:
            mn, mx = 0xffff, 0
        s8 = F(K60000 / F(mn))
        good = False
        if s8 < s9:
            s0 = F(F(K60000 / F(max(li, gi))) + F(-24.0))
            lim = F(np.frombuffer(bytes.fromhex('cdccf041'), dtype=np.float32)[0])
            if s0 < lim:
                s0 = lim
            s1 = F(K60000 / F(mx))
            if s1 > s0:
                good = True
        w22 = 1 - nfill
        w23 = g
        while True:
            e2 = self.buf2.rfh(w22)
            e2.validity = 1 if good else 2
            if good:
                e2.amp_out = e2.amp
                e2.ts2 = e2.ts
            w23 += 1
            w22 += 1
            if w23 >= end:
                break
        self._after_gap(g)

    def _output(self):
        if self.buf2.is_empty():
            return
        outs = 0
        while True:
            if outs >= self.max_out:
                return
            e = self.buf2.pop()
            if e is None:
                raise RuntimeError('native assert: pop failed')
            if e.ibi_out != 0 and self.count >= 0:
                for arr, val in ((self.dl_ibi, e.ibi_out), (self.dl_raw_amp, e.amp),
                                 (self.dl_amp, e.amp_out), (self.dl_val, e.validity),
                                 (self.dl_ts, e.ts2), (self.dl_extra, e.extra)):
                    del arr[0]
                    arr.append(val)
                if self.dl_ibi[0] != 0:
                    self._median_error()
                if self.dl_ibi[3] != 0:
                    self.n_out = (self.n_out + 1) & 0xffff
                    if self.dl_extra[3] <= 1 and self.dl_val[3] == 1:
                        self.n_valid = (self.n_valid + 1) & 0xffff
                    self.out.append((self.dl_ts[3], self.dl_ibi[3], self.dl_amp[3], self.dl_val[3] & 0xff))
                    outs += 1
            if self.buf2.is_empty():
                return


def correct_ibis(raw):
    """raw: iterable of (timestamp_ms, ibi_ms, amplitude) in ring order.
    Returns [(timestamp_ms, ibi, amplitude, validity)] exactly as the official app's
    ecore.correctIbiAndAmplitudeEvents does (fresh state per call)."""
    c = IbiCorrection()
    for ts, ibi, amp in raw:
        c.run(int(ibi), int(amp), int(ts))
    return c.out
