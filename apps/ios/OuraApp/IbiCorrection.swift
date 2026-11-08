import Foundation

// Exact Swift twin of tools/ibi_correction.py: Oura's on-phone IBI correction
// (libappecore.so rr_correction_run_ex, as the official app runs it before NSSA).
// Every floating-point step is Float (IEEE float32) in the same order as the Python
// port, which is verified bit-for-bit against the emulated native code. Validity:
// 0 first beat, 1 valid, 2 invalid (amplitude 0), 3 refilled gap beat.
// Tests: Tests/NssaParityTests.swift against Tests/NssaParityFixture.json.
enum IbiCorrection {
    /// raw: (timestamp ms, ibi ms, amplitude) in ring order → (ts, ibi, amp, validity).
    static func correct(_ raw: [(Int64, Int, Int)]) -> [(Int64, Int, Int, Int)] {
        let c = State()
        for (ts, ibi, amp) in raw { c.run(ibi, amp, ts) }
        return c.out
    }

    private static let cap = 64
    private static let k60000: Float = 60000.0

    private static func hr(_ ibi: Int) -> Float { k60000 / Float(ibi) }

    /// float32 → int, nearest with ties away from zero (FCVTAS); NaN → 0, saturating.
    static func fcvtas(_ x: Float) -> Int {
        if x.isNaN { return 0 }
        let r = x.rounded(.toNearestOrAwayFromZero)
        if r >= 9.2e18 { return Int.max }
        if r <= -9.2e18 { return Int.min }
        return Int(r)
    }
    static func fcvtzs(_ x: Float) -> Int {
        if x.isNaN { return 0 }
        let r = x.rounded(.towardZero)
        if r >= 9.2e18 { return Int.max }
        if r <= -9.2e18 { return Int.min }
        return Int(r)
    }

    final class Entry {
        var ts: Int64 = 0, ts2: Int64 = 0
        var ibi = 0, amp = 0, ibiOut = 0, ampOut = 0, validity = 0, ok = 0, extra = 0
        init(ts: Int64 = 0, ts2: Int64 = 0, ibi: Int = 0, amp: Int = 0, ok: Int = 0, extra: Int = 0, validity: Int = 0) {
            self.ts = ts; self.ts2 = ts2; self.ibi = ibi; self.amp = amp; self.ok = ok; self.extra = extra
            self.validity = validity
        }
        func copy() -> Entry {
            let e = Entry(ts: ts, ts2: ts2, ibi: ibi, amp: amp, ok: ok, extra: extra, validity: validity)
            e.ibiOut = ibiOut; e.ampOut = ampOut
            return e
        }
    }

    final class RingBuf {
        var slots: [Entry] = (0..<cap).map { _ in Entry() }
        var head = 0, tail = 0, full = false
        func push(_ e: Entry) {
            slots[head] = e.copy()
            if full { tail = tail + 1 == cap ? 0 : tail + 1 }
            head = head + 1 == cap ? 0 : head + 1
            full = head == tail
        }
        var isEmpty: Bool { !full && head == tail }
        func pop() -> Entry? {
            if isEmpty { return nil }
            let e = slots[tail].copy()
            full = false
            tail = tail + 1 == cap ? 0 : tail + 1
            return e
        }
        var count: Int { full ? cap : (head + cap - tail) % cap }
        /// read_from_head(signed char i): 0 = newest, -1 = previous … (the slot itself).
        func rfh(_ i0: Int) -> Entry {
            let i = ((i0 + 128) & 0xff) - 128
            precondition(!isEmpty && abs(i) < cap, "native abort in ibirbuffer_read_from_head")
            let idx = i + cap + head - 1
            precondition(idx >= 0, "native abort in ibirbuffer_read_from_head")
            return slots[idx % cap]
        }
    }

    static func median7U16(_ v: [Int]) -> Int {
        let M = 0xffff
        var w8 = v[0], w9 = v[1], w12 = v[2]
        var w10 = w9 > w8 ? w9 : w8
        let w13 = w9 < w8 ? w9 : w8
        var w11 = 0
        if w9 == M { w10 = M; w9 = w8 } else { w9 = w13 }
        if w12 == M { w8 = w10; w10 = M }
        else if w12 >= w10 { w8 = w10; w10 = w12 }
        else { w8 = w12 > w9 ? w12 : w9; w9 = w12 < w9 ? w12 : w9 }
        w12 = v[3]
        if w12 == M { w11 = w8; w8 = w10; w10 = M }
        else if w12 >= w10 { w11 = w8; w8 = w10; w10 = w12 }
        else if w12 >= w8 { w11 = w8; w8 = w12 }
        else { w11 = w12 > w9 ? w12 : w9; w9 = w12 < w9 ? w12 : w9 }
        w12 = v[4]
        if w12 >= w10 { w9 = w8; w8 = w10 }
        else if w12 >= w8 { w9 = w8; w8 = w12 }
        else {
            let w10_ = w11
            let t = w12 > w9 ? w12 : w9
            w11 = w12 < w10_ ? t : w11
            w9 = w12 > w10_ ? w12 : w10_
        }
        w10 = v[5]
        if w10 < w8 {
            let a8 = w11, a11 = w9
            let t = w10 > a8 ? w10 : a8
            w9 = w10 < a11 ? t : w9
            w8 = w10 > a11 ? w10 : a11
        }
        w10 = v[6]
        let t = w10 > w9 ? w10 : w9
        return w10 < w8 ? t : w8
    }

    static func s16(_ x0: Int) -> Int {
        let x = x0 & 0xffff
        return x & 0x8000 != 0 ? x - 0x10000 : x
    }

    static func median7S16(_ v: [Int]) -> Int {
        let M = 0x7fff
        var w8 = s16(v[0])
        let w9r = s16(v[1])
        var w12 = s16(v[2])
        let w10u = w9r & 0xffff
        let w13 = w9r > w8 ? w9r : w8
        let w14 = w9r < w8 ? w9r : w8
        var w9: Int, w10: Int, w11 = 0
        if w10u == M { w9 = w10u; w10 = w8 } else { w9 = w13; w10 = w14 }
        if w12 == M { w8 = w9; w9 = M }
        else if w12 >= s16(w9) { w8 = w9; w9 = w12 }
        else {
            let w10s = s16(w10)
            w8 = w12 > w10s ? w12 : w10s
            w10 = w12 < w10s ? w12 : w10s
        }
        w12 = s16(v[3])
        if w12 == M { w11 = w8; w8 = w9; w9 = M }
        else if w12 >= s16(w9) { w11 = w8; w8 = w9; w9 = w12 }
        else if w12 >= s16(w8) { w11 = w8; w8 = w12 }
        else {
            let w10s = s16(w10)
            w11 = w12 > w10s ? w12 : w10s
            w10 = w12 < w10s ? w12 : w10s
        }
        w12 = s16(v[4])
        if w12 >= s16(w9) { w10 = w8; w8 = w9 }
        else if w12 >= s16(w8) { w10 = w8; w8 = w12 }
        else {
            let w9s = s16(w10), w10s = s16(w11)
            let t = w12 > w9s ? w12 : w9s
            w11 = w12 < w10s ? t : w11
            w10 = w12 > w10s ? w12 : w10s
        }
        w9 = s16(v[5])
        if w9 < s16(w8) {
            let a8 = s16(w11), a11 = s16(w10)
            let t = w9 > a8 ? w9 : a8
            w10 = w9 < a11 ? t : w10
            w8 = w9 > a11 ? w9 : a11
        }
        w9 = s16(v[6])
        let w10s = s16(w10)
        let t = w9 > w10s ? w9 : w10s
        return w9 < s16(w8) ? t : w8
    }

    final class State {
        let buf1 = RingBuf(), buf2 = RingBuf()
        var count = 0
        let maxOut = 4, maxGap = 5000
        var prevIbi = 1000, nOut = 0, nValid = 0
        let interpTs = false, hrMode = true
        var prevHr: Float = 60.0
        let maxHr: Float = 150.0
        var badRun = 0, savedBadRun = 0
        var dlIbi = [Int](repeating: 0, count: 7), dlRawAmp = [Int](repeating: 0, count: 7)
        var dlAmp = [Int](repeating: 0, count: 7), dlVal = [Int](repeating: 0, count: 7)
        var dlTs = [Int64](repeating: 0, count: 7), dlExtra = [Int](repeating: 0, count: 7)
        var mePrevHr: Float = 0.0
        let meMaxHr: Float = 150.0
        let meFlag = true
        var mePrevIbi = 1000
        var nowTs: Int64 = 0
        var msFlag = true
        var out: [(Int64, Int, Int, Int)] = []

        func var7Buf1() -> Float {
            let hs = (-6...0).map { hr(buf1.rfh($0).ibi) }
            var s = hs[0]
            for h in hs.dropFirst() { s = s + h }
            let mean = s / Float(7.0)
            var acc: Float? = nil
            for h in hs { let d = h - mean; let d2 = d * d; acc = acc == nil ? d2 : acc! + d2 }
            return acc! / Float(6.0)
        }

        func var7Dl() -> Float {
            let A = dlIbi
            let hs: [Float] = A.map { $0 != 0 ? hr($0) : Float(0.0) }
            var s: Float = A[0] == 0 ? 0.0 : hs[0]
            for i in 1..<7 where A[i] != 0 { s = s + hs[i] }
            let mean = s / Float(7.0)
            var acc: Float? = nil
            for h in hs { let d = h - mean; let d2 = d * d; acc = acc == nil ? d2 : acc! + d2 }
            return acc! / Float(6.0)
        }

        func medianError() {
            if dlVal[3] != 1 { return }
            let w22 = dlIbi[3]
            let s8 = hr(w22)
            if meprevZero { meprevZeroFix(s8) }
            let med = s16(median7S16(dlIbi))
            let tot = dlIbi.reduce(0) { $0 + s16($1) }
            let mean = Float(tot) / Float(7.0)
            let s1 = k60000 / Float(med)
            let s0 = k60000 / Float(fcvtzs(mean))
            var bad = false
            if meFlag {
                if s8 <= 30.0 || s8 > meMaxHr || abs(s8 - s1) > 18.0 { bad = true }
                else if s8 > (meprev + Float(23.0)) + Float(4.0) { bad = true }
                else if s8 < meprev * Float(0.625) { bad = true }
                else if s8 < (meprev + Float(-16.0)) + Float(-4.0) { bad = true }
                else {
                    let d = s0 - s8
                    if d * d > var7Dl() * Float(5.0) { bad = true }
                }
            } else {
                if s8 <= 30.0 || s8 > meMaxHr || abs(s8 - s1) > 18.0 { bad = true }
                else if abs(w22 - mePrevIbi) * 100 > mePrevIbi * 25 { bad = true }
                else {
                    let d = s0 - s8
                    if d * d > var7Dl() * Float(5.0) { bad = true }
                    else if !(dlExtra[3] < 7) { bad = true }
                }
            }
            if bad { dlVal[3] = 2 }
            meprev = s8
            mePrevIbi = w22
        }
        private var meprevZero: Bool { meprev == Float(0.0) }
        private func meprevZeroFix(_ s8: Float) { meprev = s8 }
        private var meprev: Float {
            get { mePrevHr }
            set { mePrevHr = newValue }
        }

        func flush(_ start: Int, _ n: Int, _ dec: Bool) {
            var w20 = start, w23 = n
            while true {
                let e = buf1.rfh(w20)
                let o = e.copy()
                o.ibiOut = e.ibi; o.ampOut = 0; o.validity = 2; o.ts2 = e.ts
                buf2.push(o)
                e.ibi = 0; e.amp = 0; e.validity = 0
                _ = buf1.pop()
                if dec { badRun = (badRun - 1) & 0xff }
                w23 = (w23 - 1) & 0xffffffff
                w20 += 1
                if (w23 & 0xff) == 0 { break }
            }
        }

        func afterGap() {
            if badRun > 60 { flush(-63, badRun, true) }
            emitLast()
        }

        func emitLast() {
            let e = buf1.rfh(-3)
            if e.ok == 1 {
                let o = e.copy()
                o.validity = 1; o.ibiOut = e.ibi; o.ampOut = e.amp; o.ts2 = e.ts
                buf2.push(o)
            }
        }

        func run(_ ibi0: Int, _ amp0: Int, _ ts: Int64) {
            let ibi = ibi0 & 0xffff, amp = amp0 & 0xffff
            nowTs = ts
            if ibi != 0 { ingest(ibi, amp, 0) }
            output()
        }

        func ingest(_ ibi: Int, _ amp: Int, _ extra: Int) {
            count += 1
            let new = Entry(ts: nowTs, ibi: ibi, amp: amp, ok: 1, extra: extra)
            buf1.push(new)
            if count == 1 {
                let o = new.copy()
                o.ibiOut = ibi; o.ampOut = amp; o.validity = 0; o.ts2 = nowTs
                if UInt64(bitPattern: new.ts &- 1) <= 99_999_999_998 { msFlag = false }
                buf2.push(o)
                return
            }
            if count < 5 { return }
            let e = buf1.rfh(-3)
            let s8: Float = e.ibi != 0 ? hr(e.ibi) : k60000
            let n = buf1.count
            var update: Bool
            if n <= 6 {
                var bad = false
                if hrMode {
                    if s8 <= 30.0 || s8 > maxHr { bad = true }
                    else if s8 < prevHr * Float(0.625) || s8 > prevHr + Float(18.0) { bad = true }
                    else if s8 < prevHr + Float(-25.0) { bad = true }
                } else {
                    if s8 <= 30.0 || s8 > maxHr { bad = true }
                    else if abs(e.ibi - prevIbi) * 100 > prevIbi * 20 { bad = true }
                    else if !(e.extra < 7) { bad = true }
                }
                if bad { e.ok = 0; badRun = (badRun + 1) & 0xff }
                else { e.ok = 1; savedBadRun = badRun; badRun = 0 }
                update = true
            } else {
                let m = median7U16((-6...0).map { buf1.rfh($0).ibi })
                let mm = (m & 0xfffe) != 0 ? m : 1
                let s9 = k60000 / Float(mm)
                let tot = (-6...0).reduce(0) { $0 + buf1.rfh($1).ibi }
                let mean = Float(tot) / Float(7.0)
                var s0: Float = mean != 0.0 ? k60000 / mean : 0.0
                s0 = s0 - s8
                var bad = false
                if s8 <= 30.0 || s8 > maxHr || abs(s8 - s9) > 14.0 { bad = true }
                else if hrMode {
                    if s8 < prevHr * Float(0.625) || s8 > prevHr + Float(23.0) { bad = true }
                    else if s8 < prevHr + Float(-16.0) { bad = true }
                    else if s0 * s0 > var7Buf1() * Float(1.5) { bad = true }
                } else {
                    if abs(e.ibi - prevIbi) * 100 > prevIbi * 20 { bad = true }
                    else if s0 * s0 > var7Buf1() * Float(1.5) { bad = true }
                    else if !(e.extra < 7) { bad = true }
                }
                if bad {
                    e.ok = 0; badRun = (badRun + 1) & 0xff
                    update = badRun >= 4
                } else {
                    e.ok = 1; savedBadRun = badRun; badRun = 0
                    update = true
                }
            }
            if update { prevHr = s8; prevIbi = e.ibi }
            let a = buf1.rfh(-3), b = buf1.rfh(-4)
            if a.ok == 1 && b.ok == 0 { gap(); return }
            if badRun > 60 { flush(-63, badRun, true) }
            emitLast()
        }

        func gap() {
            var w24 = 0
            var g = 0
            while true {
                g = w24 - 4
                if buf1.rfh(g).ok != 0 { break }
                let w8 = w24 - 5
                w24 -= 1
                if !(-63 <= w8 && w8 <= -1) {
                    if badRun > 60 { flush(-63, badRun, true) }
                    else if savedBadRun != 0 { flush(-3 - savedBadRun, savedBadRun, false) }
                    emitLast()
                    return
                }
            }
            var w21 = w24 - 3
            let ge = buf1.rfh(g)
            let gi = ge.ibi, gts = ge.ts
            let le = buf1.rfh(-3)
            let li = le.ibi, lts = le.ts
            var sumbad = 0
            if g <= -5 { for k in w21..<(-3) { sumbad += buf1.rfh(k).ibi } }
            let w11 = li + gi
            var nfill = fcvtas(Float(sumbad) / Float(w11 >> 1))
            if nfill <= 1 { nfill = 1 }
            let nbad = -w24
            let same = nfill == nbad
            if sumbad > maxGap || (1 - w24) > maxOut {
                var lastTs: Int64 = 0
                let e = buf1.rfh(w21)
                if e.ok == 0 {
                    while true {
                        let e = buf1.rfh(w21)
                        let o = e.copy()
                        o.ibiOut = e.ibi; o.ampOut = 0; o.validity = 2; o.ts2 = e.ts
                        buf2.push(o)
                        w21 += 1
                        lastTs = e.ts
                        if buf1.rfh(w21).ok != 0 { break }
                    }
                }
                if maxGap != 0 { buf2.push(Entry(ts2: lastTs, validity: 2)) }
                afterGap()
                return
            }
            let step = fcvtas(Float(li - gi) / Float(nfill + 1))
            let end = nfill + w24 - 4
            var cur = step + gi
            var cum = 0
            var lastTs: Int64 = 0
            var w20 = g
            while true {
                let w22 = w20 + 1
                let e = buf1.rfh(w22)
                var o = e.copy()
                if nbad >= nfill || w22 < -3 { lastTs = e.ts } else { o = Entry(ts: lastTs) }
                o.ibiOut = cur & 0xffff
                o.ampOut = 0
                o.validity = 3
                buf2.push(o)
                w20 += 1
                cum += cur
                cur += step
                if w20 >= end { break }
            }
            if interpTs && ((w11 & 0xffffffff) >= 2) {
                restoreOrValidate(same, nfill, g, end, w21)
                return
            }
            let diff = sumbad - cum
            var q = Float(diff) / Float(nfill)
            q = diff < 0 ? q.rounded(.up) : q.rounded(.down)
            let w11q = fcvtzs(q)
            let scale: Float = !msFlag ? 1000.0 : 1.0
            var r = diff - nfill * w11q
            var w23 = 1 - nfill
            var keep = same
            var cts: Int64 = 0
            var w24b = g
            while true {
                let e2 = buf2.rfh(w23)
                var v: Int
                if r == 0 { v = e2.ibiOut + w11q }
                else if r < 0 { v = (w11q - 1) + e2.ibiOut; r += 1 }
                else { v = (w11q + 1) + e2.ibiOut; r -= 1 }
                v &= 0xffffffff
                e2.ibiOut = v & 0xffff
                cts = cts &+ Int64(v & 0xffff)
                let t = Int64(fcvtas(Float(cts) / scale)) &+ gts
                e2.ts2 = UInt64(bitPattern: t) > UInt64(bitPattern: lts) ? lts : t
                if keep {
                    let d = abs(buf1.rfh(end).ibi - e2.ibiOut)
                    keep = d < 126
                } else { keep = false }
                w24b += 1
                w23 += 1
                if w24b >= end { break }
            }
            if keep { restoreOrValidate(true, nfill, g, end, w21); return }
            afterGap()
        }

        func restoreOrValidate(_ keep: Bool, _ nfill: Int, _ g: Int, _ end: Int, _ w21: Int) {
            if !keep { afterGap(); return }
            var w23 = g, w24 = 1 - nfill
            while true {
                w23 += 1
                let v = buf1.rfh(w23).ibi
                buf2.rfh(w24).ibiOut = v
                w24 += 1
                if !(w23 < end) { break }
            }
            let li = buf1.rfh(-3).ibi
            let gi = buf1.rfh(g).ibi
            let s9 = (k60000 / Float(min(li, gi))) + Float(34.5)
            var mn = 0xffff, mx = 0
            if w21 <= -4 {
                for i in w21..<(-3) {
                    let v = buf1.rfh(i).ibi
                    if mn >= v { mn = v }
                    if mx < v { mx = v }
                }
            }
            let s8 = k60000 / Float(mn)
            var good = false
            if s8 < s9 {
                var s0 = (k60000 / Float(max(li, gi))) + Float(-24.0)
                let lim = Float(bitPattern: 0x41f0cccd)
                if s0 < lim { s0 = lim }
                let s1 = k60000 / Float(mx)
                if s1 > s0 { good = true }
            }
            var w22 = 1 - nfill
            var w23b = g
            while true {
                let e2 = buf2.rfh(w22)
                e2.validity = good ? 1 : 2
                if good { e2.ampOut = e2.amp; e2.ts2 = e2.ts }
                w23b += 1
                w22 += 1
                if w23b >= end { break }
            }
            afterGap()
        }

        func output() {
            if buf2.isEmpty { return }
            var outs = 0
            while true {
                if outs >= maxOut { return }
                guard let e = buf2.pop() else { preconditionFailure("native assert: pop failed") }
                if e.ibiOut != 0 && count >= 0 {
                    dlIbi.removeFirst(); dlIbi.append(e.ibiOut)
                    dlRawAmp.removeFirst(); dlRawAmp.append(e.amp)
                    dlAmp.removeFirst(); dlAmp.append(e.ampOut)
                    dlVal.removeFirst(); dlVal.append(e.validity)
                    dlTs.removeFirst(); dlTs.append(e.ts2)
                    dlExtra.removeFirst(); dlExtra.append(e.extra)
                    if dlIbi[0] != 0 { medianError() }
                    if dlIbi[3] != 0 {
                        nOut = (nOut + 1) & 0xffff
                        if dlExtra[3] <= 1 && dlVal[3] == 1 { nValid = (nValid + 1) & 0xffff }
                        out.append((dlTs[3], dlIbi[3], dlAmp[3], dlVal[3] & 0xff))
                        outs += 1
                    }
                }
                if buf2.isEmpty { return }
            }
        }
    }
}
