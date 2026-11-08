import XCTest
@testable import OuraApp

/// Swift twins of tools/nssa_inputs.py, tools/ibi_correction.py and the 5-minute
/// aggregation must match the Python reference exactly. The fixture is synthetic
/// (tools/make_nssa_fixture.py, fixed seed) and checked on the Python side too.
final class NssaParityTests: XCTestCase {
    private func fixture() throws -> [String: Any] {
        let url = try XCTUnwrap(Bundle(for: NssaParityTests.self).url(forResource: "NssaParityFixture", withExtension: "json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
    private func ints(_ v: Any?) -> [Int] { (v as? [NSNumber])?.map(\.intValue) ?? [] }
    private func bytes(_ v: Any?) -> [UInt8] { ints(v).map(UInt8.init) }

    func testAcmPeriodDecodeIsBitExact() throws {
        let cases = try XCTUnwrap(fixture()["acm"] as? [[String: Any]])
        XCTAssertFalse(cases.isEmpty)
        for c in cases {
            let got = try XCTUnwrap(NssaInputs.decodeAcmPeriod(bytes(c["body"])))
            let want = (c["values"] as! [NSNumber]).map { Float($0.doubleValue) }
            XCTAssertEqual(got.map(\.bitPattern), want.map(\.bitPattern))
        }
        XCTAssertNil(NssaInputs.decodeAcmPeriod([UInt8](repeating: 0, count: 11)))
    }

    func testSleepTempExpansion() throws {
        for c in try XCTUnwrap(fixture()["temp"] as? [[String: Any]]) {
            let got = NssaInputs.expandTempEvent(bytes(c["body"]), (c["event_ms"] as! NSNumber).int64Value)
            let want = c["samples"] as! [[NSNumber]]
            XCTAssertEqual(got.map(\.0), want.map { $0[0].int64Value })
            XCTAssertEqual(got.map { $0.1.bitPattern }, want.map { Float($0[1].doubleValue).bitPattern })
        }
    }

    func testIbiExpansionClampAndWindow() throws {
        let f = try fixture()
        var last: Int64 = 0
        var all: [(Int64, Int, Int)] = []
        for c in try XCTUnwrap(f["ibi_events"] as? [[String: Any]]) {
            let (beats, l) = NssaInputs.expandIbiEvent(ints(c["ibis"]), ints(c["amps"]), (c["event_ms"] as! NSNumber).int64Value, last)
            last = l
            let want = c["beats"] as! [[NSNumber]]
            XCTAssertEqual(beats.map(\.0), want.map { $0[0].int64Value })
            XCTAssertEqual(beats.map(\.1), want.map { $0[1].intValue })
            XCTAssertEqual(beats.map(\.2), want.map { $0[2].intValue })
            all += beats
        }
        for w in try XCTUnwrap(f["ibi_windows"] as? [[String: Any]]) {
            let kept = NssaInputs.inclusiveWindow(all, (w["start"] as! NSNumber).int64Value, (w["end"] as! NSNumber).int64Value) { $0.0 }
            XCTAssertEqual(kept.count, (w["kept"] as! NSNumber).intValue)
            XCTAssertEqual(kept.first?.0, (w["first"] as? NSNumber)?.int64Value)
        }
    }

    func testIbiCorrectionMatchesPythonPort() throws {
        let streams = try XCTUnwrap(fixture()["ibi_correction"] as? [[String: Any]])
        XCTAssertEqual(streams.count, 6)
        for s in streams {
            let raw = (s["raw"] as! [[NSNumber]]).map { ($0[0].int64Value, $0[1].intValue, $0[2].intValue) }
            let want = (s["corrected"] as! [[NSNumber]]).map { [$0[0].int64Value, $0[1].int64Value, $0[2].int64Value, $0[3].int64Value] }
            let got = IbiCorrection.correct(raw).map { [$0.0, Int64($0.1), Int64($0.2), Int64($0.3)] }
            XCTAssertEqual(got.count, want.count)
            XCTAssertEqual(got, want)
        }
    }

    func testFiveMinuteAggregation() throws {
        for c in try XCTUnwrap(fixture()["aggregate_5min"] as? [[String: Any]]) {
            XCTAssertEqual(Sleep.aggregate5Min(ints(c["stages"])), ints(c["five_min"]))
        }
        XCTAssertEqual(Sleep.aggregate5Min([2, 2, 2, 2, 2, 2, 4, 4, 4, 4]), [4])
        XCTAssertEqual(Sleep.aggregate5Min([2, 2, 2, 1, 1, 1, 4, 4, 3, 3]), [1])
        XCTAssertTrue(Sleep.isOuraStaging("nssa"))
        XCTAssertFalse(Sleep.isOuraStaging("signals"))
    }

    func testGridMappingMirrorsRunner() {
        // start 23:34:30.5 style offset: epochs on floor30 map to the cell holding their midpoint.
        let start: Int64 = 1_000_500, end: Int64 = start + 5 * 30_000 + 1_000
        let ts: [Int64] = [1_000_000, 1_030_000, 1_060_000, 1_090_000]
        let grid = NssaInputs.gridStages(periods: [(ts, [1, 2, 3, 1])], failed: [], startMs: start, endMs: end)
        XCTAssertEqual(grid, [1, 2, 3, 1, 4, 4])
        let failed = NssaInputs.gridStages(periods: [(ts, [1, 2, 3, 1])], failed: [(1_120_000, 1_200_000)],
                                           startMs: start, endMs: end)
        XCTAssertEqual(failed, [1, 2, 3, 1, 0, 0])
        // an epoch ending before cell 0's midpoint is not painted; 10 s before still is
        XCTAssertEqual(NssaInputs.gridStages(periods: [([980_000], [3])], failed: [], startMs: 1_000_000, endMs: 1_030_000), [4])
        XCTAssertEqual(NssaInputs.gridStages(periods: [([990_000], [3])], failed: [], startMs: 1_000_000, endMs: 1_030_000), [3])
    }

    func testDemographicsMatchStager() {
        let d = NssaInputs.demographics(age: 30.9, heightM: 1.78, weightKg: 75, sex: "M")
        XCTAssertEqual(d, [30, Float(75.0 / (1.78 * 1.78)), 1])
        XCTAssertEqual(NssaInputs.demographics(age: 40, heightM: 1.6, weightKg: 60, sex: "F")[2], -1)
        XCTAssertEqual(NssaInputs.demographics(age: 40, heightM: 1.6, weightKg: 60, sex: "O")[2], 0)
    }
}

#if TORCH
/// The bundled .ptl must find its oura_ops custom ops (TorchBridge.mm) and run.
final class NssaInferenceSmokeTests: XCTestCase {
    func testNssaRunsOnSyntheticNight() throws {
        guard let model = Bundle.main.path(forResource: "sleepstaging_2_6_0", ofType: "ptl"),
              let blob = Bundle.main.path(forResource: "sleepstaging_2_6_0_classifier", ofType: "bin") else {
            throw XCTSkip("NSSA model not bundled (private Oura weights)")
        }
        oura_nssa_set_classifier_path(blob)
        let start: Int64 = 1_792_103_670_000, end = start + 7 * 3600 * 1000
        var acmTs: [Int64] = [], acmVal: [Float] = [], tempTs: [Int64] = [], tempVal: [Float] = []
        var t = start
        while t < end {
            acmTs.append(t); acmVal += [0.5, 1.5, 0.5, 15, 25, 18]
            tempTs.append(t); tempVal.append(35.2)
            t += 30_000
        }
        var ibiTs: [Int64] = [], ibiVal: [Float] = []
        t = start
        var k = 0
        while t < end {
            let ibi = 1000 + (k % 7) * 15
            ibiTs.append(t); ibiVal += [Float(ibi), 800, k == 0 ? 0 : 1]
            t += Int64(ibi); k += 1
        }
        var demo: [Float] = [30, 23.7, 1]
        var out = [Int32](repeating: 0, count: 2000), ts = [Int64](repeating: 0, count: 2000)
        let n = oura_nssa(model, &acmTs, &acmVal, Int32(acmTs.count), &ibiTs, &ibiVal, Int32(ibiTs.count),
                          &tempTs, &tempVal, Int32(tempTs.count), &demo, start, end, &out, &ts, 2000)
        XCTAssertEqual(n, 840)
        XCTAssertTrue(out.prefix(Int(max(n, 0))).allSatisfy { (1...4).contains($0) })
        XCTAssertEqual(ts.first, start)
        // tools/nssa_stager.py on the same synthetic input: 840 WAKE epochs
        XCTAssertEqual(out.prefix(Int(max(n, 0))).map(Int.init), Array(repeating: 4, count: 840))
    }
}
#endif
