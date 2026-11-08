#ifndef OURA_NSSA_OPS_H
#define OURA_NSSA_OPS_H
// Native reimplementation of the `oura_ops::*` TorchScript custom ops that Oura's
// NSSA sleep stager (sleepstaging_2_6_0) calls, plus its tree-ensemble classifier.
// One C source serves both clients: TorchBridge.mm registers these as torch ops on
// iOS, and tools/nssa_ops.py loads the same file through ctypes on the desktop
// (with a pure-Python fallback that tools/test_nssa_ops.py keeps bit-identical).
// Semantics are transcribed from Oura's libalgos.so (arm64); addresses are in
// docs/algorithms/sleep-staging.md. Compile with -ffp-contract=off: the biquad
// relies on explicit fmaf() and no other contraction.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// nssaops::create_windows @0x5602fc. `ts` are N epoch-ms stamps (sorted). Returns the
// row count n; when `out_ts` / `out_windows` are non-NULL they receive n window end
// stamps and n×4 index rows [5-min lookback, 1-min lookback, start, end).
int64_t nssa_create_windows(const int64_t *ts, int64_t n_ts, double window_s,
                            int64_t *out_ts, int64_t *out_windows);

// nssaops::find_indices @0x561248: running "low" index for interp1d. Returns 0, or
// -1 where the native code would read past `x` (undefined behaviour there).
int nssa_find_indices(const int64_t *x, int64_t n_x, const int64_t *xnew, int64_t n_new,
                      int64_t *out);

// nssaops::biquad_cascade @0x55e860 + Biquad<float>::step @0x564234: transposed
// direct form II, float32 fused multiply-adds, zero initial state.
void nssa_biquad_cascade(const float *sos, int64_t n_sections, const float *x, int64_t n,
                         float *out);

// nssaops::find_peaks @0x55ee1c (scipy _local_maxima_1d). `out` needs n/2 slots;
// returns the number of peak midpoints written.
int64_t nssa_find_peaks(const float *x, int64_t n, int64_t *out);

// oura_sleep_classifier: Oura's treelite-compiled gradient-boosted forest (4 classes,
// 182 features) lifted from libalgos.so into a flat node table (see
// tools/extract_nssa_classifier.py). `blob` is the file's bytes. Writes n×4 softmax
// probabilities computed exactly like treelite's predict_multiclass. Returns 0 or -1.
int nssa_classifier_predict(const uint8_t *blob, int64_t blob_len, const float *features,
                            int64_t n_rows, int64_t n_cols, float *out_probs);

#ifdef __cplusplus
}
#endif
#endif
