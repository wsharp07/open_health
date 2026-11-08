// See NssaOps.h. Transcribed from libalgos.so (Oura Android 7.18.0); every
// comparison and rounding step mirrors the arm64 code so results are bit-identical.
#include "NssaOps.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static int64_t c_rem(int64_t a, int64_t b) { return a - (a / b) * b; }  // C99 truncation

int64_t nssa_create_windows(const int64_t *ts, int64_t n_ts, double window_s,
                            int64_t *out_ts, int64_t *out_windows) {
    if (n_ts <= 0) return 0;
    const int64_t w = (int64_t)(window_s * 1000.0);
    if (w <= 0) return 0;
    const int64_t first = ts[0], last = ts[n_ts - 1];
    const int64_t start = first - c_rem(c_rem(first, w) + w, w);
    const int64_t end = last + w - c_rem(last, w);
    const int64_t n = (int64_t)(int32_t)((end - start) / w);
    if (out_ts == NULL || out_windows == NULL) return n;
    for (int64_t i = 0; i < n; i++) {
        out_ts[i] = 0;
        for (int c = 0; c < 4; c++) out_windows[i * 4 + c] = -1;
    }
    if (end <= start) return n;
    int32_t p_start = 0, p_end = 0, p_1 = 0, p_5 = 0;
    int64_t cur = start, row = 0;
    for (;;) {
        while (p_start < n_ts && ts[p_start] < cur) p_start++;
        while (p_end < n_ts && ts[p_end] < cur + w - 1) p_end++;
        while (p_1 < n_ts && ts[p_1] < cur + w - 60000) p_1++;
        while (p_5 < n_ts && ts[p_5] < cur + w - 300000) p_5++;
        const int64_t nxt = cur + w;
        if (row < n) {
            int64_t *r = out_windows + row * 4;
            r[0] = p_5; r[1] = p_1; r[2] = p_start; r[3] = p_end;
            if (nxt - 300000 < start) r[0] = -1;
            if (nxt - 60000 < start) r[1] = r[2] = r[3] = -1;
            out_ts[row] = nxt;
        }
        row++;
        cur = nxt;
        if (!(cur < end)) break;
    }
    return n;
}

int nssa_find_indices(const int64_t *x, int64_t n_x, const int64_t *xnew, int64_t n_new,
                      int64_t *out) {
    int32_t j = 0;
    for (int64_t k = 0; k < n_new; k++) {
        const int64_t v = xnew[k];
        int64_t m = (int64_t)j + 1;
        for (;;) {
            if (m >= n_x) return -1;
            if (v > x[m]) { m++; continue; }
            break;
        }
        j = (int32_t)(m - 1);
        out[k] = j;
    }
    return 0;
}

void nssa_biquad_cascade(const float *sos, int64_t n_sections, const float *x, int64_t n,
                         float *out) {
    // per section: b0, b1, b2, a1, a2 (normalised by a0 in float32), z1, z2
    float coef[64][7];
    if (n_sections > 64) n_sections = 64;
    for (int64_t s = 0; s < n_sections; s++) {
        const float *r = sos + s * 6;
        const float a0 = r[3];
        coef[s][0] = r[0] / a0; coef[s][1] = r[1] / a0; coef[s][2] = r[2] / a0;
        coef[s][3] = r[4] / a0; coef[s][4] = r[5] / a0;
        coef[s][5] = 0.0f; coef[s][6] = 0.0f;
    }
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        for (int64_t s = 0; s < n_sections; s++) {
            float *c = coef[s];
            const float y = fmaf(v, c[0], c[5]);
            const float t1 = -(c[3] * y);
            const float z1 = c[6] + fmaf(v, c[1], t1);
            const float t2 = -(c[4] * y);
            c[6] = fmaf(v, c[2], t2);
            c[5] = z1;
            v = y;
        }
        out[i] = v;
    }
}

// ---- tree-ensemble classifier (file layout in tools/nssa_classifier.py) ----------
typedef struct {
    uint8_t kind, cc, missyes, pad;
    int32_t feat;
    double thr;
    int32_t yes, no;
} NssaNode;

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static double eval_tree(const NssaNode *nodes, int64_t n_nodes, uint32_t root,
                        const double *x, const uint8_t *miss, int *err) {
    int64_t i = root;
    for (int guard = 0; guard < 4096; guard++) {
        if (i < 0 || i >= n_nodes) { *err = 1; return 0.0; }
        const NssaNode *nd = &nodes[i];
        if (nd->kind == 0) return nd->thr;
        const int f = nd->feat;
        int take;
        if (nd->kind == 1) {
            take = miss[f];
        } else if (nd->kind == 3 && miss[f]) {
            take = nd->missyes;
        } else {
            const int le = x[f] <= nd->thr;  // false when unordered (AArch64 'ls')
            take = nd->cc == 0 ? le : !le;   // 'hi' = x > thr or unordered
        }
        i = take ? nd->yes : nd->no;
    }
    *err = 1;
    return 0.0;
}

int nssa_classifier_predict(const uint8_t *blob, int64_t blob_len, const float *features,
                            int64_t n_rows, int64_t n_cols, float *out_probs) {
    if (blob_len < 32 || memcmp(blob, "NSSAFOR1", 8) != 0) return -1;
    const uint32_t num_class = rd32(blob + 8), num_feature = rd32(blob + 12);
    const uint32_t n_nodes = rd32(blob + 16), n_trees = rd32(blob + 20);
    const uint32_t n_units = rd32(blob + 24), n_ops = rd32(blob + 28);
    if (num_class != 4 || (int64_t)num_feature != n_cols || num_feature > 256) return -1;
    const int64_t off_roots = 32, off_nodes = off_roots + 4 * (int64_t)n_trees;
    const int64_t off_progs = off_nodes + (int64_t)sizeof(NssaNode) * n_nodes;
    const int64_t off_ops = off_progs + 8 * (int64_t)n_units * num_class;
    if (sizeof(NssaNode) != 24 || off_ops + 4 * (int64_t)n_ops > blob_len) return -1;
    const NssaNode *nodes = (const NssaNode *)(const void *)(blob + off_nodes);
    double x[256];
    uint8_t miss[256];
    double stack[64];
    for (int64_t r = 0; r < n_rows; r++) {
        const float *row = features + r * n_cols;
        for (uint32_t f = 0; f < num_feature; f++) {
            const float v = row[f];
            if (v != v) {  // NaN -> treelite missing Entry (bits 0x00000000FFFFFFFF)
                const uint64_t bits = 0xFFFFFFFFull;
                memcpy(&x[f], &bits, sizeof bits);
                miss[f] = 1;
            } else {
                x[f] = (double)v;
                miss[f] = 0;
            }
        }
        double res[4] = {0.0, 0.0, 0.0, 0.0};
        for (uint32_t u = 0; u < n_units; u++) {
            double next[4];
            for (uint32_t k = 0; k < num_class; k++) {
                const uint8_t *pp = blob + off_progs + 8 * ((int64_t)u * num_class + k);
                const uint32_t first = rd32(pp), count = rd32(pp + 4);
                if ((int64_t)first + count > n_ops) return -1;
                int sp = 0, err = 0;
                for (uint32_t o = 0; o < count; o++) {
                    const uint32_t op = rd32(blob + off_ops + 4 * ((int64_t)first + o));
                    const uint32_t code = op >> 30, arg = op & 0x3FFFFFFFu;
                    if (code == 2) {
                        if (sp < 2) return -1;
                        stack[sp - 2] = stack[sp - 2] + stack[sp - 1];
                        sp--;
                    } else {
                        if (sp >= 64) return -1;
                        if (code == 1) {
                            if (arg >= 4) return -1;
                            stack[sp++] = res[arg];
                        } else {
                            if (arg >= n_trees) return -1;
                            stack[sp++] = eval_tree(nodes, n_nodes, rd32(blob + off_roots + 4 * (int64_t)arg),
                                                    x, miss, &err);
                            if (err) return -1;
                        }
                    }
                }
                if (sp != 1) return -1;
                next[k] = stack[0];
            }
            memcpy(res, next, sizeof res);
        }
        for (int k = 0; k < 4; k++) res[k] = res[k] + 0.0;  // global bias
        double mx = res[1] > res[0] ? res[1] : res[0];
        mx = res[2] > mx ? res[2] : mx;
        mx = res[3] > mx ? res[3] : mx;
        const double e0 = exp(res[0] - mx), e1 = exp(res[1] - mx);
        const double e2 = exp(res[2] - mx), e3 = exp(res[3] - mx);
        const double s = ((e0 + e1) + e2) + e3;
        out_probs[r * 4 + 0] = (float)(e0 / s);
        out_probs[r * 4 + 1] = (float)(e1 / s);
        out_probs[r * 4 + 2] = (float)(e2 / s);
        out_probs[r * 4 + 3] = (float)(e3 / s);
    }
    return 0;
}

int64_t nssa_find_peaks(const float *x, int64_t n, int64_t *out) {
    int64_t m = 0;
    const int64_t i_max = (int64_t)(int32_t)n - 1;
    if (i_max < 2) return 0;
    int64_t i = 1;
    while (i < i_max) {
        if (x[i - 1] < x[i]) {
            int64_t ahead = i + 1;
            while (ahead < i_max && x[ahead] == x[i]) ahead++;
            if (x[ahead] < x[i]) {
                out[m++] = (i + ahead - 1) / 2;
                i = ahead;
            }
        }
        i++;
    }
    return m;
}
