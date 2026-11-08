"""Oura's NSSA sleep classifier (`oura_ops::oura_sleep_classifier`): a treelite-compiled
gradient-boosted forest (4 classes, 182 features, 2000 trees in 63 code units) in
libalgos.so, lifted by tools/extract_nssa_classifier.py into a flat node table
`sleepstaging_2_6_0_classifier.bin` stored next to the decrypted models (Oura's
weights: gitignored like the models themselves).

Bit-exact with the native op (verified under emulation, see
docs/algorithms/sleep-staging.md):
* each float32 feature becomes a treelite Entry: NaN -> missing (bits
  0x00000000FFFFFFFF), anything else -> (double)value;
* every unit adds its trees per class in the native fadd order, then the global bias
  (+0.0) and treelite's softmax (max by strict '>' over r0..r3, ((e0+e1)+e2)+e3);
* the 4 doubles are cast to float32.
apps/ios/OuraApp/NssaOps.c implements the same evaluator over the same file.

File layout (little endian), shared with NssaOps.c:
  magic "NSSAFOR1", u32 num_class, u32 num_feature, u32 n_nodes, u32 n_trees,
  u32 n_units, u32 n_ops
  n_trees x u32 root node index
  n_nodes x node: u8 kind (0 leaf, 1 missing?, 2 compare, 3 test), u8 cc (0 'ls'
      x<=t, 1 'hi' x>t or unordered), u8 missing_goes_yes, u8 0, i32 feature,
      f64 threshold (leaf: value), i32 yes, i32 no
  n_units*num_class x (u32 first_op, u32 n_ops)
  n_ops x u32 op: (0 << 30 | tree) push tree, (1 << 30 | k) push result[k], (2 << 30) add
"""
import math
import struct

import numpy as np

MAGIC = b"NSSAFOR1"
CLASSIFIER_NAME = "sleepstaging_2_6_0_classifier.bin"
MISSING_BITS = np.uint64(0x00000000FFFFFFFF)
NODE = struct.Struct("<BBBBidii")
HEADER = struct.Struct("<8s6I")
OP_TREE, OP_RES, OP_ADD = 0, 1, 2


def write_forest(trees_json, path):
    """Serialise the lifter's JSON (units / trees, see extract_nssa_classifier.py)."""
    roots, nodes = [], []
    for tree in trees_json["trees"]:
        base = len(nodes)
        roots.append(base)
        for nd in tree:
            if len(nd) == 1:
                nodes.append((0, 0, 0, 0, -1, float(nd[0]), -1, -1))
            elif nd[0] == "miss":
                nodes.append((1, 0, 0, 0, nd[1], 0.0, base + nd[2], base + nd[3]))
            elif nd[0] == "cmp":
                nodes.append((2, ("ls", "hi").index(nd[2]), 0, 0, nd[1], float(nd[3]), base + nd[4], base + nd[5]))
            else:
                nodes.append((3, ("ls", "hi").index(nd[2]), 1 if nd[4] else 0, 0, nd[1], float(nd[3]),
                              base + nd[5], base + nd[6]))
    num_class = trees_json["num_class"]
    progs, ops = [], []

    def emit(e):
        if e[0] == "+":
            emit(e[1])
            emit(e[2])
            ops.append(OP_ADD << 30)
        elif e[0] == "res":
            ops.append((OP_RES << 30) | e[1])
        else:
            ops.append((OP_TREE << 30) | e[1])

    for unit in trees_json["units"]:
        for k in range(num_class):
            first = len(ops)
            emit(unit[k])
            progs.append((first, len(ops) - first))
    with open(path, "wb") as f:
        f.write(HEADER.pack(MAGIC, num_class, trees_json["num_feature"], len(nodes), len(roots),
                            len(trees_json["units"]), len(ops)))
        f.write(struct.pack("<%dI" % len(roots), *roots))
        for nd in nodes:
            f.write(NODE.pack(*nd))
        for first, n in progs:
            f.write(struct.pack("<II", first, n))
        f.write(struct.pack("<%dI" % len(ops), *ops))


class Forest:
    def __init__(self, path):
        blob = open(path, "rb").read()
        magic, self.num_class, self.num_feature, n_nodes, n_trees, n_units, n_ops = HEADER.unpack_from(blob, 0)
        if magic != MAGIC:
            raise ValueError("not an NSSA classifier file: %s" % path)
        off = HEADER.size
        self.roots = np.frombuffer(blob, "<u4", n_trees, off).astype(np.int64)
        off += 4 * n_trees
        dt = np.dtype([("kind", "u1"), ("cc", "u1"), ("missyes", "u1"), ("pad", "u1"),
                       ("feat", "<i4"), ("thr", "<f8"), ("yes", "<i4"), ("no", "<i4")])
        nodes = np.frombuffer(blob, dt, n_nodes, off)
        off += dt.itemsize * n_nodes
        self.kind, self.cc, self.missyes = nodes["kind"].copy(), nodes["cc"].copy(), nodes["missyes"].astype(bool)
        self.feat, self.thr = nodes["feat"].copy(), nodes["thr"].copy()
        self.yes, self.no = nodes["yes"].copy(), nodes["no"].copy()
        progs = np.frombuffer(blob, "<u4", 2 * n_units * self.num_class, off).reshape(-1, 2)
        off += 8 * n_units * self.num_class
        ops = np.frombuffer(blob, "<u4", n_ops, off)
        self.programs = [[[int(o) for o in ops[a:a + n]] for a, n in progs[u * self.num_class:(u + 1) * self.num_class]]
                         for u in range(n_units)]
        self.blob = blob

    def _tree(self, ti, x, miss):
        n = x.shape[0]
        idx = np.full(n, self.roots[ti], np.int64)
        rows = np.arange(n)
        while True:
            k = self.kind[idx]
            live = k != 0
            if not live.any():
                return self.thr[idx]
            r, i = rows[live], idx[live]
            f = self.feat[i]
            xv, m = x[r, f], miss[r, f]
            le = xv <= self.thr[i]  # numpy: False when unordered, like `ls`
            cond = np.where(self.cc[i] == 0, le, ~le)
            kk = self.kind[i]
            take = np.where(kk == 1, m, np.where((kk == 3) & m, self.missyes[i], cond))
            idx[live] = np.where(take, self.yes[i], self.no[i])

    def margins(self, features):
        f32 = np.asarray(features, dtype=np.float32)
        miss = np.isnan(f32)
        bits = f32.astype(np.float64).view(np.uint64).copy()
        bits[miss] = MISSING_BITS
        x = bits.view(np.float64)
        n = x.shape[0]
        res = np.zeros((n, self.num_class), np.float64)
        with np.errstate(all="ignore"):
            for unit in self.programs:
                cache, new = {}, []
                for prog in unit:
                    stack = []
                    for op in prog:
                        code, arg = op >> 30, op & 0x3FFFFFFF
                        if code == OP_TREE:
                            if arg not in cache:
                                cache[arg] = self._tree(arg, x, miss)
                            stack.append(cache[arg])
                        elif code == OP_RES:
                            stack.append(res[:, arg])
                        else:
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(a + b)
                    new.append(stack[0])
                res = np.stack(new, 1)
            res = res + 0.0  # global bias
        return res

    @staticmethod
    def softmax(m):
        out = np.empty_like(m)
        for r in range(m.shape[0]):
            r0, r1, r2, r3 = (float(v) for v in m[r])
            mx = r1 if r1 > r0 else r0
            mx = r2 if r2 > mx else mx
            mx = r3 if r3 > mx else mx
            e0, e1, e2, e3 = math.exp(r0 - mx), math.exp(r1 - mx), math.exp(r2 - mx), math.exp(r3 - mx)
            s = ((e0 + e1) + e2) + e3
            out[r] = (e0 / s, e1 / s, e2 / s, e3 / s)
        return out

    def predict(self, features):
        """float32-castable [N,182] -> float32 [N,4] probabilities (the op's output)."""
        if len(features) == 0:
            return np.zeros((0, self.num_class), np.float32)
        return self.softmax(self.margins(features)).astype(np.float32)
