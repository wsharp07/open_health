"""Symbolic lifter for treelite-compiled predict_margin_multiclass_unitN (arm64).

Every value is a 64-bit bit pattern expression (hash-consed DAG).  Control flow
(a DAG) is executed region by region: at a conditional branch we execute both
successors up to the branch's immediate post-dominator and merge the states with
select nodes.  The result is, per result slot, an exact expression DAG.
"""
import os, re, struct, sys, pickle
from elftools.elf.elffile import ELFFile
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM

SO = os.environ["NSSA_LIBALGOS"]  # set by tools/extract_nssa_classifier.py
M64 = (1 << 64) - 1
M32 = (1 << 32) - 1

# ---------------------------------------------------------------- ELF access
_f = open(SO, 'rb')
ELF = ELFFile(_f)
SEGS = []
for seg in ELF.iter_segments():
    if seg['p_type'] == 'PT_LOAD':
        SEGS.append((seg['p_vaddr'], seg['p_filesz'], seg.data()))


def read_mem(addr, n):
    for va, sz, d in SEGS:
        if va <= addr and addr + n <= va + sz:
            return d[addr - va:addr - va + n]
    raise KeyError(hex(addr))


def unit_symbols():
    out = {}
    for s in ELF.get_section_by_name('.dynsym').iter_symbols():
        m = re.match(r'predict_margin_multiclass_unit(\d+)$', s.name)
        if m:
            out[int(m.group(1))] = (s['st_value'], s['st_size'])
    return out


# ---------------------------------------------------------------- expressions
NODES = []      # id -> tuple
INTERN = {}


def mk(t):
    i = INTERN.get(t)
    if i is None:
        i = len(NODES)
        NODES.append(t)
        INTERN[t] = i
    return i


def C(v):
    return mk(('c', v & M64))


def is_c(e):
    return NODES[e][0] == 'c'


def cv(e):
    return NODES[e][1]


def b2d(b):
    return struct.unpack('<d', struct.pack('<Q', b))[0]


def d2b(d):
    return struct.unpack('<Q', struct.pack('<d', d))[0]


def fcmp_nzcv(a, b):
    x, y = b2d(a), b2d(b)
    if x != x or y != y:
        return 0b0011
    if x == y:
        return 0b0110
    if x < y:
        return 0b1000
    return 0b0010


def sub_nzcv(a, b, bits):
    m = (1 << bits) - 1
    a &= m; b &= m
    r = (a - b) & m
    n = (r >> (bits - 1)) & 1
    z = int(r == 0)
    c = int(a >= b)
    sa, sb, sr = a >> (bits - 1), b >> (bits - 1), r >> (bits - 1)
    v = int(sa != sb and sr != sa)
    return (n << 3) | (z << 2) | (c << 1) | v


def add_nzcv(a, b, bits):
    m = (1 << bits) - 1
    a &= m; b &= m
    full = a + b
    r = full & m
    n = (r >> (bits - 1)) & 1
    z = int(r == 0)
    c = int(full > m)
    sa, sb, sr = a >> (bits - 1), b >> (bits - 1), r >> (bits - 1)
    v = int(sa == sb and sr != sa)
    return (n << 3) | (z << 2) | (c << 1) | v


def cond_holds(nz, cc):
    N, Z, Cf, V = (nz >> 3) & 1, (nz >> 2) & 1, (nz >> 1) & 1, nz & 1
    base = cc
    r = {
        'eq': Z == 1, 'ne': Z == 0, 'hs': Cf == 1, 'cs': Cf == 1, 'lo': Cf == 0, 'cc': Cf == 0,
        'mi': N == 1, 'pl': N == 0, 'vs': V == 1, 'vc': V == 0,
        'hi': Cf == 1 and Z == 0, 'ls': not (Cf == 1 and Z == 0),
        'ge': N == V, 'lt': N != V, 'gt': Z == 0 and N == V, 'le': not (Z == 0 and N == V),
        'al': True,
    }[base]
    return int(r)


def E(op, *args):
    """Build node with constant folding."""
    if all(is_c(a) for a in args if isinstance(a, int)) and op in FOLD:
        return C(FOLD[op](*[cv(a) if isinstance(a, int) else a for a in args]))
    return mk((op,) + args)


FOLD = {
    'fadd': lambda a, b: d2b(b2d(a) + b2d(b)),
    'and': lambda a, b: a & b,
    'nzf': lambda a, b: fcmp_nzcv(a, b),
    'nzs64': lambda a, b: sub_nzcv(a, b, 64),
    'nzs32': lambda a, b: sub_nzcv(a, b, 32),
    'nza32': lambda a, b: add_nzcv(a, b, 32),
    'nzl32': lambda a: ((((a & M32) >> 31) & 1) << 3) | (int((a & M32) == 0) << 2),
}


def COND(flags, cc):
    if is_c(flags):
        return C(cond_holds(cv(flags), cc))
    return mk(('cond', flags, cc))


def SEL(c, a, b):
    if a == b:
        return a
    if is_c(c):
        return a if cv(c) else b
    # sel(c, sel(c,x,y), z) -> sel(c,x,z)
    na, nb = NODES[a], NODES[b]
    if na[0] == 'sel' and na[1] == c:
        a = na[2]
    if nb[0] == 'sel' and nb[1] == c:
        b = nb[3]
    if a == b:
        return a
    return mk(('sel', c, a, b))


UNDEF = mk(('undef',))


class Ptr:
    __slots__ = ('region', 'off')

    def __init__(self, region, off):
        self.region, self.off = region, off

    def __eq__(self, o):
        return isinstance(o, Ptr) and o.region == self.region and o.off == self.off

    def __hash__(self):
        return hash((self.region, self.off))

    def __repr__(self):
        return 'Ptr(%s,%d)' % (self.region, self.off)


# ---------------------------------------------------------------- state
class State:
    __slots__ = ('x', 'v', 'nz', 'stack', 'res')

    def copy(self):
        s = State()
        s.x = dict(self.x); s.v = dict(self.v); s.nz = self.nz
        s.stack = dict(self.stack); s.res = dict(self.res)
        return s


def initial_state():
    s = State()
    s.x = {i: UNDEF for i in range(31)}
    s.x[0] = Ptr('data', 0)
    s.x[1] = Ptr('res', 0)
    s.x['sp'] = Ptr('stack', 0)
    s.v = {i: (UNDEF, UNDEF) for i in range(32)}
    s.nz = UNDEF
    s.stack = {}
    s.res = {k: mk(('res', k)) for k in range(4)}
    return s


def merge_val(c, a, b):
    if isinstance(a, Ptr) or isinstance(b, Ptr):
        if a == b:
            return a
        # dead pointer-vs-value differences: keep undef
        return UNDEF
    if a == UNDEF:
        return b
    if b == UNDEF:
        return a
    return SEL(c, a, b)


def merge(c, sa, sb):
    s = State()
    s.x = {k: merge_val(c, sa.x[k], sb.x[k]) for k in sa.x}
    s.v = {k: (merge_val(c, sa.v[k][0], sb.v[k][0]), merge_val(c, sa.v[k][1], sb.v[k][1])) for k in sa.v}
    s.nz = merge_val(c, sa.nz, sb.nz)
    s.stack = {}
    for k in set(sa.stack) | set(sb.stack):
        s.stack[k] = merge_val(c, sa.stack.get(k, UNDEF), sb.stack.get(k, UNDEF))
    s.res = {k: merge_val(c, sa.res[k], sb.res[k]) for k in sa.res}
    return s


# ---------------------------------------------------------------- decoding
RE_IMM = re.compile(r'#(-?0x[0-9a-f]+|-?\d+)')


def imm(s):
    return int(s.lstrip('#'), 0)


def reg_kind(r):
    if r in ('xzr', 'wzr'):
        return ('zr', None)
    if r == 'sp':
        return ('x', 'sp')
    k = r[0]
    n = int(r[1:].split('.')[0])
    return (k, n)


class Unit:
    def __init__(self, idx, addr, size):
        self.idx, self.addr, self.size = idx, addr, size
        self.ins = {}
        self.build_cfg()

    def decode(self, a):
        if a not in self.ins:
            i = next(self.md.disasm(read_mem(a, 4), a))
            self.ins[a] = (i.mnemonic, i.op_str)
        return self.ins[a]

    def build_cfg(self):
        self.md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        self.blocks = {}
        targets = {self.addr}
        # first pass: find leaders by exploring reachable code
        leaders = {self.addr}
        work = [self.addr]
        seen = set()
        while work:
            a = work.pop()
            while a not in seen:
                seen.add(a)
                m, o = self.decode(a)
                if m == 'ret':
                    break
                if m == 'b':
                    t = imm(o); leaders.add(t); work.append(t); break
                if m.startswith('b.') or m in ('tbnz', 'tbz'):
                    t = imm(o.split(', ')[-1]); leaders.add(t); work.append(t)
                    leaders.add(a + 4); work.append(a + 4); break
                a += 4
        lset = leaders
        for l in leaders:
            body = []
            a = l
            while True:
                body.append(a)
                m, o = self.decode(a)
                if m == 'b' or m.startswith('b.') or m in ('tbnz', 'tbz', 'ret') or a + 4 in lset:
                    break
                a += 4
            last = body[-1]
            m, o = self.ins[last]
            if m == 'ret':
                succ = []
            elif m == 'b':
                succ = [imm(o)]
            elif m.startswith('b.') or m in ('tbnz', 'tbz'):
                succ = [imm(o.split(', ')[-1]), last + 4]
            else:
                succ = [last + 4]
            self.blocks[l] = (body, succ)
        # post-dominators on DAG: process in reverse topological order
        order = []
        seen = set()
        sys.setrecursionlimit(1000000)
        stack = [(self.addr, 0)]
        while stack:
            b, i = stack.pop()
            if i == 0:
                if b in seen:
                    continue
                seen.add(b)
            succ = self.blocks[b][1]
            if i < len(succ):
                stack.append((b, i + 1))
                if succ[i] not in seen:
                    stack.append((succ[i], 0))
            else:
                order.append(b)        # post-order
        self.postorder = order
        rank = {b: i for i, b in enumerate(order)}   # exit has smallest rank
        self.rank = rank
        ipdom = {}
        EXIT = 'EXIT'
        for b in order:
            succ = self.blocks[b][1]
            if not succ:
                ipdom[b] = EXIT
                continue
            cand = None
            for s in succ:
                cand = s if cand is None else self._isect(cand, s, ipdom, rank)
            ipdom[b] = cand
        self.ipdom = ipdom

    def _isect(self, a, b, ipdom, rank):
        while a != b:
            if a == 'EXIT' or b == 'EXIT':
                return 'EXIT'
            if rank[a] < rank[b]:
                b = ipdom[b]
            else:
                a = ipdom[a]
        return a

    # ------------------------------------------------------------ semantics
    def getx(self, s, r):
        k, n = reg_kind(r)
        if k == 'zr':
            return C(0)
        v = s.x[n]
        if k == 'w':
            if isinstance(v, Ptr):
                raise RuntimeError('w of ptr')
            return E('and', v, C(M32))
        return v

    def setx(self, s, r, val):
        k, n = reg_kind(r)
        if k == 'zr':
            return
        if k == 'w' and not isinstance(val, Ptr):
            val = E('and', val, C(M32))
        s.x[n] = val

    def getd(self, s, r):
        k, n = reg_kind(r)
        return s.v[n][0]

    def setd(self, s, r, val):
        k, n = reg_kind(r)
        s.v[n] = (val, C(0))

    def addr_of(self, s, mem):
        # mem like "[x0, #0x408]" / "[sp]" / "[x15, w14, uxtw #3]"
        inner = mem.strip('[]!').split(', ')
        base = s.x[reg_kind(inner[0])[1]] if inner[0] != 'sp' else s.x['sp']
        if len(inner) == 1:
            return base, 0, None
        if inner[1].startswith('#'):
            return base, imm(inner[1]), None
        # register index
        idxr = inner[1]
        sh = int(inner[2].split('#')[1]) if len(inner) > 2 else 0
        return base, 0, (self.getx(s, idxr), sh)

    def load8(self, s, base, off):
        if isinstance(base, Ptr):
            o = base.off + off
            if base.region == 'data':
                assert o % 8 == 0
                return mk(('f', o // 8))
            if base.region == 'res':
                assert o % 8 == 0
                return s.res[o // 8]
            if base.region == 'stack':
                return s.stack.get(o, UNDEF)
            raise RuntimeError(base)
        assert is_c(base), NODES[base]
        a = cv(base) + off
        return C(struct.unpack('<Q', read_mem(a, 8))[0])

    def load4(self, s, base, off):
        if isinstance(base, Ptr) and base.region == 'data':
            o = base.off + off
            assert o % 8 == 0
            return E('and', mk(('f', o // 8)), C(M32))
        raise RuntimeError('load4 %r' % base)

    def store8(self, s, base, off, val):
        assert isinstance(base, Ptr)
        o = base.off + off
        if base.region == 'stack':
            s.stack[o] = val
        elif base.region == 'res':
            assert o % 8 == 0
            s.res[o // 8] = val
        else:
            raise RuntimeError('store to %r' % base)

    def table_load(self, s, base, off, idx):
        ix, sh = idx
        assert is_c(base) and sh == 3
        b = cv(base) + off

        def go(e):
            if is_c(e):
                return C(struct.unpack('<Q', read_mem(b + 8 * cv(e), 8))[0])
            n = NODES[e]
            if n[0] == 'and' and is_c(n[2]) and cv(n[2]) == M32:
                return go(n[1])
            if n[0] == 'cond':
                return SEL(e, go(C(1)), go(C(0)))
            if n[0] == 'sel':
                return SEL(n[1], go(n[2]), go(n[3]))
            raise RuntimeError('table idx %r' % (n,))
        return go(ix)

    def step(self, s, a):
        m, o = self.ins[a]
        ops = o.split(', ') if o else []
        if m in ('b',) or m.startswith('b.') or m in ('tbnz', 'tbz', 'ret', 'nop'):
            return
        if m == 'ldr' or m == 'ldur':
            dst = ops[0]
            mem = o[o.index('['):]
            post = None
            if '], #' in mem:          # post-index
                mem, post = mem.split('], ')
                mem = mem + ']'
                post = imm(post)
            pre = mem.endswith('!')
            base, off, idx = self.addr_of(s, mem)
            k = dst[0]
            if idx is not None:
                val = self.table_load(s, base, off, idx)
                self.setd(s, dst, val)
            elif k == 'w':
                self.setx(s, dst, self.load4(s, base, off))
            elif k == 'x':
                self.setx(s, dst, self.load8(s, base, off))
            elif k == 'd':
                self.setd(s, dst, self.load8(s, base, off))
            elif k == 'q':
                n = reg_kind(dst)[1]
                s.v[n] = (self.load8(s, base, off), self.load8(s, base, off + 8))
            else:
                raise RuntimeError((m, o))
            self._wb(s, mem, pre, post, off)
            return
        if m in ('str', 'stur'):
            src = ops[0]
            mem = o[o.index('['):]
            post = None
            if '], #' in mem:
                mem, post = mem.split('], ')
                mem = mem + ']'
                post = imm(post)
            pre = mem.endswith('!')
            base, off, idx = self.addr_of(s, mem)
            k = src[0]
            if k == 'd':
                self.store8(s, base, off, self.getd(s, src))
            elif k == 'x':
                self.store8(s, base, off, self.getx(s, src))
            elif k == 'q':
                n = reg_kind(src)[1]
                self.store8(s, base, off, s.v[n][0])
                self.store8(s, base, off + 8, s.v[n][1])
            else:
                raise RuntimeError((m, o))
            self._wb(s, mem, pre, post, off)
            return
        if m in ('ldp', 'stp'):
            r1, r2 = ops[0], ops[1]
            mem = o[o.index('['):]
            post = None
            if '], #' in mem:
                mem, post = mem.split('], ')
                mem = mem + ']'
                post = imm(post)
            pre = mem.endswith('!')
            base, off, idx = self.addr_of(s, mem)
            sz = 8
            for j, r in enumerate((r1, r2)):
                k = r[0]
                if m == 'ldp':
                    v = self.load8(s, base, off + sz * j)
                    if k == 'd':
                        self.setd(s, r, v)
                    else:
                        self.setx(s, r, v)
                else:
                    v = self.getd(s, r) if k == 'd' else self.getx(s, r)
                    self.store8(s, base, off + sz * j, v)
            self._wb(s, mem, pre, post, off)
            return
        if m == 'adrp' or m == 'adr':
            self.setx(s, ops[0], C(imm(ops[1])))
            return
        if m == 'add' or m == 'sub':
            d, a1, b1 = ops
            if a1 == 'sp':
                p = s.x['sp']
                v = imm(b1)
                s.x['sp'] = Ptr(p.region, p.off + (v if m == 'add' else -v))
                assert d == 'sp'
                return
            x = self.getx(s, a1)
            assert is_c(x) and b1.startswith('#'), (m, o)
            r = cv(x) + imm(b1) if m == 'add' else cv(x) - imm(b1)
            self.setx(s, d, C(r))
            return
        if m == 'mov':
            d, src = ops[0], ops[1]
            if d.startswith('v'):
                # mov vD.d[1], vS.d[0]
                dn = int(d[1:d.index('.')]); sn = int(src[1:src.index('.')])
                assert d.endswith('.d[1]') and src.endswith('.d[0]')
                s.v[dn] = (s.v[dn][0], s.v[sn][0])
                return
            if src.startswith('#'):
                v = imm(src)
                if d[0] == 'w':
                    v &= M32
                self.setx(s, d, C(v))
            else:
                self.setx(s, d, self.getx(s, src))
            return
        if m == 'movk':
            d, v, sh = ops
            x = self.getx(s, d)
            assert is_c(x)
            sh = int(sh.split('#')[1])
            val = (cv(x) & ~(0xffff << sh)) | (imm(v) << sh)
            self.setx(s, d, C(val))
            return
        if m == 'fmov':
            d, src = ops
            if d[0] == 'd' and src[0] == 'd':
                self.setd(s, d, self.getd(s, src))
            elif d[0] == 'd' and src[0] == 'x':
                self.setd(s, d, self.getx(s, src))
            elif d[0] == 'x' and src[0] == 'd':
                self.setx(s, d, self.getd(s, src))
            elif d[0] == 'd' and src.startswith('#'):
                self.setd(s, d, C(d2b(float(src[1:]))))
            else:
                raise RuntimeError((m, o))
            return
        if m == 'fadd':
            d, a1, b1 = ops
            if d.startswith('v'):
                dn, an, bn = [int(x[1:x.index('.')]) for x in ops]
                A, B = s.v[an], s.v[bn]
                s.v[dn] = (E('fadd', A[0], B[0]), E('fadd', A[1], B[1]))
            else:
                self.setd(s, d, E('fadd', self.getd(s, a1), self.getd(s, b1)))
            return
        if m == 'fcmp':
            a1, b1 = ops
            bv = C(d2b(float(b1[1:]))) if b1.startswith('#') else self.getd(s, b1)
            s.nz = E('nzf', self.getd(s, a1), bv)
            return
        if m == 'cmn':
            a1, b1 = ops
            assert a1[0] == 'w'
            bv = C(imm(b1)) if b1.startswith('#') else self.getx(s, b1)
            s.nz = E('nza32', self.getx(s, a1), bv)
            return
        if m == 'cmp':
            a1, b1 = ops
            bv = C(imm(b1)) if b1.startswith('#') else self.getx(s, b1)
            s.nz = E('nzs64' if a1[0] == 'x' else 'nzs32', self.getx(s, a1), bv)
            return
        if m in ('ccmp', 'ccmn', 'fccmp'):
            a1, b1, nzimm, cc = ops
            c = COND(s.nz, cc)
            if m == 'fccmp':
                new = E('nzf', self.getd(s, a1), self.getd(s, b1))
            else:
                bv = C(imm(b1)) if b1.startswith('#') else self.getx(s, b1)
                if m == 'ccmn':
                    assert a1[0] == 'w'
                    new = E('nza32', self.getx(s, a1), bv)
                else:
                    new = E('nzs64' if a1[0] == 'x' else 'nzs32', self.getx(s, a1), bv)
            s.nz = SEL(c, new, C(imm(nzimm)))
            return
        if m == 'fcsel':
            d, a1, b1, cc = ops
            self.setd(s, d, SEL(COND(s.nz, cc), self.getd(s, a1), self.getd(s, b1)))
            return
        if m == 'cset':
            d, cc = ops
            self.setx(s, d, COND(s.nz, cc))
            return
        if m == 'and':
            d, a1, b1 = ops
            self.setx(s, d, E('and', self.getx(s, a1), C(imm(b1)) if b1.startswith('#') else self.getx(s, b1)))
            return
        if m == 'ands':
            d, a1, b1 = ops
            r = E('and', self.getx(s, a1), self.getx(s, b1))
            self.setx(s, d, r)
            s.nz = E('nzl32', r)
            return
        raise RuntimeError('unhandled %x %s %s' % (a, m, o))

    def _wb(self, s, mem, pre, post, off):
        inner = mem.strip('[]!').split(', ')
        if pre or post is not None:
            assert inner[0] == 'sp'
            p = s.x['sp']
            s.x['sp'] = Ptr(p.region, p.off + (off if pre else post))

    def branch_cond(self, s, a):
        """Condition (0/1 expr) that the branch at a is taken."""
        m, o = self.ins[a]
        ops = o.split(', ')
        if m.startswith('b.'):
            return COND(s.nz, m[2:])
        if m in ('tbnz', 'tbz'):
            x = self.getx(s, ops[0])
            bit = imm(ops[1])
            assert bit == 0
            c = self.bit0(x)
            if m == 'tbz':
                c = self.notc(c)
            return c
        raise RuntimeError(m)

    def bit0(self, x):
        n = NODES[x]
        if is_c(x):
            return C(cv(x) & 1)
        if n[0] == 'and' and is_c(n[2]) and (cv(n[2]) & 1):
            return self.bit0(n[1])
        if n[0] == 'cond':
            return x
        if n[0] == 'sel':
            return SEL(n[1], self.bit0(n[2]), self.bit0(n[3]))
        return mk(('bit0', x))

    def notc(self, c):
        return SEL(c, C(0), C(1))

    # ------------------------------------------------------------ execution
    def run_region(self, b, s, stop):
        while b != stop:
            body, succ = self.blocks[b]
            for a in body:
                self.step(s, a)
            if not succ:
                assert stop == 'EXIT'
                return s, 'EXIT'
            if len(succ) == 1:
                b = succ[0]
                continue
            c = self.branch_cond(s, body[-1])
            j = self.ipdom[b]
            if is_c(c):
                b = succ[0] if cv(c) else succ[1]
                continue
            st, _ = self.run_region(succ[0], s.copy(), j)
            sf, _ = self.run_region(succ[1], s, j)
            s = merge(c, st, sf)
            b = j
        return s, b

    def lift(self):
        s, _ = self.run_region(self.addr, initial_state(), 'EXIT')
        assert s.x['sp'] == Ptr('stack', 0)
        return [s.res[k] for k in range(4)]


if __name__ == '__main__':
    units = unit_symbols()
    out = {}
    import time
    for i in sorted(units):
        t = time.time()
        u = Unit(i, *units[i])
        out[i] = u.lift()
        print(i, len(NODES), '%.1fs' % (time.time() - t), flush=True)
    pickle.dump({'nodes': NODES, 'units': out}, open(sys.argv[1] if len(sys.argv) > 1 else 'lifted.pkl', 'wb'))
