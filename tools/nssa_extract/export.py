"""Convert lifted DAGs into an explicit tree ensemble (trees.json)."""
import json, pickle, struct, sys

M32 = 0xFFFFFFFF


def b2d(b):
    return struct.unpack('<d', struct.pack('<Q', b))[0]


def main(pkl, out):
    D = pickle.load(open(pkl, 'rb'))
    N = D['nodes']
    trees = []          # each: {"nodes": [...]}   node: [feat, kind, thr, yes, no] / leaf: [value]
    tree_ids = {}       # expr id -> tree index
    stats = {'nodes': 0, 'leaves': 0}

    def elem(flags, cc):
        """Return ('miss', k, positive) or ('cmp', k, cc, t) or None."""
        t = N[flags]
        if t[0] == 'nzf':
            a, b = N[t[1]], N[t[2]]
            if a[0] == 'f' and b[0] == 'c':
                return ('cmp', a[1], cc, b2d(b[1]))
            raise RuntimeError('nzf form %r %r' % (a, b))
        if t[0] in ('nza32', 'nzs64', 'nzs32'):
            a, b = N[t[1]], N[t[2]]
            while a[0] == 'and' and N[a[2]] == ('c', M32) and N[a[1]][0] == 'and' and N[N[a[1]][2]] == ('c', M32):
                a = N[a[1]]
            if a[0] == 'and' and N[a[2]] == ('c', M32) and N[a[1]][0] == 'f' and b[0] == 'c':
                k = N[a[1]][1]
                if (t[0] == 'nza32' and b[1] == 1) or (t[0] in ('nzs64', 'nzs32') and b[1] == M32):
                    if cc == 'eq':
                        return ('miss', k, True)
                    if cc == 'ne':
                        return ('miss', k, False)
            raise RuntimeError('int flag form %r cc %s a=%r b=%r' % (t, cc, a, b))
        raise RuntimeError('flags %r' % (t,))

    def build(e, nodes):
        """Append decision tree for value e; return node index."""
        t = N[e]
        if t[0] == 'c':
            nodes.append([b2d(t[1])])
            stats['leaves'] += 1
            return len(nodes) - 1
        if t[0] == 'sel':
            return branch(t[1], lambda: build(t[2], nodes), lambda: build(t[3], nodes), nodes)
        raise RuntimeError('value %r' % (t,))

    def branch(c, mk_yes, mk_no, nodes):
        t = N[c]
        if t[0] == 'c':
            return mk_yes() if t[1] else mk_no()
        if t[0] == 'cond':
            fl = N[t[1]]
            if fl[0] == 'sel':
                c2 = fl[1]
                return branch(c2,
                              lambda: branch(mkcond(fl[2], t[2]), mk_yes, mk_no, nodes),
                              lambda: branch(mkcond(fl[3], t[2]), mk_yes, mk_no, nodes), nodes)
            if fl[0] == 'c':
                from nssa_extract import lift as L
                return mk_yes() if L.cond_holds(fl[1], t[2]) else mk_no()
            if fl[0] == 'nzl32' and t[2] in ('ne', 'eq'):
                # operand is an AND of 0/1 cset values -> nonzero == bit0
                def chk(x):
                    u = N[x]
                    if u[0] == 'and':
                        return chk(u[1]) and chk(u[2])
                    return u[0] == 'cond' or (u[0] == 'c' and u[1] in (0, 1, 0xFFFFFFFF))
                assert chk(fl[1]), N[fl[1]]
                if t[2] == 'ne':
                    return bitbranch(fl[1], mk_yes, mk_no, nodes)
                return bitbranch(fl[1], mk_no, mk_yes, nodes)
            # push selects inside flag operands outward
            for j, a in enumerate(fl[1:], 1):
                if isinstance(a, int) and N[a][0] == 'sel':
                    sa = N[a]
                    def sub(v, j=j):
                        lst = list(fl); lst[j] = v
                        return mkcond(newnode(tuple(lst)), t[2])
                    return branch(sa[1], lambda: branch(sub(sa[2]), mk_yes, mk_no, nodes),
                                  lambda: branch(sub(sa[3]), mk_yes, mk_no, nodes), nodes)
                if isinstance(a, int) and N[a][0] == 'and' and N[N[a][1]][0] == 'sel' and fl[0] != 'nzf':
                    sa = N[N[a][1]]
                    def sub2(v, j=j, a=a):
                        lst = list(fl); lst[j] = newnode(('and', v, N[a][2]))
                        return mkcond(newnode(tuple(lst)), t[2])
                    return branch(sa[1], lambda: branch(sub2(sa[2]), mk_yes, mk_no, nodes),
                                  lambda: branch(sub2(sa[3]), mk_yes, mk_no, nodes), nodes)
            el = elem(t[1], t[2])
            if el[0] == 'miss':
                k = el[1]
                if k in facts_miss:
                    return mk_yes() if facts_miss[k] == el[2] else mk_no()
                idx = len(nodes); nodes.append(None)
                facts_miss[k] = True
                ym = mk_yes() if el[2] else mk_no()
                facts_miss[k] = False
                nm = mk_no() if el[2] else mk_yes()
                del facts_miss[k]
                nodes[idx] = ['miss', k, ym, nm]       # first child == feature missing
                stats['nodes'] += 1
                return idx
            _, k, cc, thr = el
            hv = known_hi(k, thr)
            if hv is not None:
                taken = hv if cc == 'hi' else (not hv)
                return mk_yes() if taken else mk_no()
            idx = len(nodes); nodes.append(None)
            lst = facts_cmp.setdefault(k, [])
            lst.append((thr, cc == 'hi'))
            y = mk_yes()
            lst[-1] = (thr, cc != 'hi')
            n = mk_no()
            lst.pop()
            nodes[idx] = ['cmp', k, cc, thr, y, n]
            stats['nodes'] += 1
            return idx
        if t[0] == 'sel':
            # boolean select: sel(c2, b1, b2) as condition
            return branch(t[1], lambda: branch(t[2], mk_yes, mk_no, nodes),
                          lambda: branch(t[3], mk_yes, mk_no, nodes), nodes)
        if t[0] == 'bit0':
            return bitbranch(t[1], mk_yes, mk_no, nodes)
        raise RuntimeError('cond %r' % (t,))

    cond_cache = {}
    facts_miss = {}
    facts_cmp = {}

    def known_hi(k, thr):
        # hi(t) == (x > t) or NaN ; facts are (t1, hi_value)
        for t1, hv in facts_cmp.get(k, ()):
            if hv and t1 >= thr:
                return True
            if not hv and t1 <= thr:
                return False
        return None

    def bitbranch(x, mk_yes, mk_no, nodes):
        u = N[x]
        if u[0] == 'c':
            return mk_yes() if u[1] & 1 else mk_no()
        if u[0] == 'cond':
            return branch(x, mk_yes, mk_no, nodes)
        if u[0] == 'and':
            a, b = u[1], u[2]
            if N[b][0] == 'c':
                return bitbranch(a, mk_yes, mk_no, nodes) if N[b][1] & 1 else mk_no()
            return bitbranch(a, lambda: bitbranch(b, mk_yes, mk_no, nodes), mk_no, nodes)
        if u[0] == 'sel':
            return branch(u[1], lambda: bitbranch(u[2], mk_yes, mk_no, nodes),
                          lambda: bitbranch(u[3], mk_yes, mk_no, nodes), nodes)
        raise RuntimeError('bit0 of %r' % (u,))

    def mkcond(flags, cc):
        # build/locate a ('cond', flags, cc) node or constant
        t = N[flags]
        if t[0] == 'c':
            from nssa_extract.lift import cond_holds
            return const(cond_holds(t[1], cc))
        key = ('cond', flags, cc)
        if key not in cond_cache:
            N.append(key)
            cond_cache[key] = len(N) - 1
        return cond_cache[key]

    const_cache = {}
    node_cache = {}

    def newnode(t):
        from nssa_extract import lift as L
        if t[0] in L.FOLD and all(N[a][0] == 'c' for a in t[1:]):
            return const(L.FOLD[t[0]](*[N[a][1] for a in t[1:]]))
        if t not in node_cache:
            N.append(t)
            node_cache[t] = len(N) - 1
        return node_cache[t]

    def const(v):
        if v not in const_cache:
            N.append(('c', v))
            const_cache[v] = len(N) - 1
        return const_cache[v]

    def get_tree(e):
        if e not in tree_ids:
            nodes = []
            build(e, nodes)
            tree_ids[e] = len(trees)
            trees.append(reduce_tree(nodes))
        return tree_ids[e]

    norm_cache = {}

    def reduce_tree(nodes):
        table = {}      # canonical key -> new index
        out = []
        memo = {}
        sys.setrecursionlimit(100000)

        def canon(i):
            if i in memo:
                return memo[i]
            nd = nodes[i]
            if len(nd) == 1:
                key = ('L', struct.pack('<d', nd[0]))
            elif nd[0] == 'miss':
                a, b = canon(nd[2]), canon(nd[3])
                if a == b:
                    memo[i] = a
                    return a
                kb = out[b]
                if kb[0] == 'C' and kb[1] == nd[1] and a in (kb[4], kb[5]):
                    key = ('T', kb[1], kb[2], kb[3], a == kb[4], kb[4], kb[5])
                else:
                    key = ('M', nd[1], a, b)
            else:
                a, b = canon(nd[4]), canon(nd[5])
                if a == b:
                    memo[i] = a
                    return a
                key = ('C', nd[1], nd[2], nd[3], a, b)
            r = table.get(key)
            if r is None:
                r = len(out)
                out.append(key)
                table[key] = r
            memo[i] = r
            return r
        root = canon(0)
        # re-emit depth first from root so index 0 is the root (tree, not DAG)
        res = []

        def emit(k):
            key = out[k]
            idx = len(res)
            res.append(None)
            if key[0] == 'L':
                res[idx] = [struct.unpack('<d', key[1])[0]]
            elif key[0] == 'M':
                y = emit(key[2]); n = emit(key[3])
                res[idx] = ['miss', key[1], y, n]
            elif key[0] == 'T':
                y = emit(key[5]); n = emit(key[6])
                res[idx] = ['test', key[1], key[2], key[3], key[4], y, n]
            else:
                y = emit(key[4]); n = emit(key[5])
                res[idx] = ['cmp', key[1], key[2], key[3], y, n]
            return idx
        emit(root)
        return res

    def norm(e):
        if e in norm_cache:
            return norm_cache[e]
        t = N[e]
        r = e
        if t[0] == 'sel':
            a, b = norm(t[2]), norm(t[3])
            if a == b:
                r = a
            elif N[a][0] == 'fadd' and N[b][0] == 'fadd' and N[a][1] == N[b][1]:
                r = newnode(('fadd', N[a][1], norm(newnode(('sel', t[1], N[a][2], N[b][2])))))
            elif (a, b) != (t[2], t[3]):
                r = newnode(('sel', t[1], a, b))
        elif t[0] == 'fadd':
            a, b = norm(t[1]), norm(t[2])
            if (a, b) != (t[1], t[2]):
                r = newnode(('fadd', a, b))
        norm_cache[e] = r
        return r

    def sumexpr(e):
        e = norm(e)
        t = N[e]
        if t[0] == 'fadd':
            return ['+', sumexpr(t[1]), sumexpr(t[2])]
        if t[0] == 'res':
            return ['res', t[1]]
        return ['tree', get_tree(e)]

    units = []
    for u in sorted(D['units']):
        units.append([sumexpr(r) for r in D['units'][u]])
    # sanity: every unit slot k must reference res k
    out_obj = {
        'format': 'oura-nssa-treelite-v1',
        'num_class': 4, 'num_feature': 182,
        'doc': ('units[u][k] is the exact fadd expression the native unit u stores into result[k]; '
                'evaluate units in order 0..62 with result starting at 0.0, then result[k] += 0.0 '
                '(global bias), then softmax. Tree node forms: [leaf_value] | '
                '["miss", f, child_if_missing, child_if_present] | ["cmp", f, cc, thr, yes, no] | '
                '["test", f, cc, thr, missing_goes_yes, yes, no] (missing -> yes/no per flag, else cmp). '
                'cc is the AArch64 condition after fcmp(x_f, thr): ls = x<=thr (false if unordered), '
                'hi = x>thr or unordered. A feature is missing when the low 32 bits of its Entry are '
                '0xFFFFFFFF; x_f is the Entry reinterpreted as double (a missing Entry reads as '
                '2.1219957905e-314 if compared). Node 0 is the root; child indices are absolute.'),
        'units': units,
        'trees': trees,
    }
    json.dump(out_obj, open(out, 'w'), separators=(',', ':'))
    ni = sum(1 for t in trees for n in t if len(n) > 1)
    nl = sum(1 for t in trees for n in t if len(n) == 1)
    print('trees', len(trees), 'internal nodes', ni, 'leaves', nl, '(before reduction %d)' % stats['nodes'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
