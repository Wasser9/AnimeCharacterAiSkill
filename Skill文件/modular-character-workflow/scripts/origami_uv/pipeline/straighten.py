"""Boundary straightening for face labels on a triangle mesh without edge loops.

Each boundary chain between two regions (A|B) is split into short segments; every segment is replaced
by the shortest edge path between its end points inside a narrow corridor around the old boundary, and
the corridor faces are re-assigned to A or B by flood fill on either side of the new path.
Junctions where three or more regions meet, mesh borders and non-manifold vertices stay fixed.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
from collections import defaultdict, deque


def straighten(m, labels, band=0.012, seg_len=0.06, min_edges=3, passes=3):
    labels = labels.copy()
    nvw = m.nvw
    pos = np.zeros((nvw, 3)); pos[m.Fw.ravel()] = m.co[m.F.ravel()]
    Fw = m.Fw
    # global edge graph (welded)
    e = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    L = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1) + 1e-9
    G = sp.coo_matrix((np.r_[L, L], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nvw, nvw)).tocsr()
    # faces around each welded vertex
    vf = sp.coo_matrix((np.ones(3 * m.nf), (Fw.ravel(), np.repeat(np.arange(m.nf), 3))), shape=(nvw, m.nf)).tocsr()
    # vertices that must stay fixed: mesh border / non-manifold edge vertices
    bad_keys = np.r_[m.boundary_keys, m.nonmanifold_keys]
    fixed_v = np.zeros(nvw, bool)
    fixed_v[bad_keys // nvw] = True; fixed_v[bad_keys % nvw] = True
    a1 = m.adj_c1; f1 = m.adj_f1; f2 = m.adj_f2
    ea = Fw.ravel()[a1]; eb = Fw[:, [1, 2, 0]].ravel()[a1]
    # face adjacency (manifold) for flood fill, with edge keys
    for _ in range(passes):
        la = labels[f1]; lb = labels[f2]
        isb = (la != lb) & (la >= 0) & (lb >= 0)
        # number of distinct labels around each vertex
        vv = Fw.ravel(); ll = np.repeat(labels, 3)
        pairs = np.unique(vv.astype(np.int64) * (labels.max() + 3) + (ll + 2))
        nlab = np.bincount(pairs // (labels.max() + 3), minlength=nvw)
        junction = (nlab >= 3) | fixed_v
        pa = np.minimum(la[isb], lb[isb]); pb = np.maximum(la[isb], lb[isb])
        E_a = ea[isb]; E_b = eb[isb]
        chains_by_pair = defaultdict(list)
        for x, y, u, v in zip(pa, pb, E_a, E_b):
            chains_by_pair[(int(x), int(y))].append((int(u), int(v)))
        for (A, B), edges in chains_by_pair.items():
            for chain in _chains(edges, junction):
                for seg in _split(chain, pos, seg_len):
                    if len(seg) - 1 < min_edges:
                        continue
                    labels = _relax_segment(m, labels, seg, A, B, pos, G, vf, junction, band)
    return labels


def _chains(edges, junction):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    nodes = {v for v, n in adj.items() if len(n) != 2 or junction[v]}
    seen = set()
    out = []
    for s in nodes:
        for nb in adj[s]:
            if (s, nb) in seen:
                continue
            path = [s, nb]; seen.add((s, nb)); seen.add((nb, s))
            while path[-1] not in nodes:
                cur = path[-1]
                nxt = [w for w in adj[cur] if (cur, w) not in seen]
                if not nxt:
                    break
                w = nxt[0]; seen.add((cur, w)); seen.add((w, cur)); path.append(w)
            out.append(path)
    # closed loops without any node
    rest = {v for v in adj if v not in nodes}
    for v0 in list(rest):
        if any((v0, w) in seen for w in adj[v0]):
            continue
        path = [v0]; cur = v0; prev = None
        while True:
            nxt = [w for w in adj[cur] if w != prev and (cur, w) not in seen]
            if not nxt:
                break
            w = nxt[0]; seen.add((cur, w)); seen.add((w, cur)); path.append(w); prev, cur = cur, w
            if w == v0:
                break
        if len(path) > 4:
            h = len(path) // 2
            out.append(path[:h + 1]); out.append(path[h:])
    return out


def _split(chain, pos, seg_len):
    p = pos[chain]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    total = d[-1]
    n = max(1, int(round(total / seg_len)))
    cuts = [0] + [int(np.searchsorted(d, total * k / n)) for k in range(1, n)] + [len(chain) - 1]
    cuts = sorted(set(cuts))
    return [chain[a:b + 1] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _relax_segment(m, labels, seg, A, B, pos, G, vf, junction, band):
    s, t = seg[0], seg[-1]
    # corridor vertices: within `band` of the old segment (at least ~3 local edge lengths wide)
    el = np.linalg.norm(np.diff(pos[seg], axis=0), axis=1).mean()
    band = max(band, 3.0 * el)
    dist = cg.dijkstra(G, directed=False, indices=np.asarray(seg), min_only=True, limit=band)
    cv = np.nonzero(np.isfinite(dist))[0]
    # corridor faces: faces with all vertices in the corridor and label A/B
    inC = np.zeros(m.nvw, bool); inC[cv] = True
    cand = np.unique(vf[cv].indices)
    cand = cand[np.isin(labels[cand], (A, B)) & inC[m.Fw[cand]].all(1)]
    if len(cand) < 3:
        return labels
    # allowed vertices: corridor vertices only touching A/B faces, no junctions except s/t
    vfl = vf[cv]
    ok = np.ones(len(cv), bool)
    for i, v in enumerate(cv):
        fl = vfl.indices[vfl.indptr[i]:vfl.indptr[i + 1]]
        if (~np.isin(labels[fl], (A, B))).any() or (junction[v] and v not in (s, t)):
            ok[i] = False
    allowed = np.zeros(m.nvw, bool); allowed[cv[ok]] = True
    allowed[s] = allowed[t] = True
    # sub-graph of edges of corridor faces between allowed vertices
    Fc = m.Fw[cand]
    e = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    e = e[allowed[e[:, 0]] & allowed[e[:, 1]]]
    if not len(e):
        return labels
    w = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1) + 1e-9
    Gs = sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(m.nvw, m.nvw)).tocsr()
    d, pred = cg.dijkstra(Gs, directed=False, indices=s, return_predecessors=True)
    if not np.isfinite(d[t]):
        return labels
    old_len = np.linalg.norm(np.diff(pos[seg], axis=0), axis=1).sum()
    if d[t] >= old_len - 1e-9:
        return labels
    path = [t]
    while path[-1] != s:
        path.append(pred[path[-1]])
    path = path[::-1]
    pk = set((min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:]))
    # flood fill corridor faces from their non-corridor neighbours, never crossing the new path
    cset = set(cand.tolist())
    lab_new = {}
    dq = deque()
    nf_adj = _face_adj(m)
    for f in cand:
        for g, (a, b) in nf_adj[f]:
            if g not in cset and labels[g] in (A, B) and (min(a, b), max(a, b)) not in pk:
                if f not in lab_new:
                    lab_new[f] = labels[g]; dq.append(f)
    if not dq:
        return labels
    while dq:
        f = dq.popleft()
        for g, (a, b) in nf_adj[f]:
            if g in cset and g not in lab_new and (min(a, b), max(a, b)) not in pk:
                lab_new[g] = lab_new[f]; dq.append(g)
    out = labels.copy()
    for f, l in lab_new.items():
        out[f] = l
    return out


def _face_adj(m):
    """Face adjacency with the shared welded edge, cached on the mesh itself. (A module cache keyed by id(m) handed a
    later mesh the adjacency of a freed one whenever CPython reused the address, i.e. when one worker process unwrapped
    two objects in a row - results then depended on the number of workers.)"""
    adj = getattr(m, '_face_adj_cache', None)
    if adj is None:
        adj = [[] for _ in range(m.nf)]
        ea = m.Fw.ravel()[m.adj_c1]; eb = m.Fw[:, [1, 2, 0]].ravel()[m.adj_c1]
        for f1, f2, a, b in zip(m.adj_f1.tolist(), m.adj_f2.tolist(), ea.tolist(), eb.tolist()):
            adj[f1].append((f2, (a, b))); adj[f2].append((f1, (a, b)))
        m._face_adj_cache = adj
    return adj
