"""Core mesh topology / chart / parameterisation utilities for the origami UV pipeline.

Vocabulary
  face labels  : semantic region id per triangle (from segment.py)
  cut keys     : undirected welded-edge keys that must become UV seams
  chart        : connected set of faces with the same label, not crossing a cut
  uv vertex    : group of face corners that share one UV coordinate
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
from scipy.spatial import cKDTree
import igl


class Mesh:
    def __init__(self, path, weld_tol=1e-6):
        d = np.load(path)
        self.name = str(path).replace('\\', '/').split('/')[-1][:-4]
        self.co = d['co'].astype(np.float64)
        self.F = d['loop_vert'].reshape(-1, 3).astype(np.int64)
        self.uv_orig = d['uv_UV_orig'].reshape(-1, 3, 2).astype(np.float64)
        self.W = d['weights'].astype(np.float64)
        self.groups = [str(g) for g in d['groups']]
        self.prev_panel = d['attr_FACE_UV_panel'] if 'attr_FACE_UV_panel' in d else None
        nv = len(self.co)
        # weld coincident vertices for topology only (never written back)
        pairs = cKDTree(self.co).query_pairs(weld_tol, output_type='ndarray')
        if len(pairs):
            g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(nv, nv))
            _, rep = cg.connected_components(g, directed=False)
        else:
            rep = np.arange(nv)
        self.weld = rep
        self.nf = len(self.F)
        self.F_src = self.F.copy()
        # Working copy with locally consistent winding (AI meshes contain flipped faces).
        # corner (f,k) of the working mesh is source loop 3f+perm[f,k].
        flip = self._orientation(rep[self.F])
        self.flipped = flip
        self.perm = np.tile(np.arange(3), (self.nf, 1))
        self.perm[flip] = [0, 2, 1]
        self.F = np.take_along_axis(self.F_src, self.perm, 1)
        self.uv_orig = np.take_along_axis(self.uv_orig, self.perm[:, :, None], 1)
        self.Fw = rep[self.F]
        a, b, c = (self.co[self.F[:, i]] for i in range(3))
        n = np.cross(b - a, c - a)
        self.area = 0.5 * np.linalg.norm(n, axis=1)
        self.normal = n / (2 * self.area[:, None] + 1e-30)
        self.cent = (a + b + c) / 3
        self._edges()

    def bone_index(self, name):
        return self.groups.index(name)

    def _orientation(self, Fw):
        """Choose per-face flips so that faces sharing a 2-manifold edge have opposite directed edges."""
        nf = len(Fw)
        nvw = int(Fw.max()) + 1
        va = Fw.ravel(); vb = Fw[:, [1, 2, 0]].ravel()
        key = np.minimum(va, vb) * nvw + np.maximum(va, vb)
        order = np.argsort(key, kind='stable'); ks = key[order]
        starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
        counts = np.diff(np.r_[starts, len(ks)])
        two = counts == 2
        c1 = order[starts[two]]; c2 = order[starts[two] + 1]
        parity = (va[c1] == va[c2]).astype(np.int8)  # same direction -> one of them must flip
        f1 = c1 // 3; f2 = c2 // 3
        g = sp.coo_matrix((np.ones(len(f1)), (f1, f2)), shape=(nf, nf)).tocsr()
        ncomp, comp = cg.connected_components(g, directed=False)
        # BFS spanning tree to assign flips
        adj = [[] for _ in range(nf)]
        for a_, b_, p in zip(f1.tolist(), f2.tolist(), parity.tolist()):
            adj[a_].append((b_, p)); adj[b_].append((a_, p))
        flip = np.full(nf, -1, np.int8)
        from collections import deque
        for s in range(nf):
            if flip[s] >= 0:
                continue
            flip[s] = 0; dq = deque([s])
            while dq:
                u = dq.popleft()
                for v, p in adj[u]:
                    if flip[v] < 0:
                        flip[v] = flip[u] ^ p; dq.append(v)
        flip = flip.astype(bool)
        a, b, c = (self.co[self.F[:, i]] for i in range(3))
        ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        for k in range(ncomp):
            m = comp == k
            if ar[m & flip].sum() > ar[m & ~flip].sum():
                flip[m] = ~flip[m]
        self.n_orient_conflicts = int(((flip[f1] ^ flip[f2]) != parity.astype(bool)).sum())
        return flip

    def _edges(self):
        Fw = self.Fw
        nvw = int(Fw.max()) + 1
        self.nvw = nvw
        va = Fw.ravel()  # corner c=3f+k : edge from vertex(c) to vertex(next corner)
        vb = Fw[:, [1, 2, 0]].ravel()
        lo = np.minimum(va, vb); hi = np.maximum(va, vb)
        key = lo * nvw + hi
        self.ekey = key.reshape(-1, 3)
        order = np.argsort(key, kind='stable')
        ks = key[order]
        starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
        counts = np.diff(np.r_[starts, len(ks)])
        self.edge_count = dict()
        uk = ks[starts]
        self.ukeys = uk; self.ucounts = counts
        two = counts == 2
        c1 = order[starts[two]]; c2 = order[starts[two] + 1]
        # consistent orientation: directed edges must be opposite
        cons = va[c1] == vb[c2]
        c1 = c1[cons]; c2 = c2[cons]
        self.adj_c1 = c1; self.adj_c2 = c2
        self.adj_f1 = c1 // 3; self.adj_f2 = c2 // 3
        self.adj_key = key[c1]
        self.nonmanifold_keys = uk[counts > 2]
        self.boundary_keys = uk[counts == 1]
        # candidate pairs across non-manifold edges (opposite directed corners); used per label in Charts
        nm1 = []; nm2 = []
        for st, ct in zip(starts[counts > 2], counts[counts > 2]):
            cs = order[st:st + ct]
            for i in range(ct):
                for j in range(i + 1, ct):
                    if va[cs[i]] == vb[cs[j]]:
                        nm1.append(cs[i]); nm2.append(cs[j])
        self.nm_c1 = np.asarray(nm1, np.int64); self.nm_c2 = np.asarray(nm2, np.int64)
        self.nm_key = key[self.nm_c1] if len(nm1) else np.zeros(0, np.int64)

    def edge_key(self, wa, wb):
        wa = np.asarray(wa); wb = np.asarray(wb)
        return np.minimum(wa, wb) * self.nvw + np.maximum(wa, wb)

    def face_adjacency(self, mask_pairs=None):
        f1, f2 = self.adj_f1, self.adj_f2
        if mask_pairs is not None:
            f1 = f1[mask_pairs]; f2 = f2[mask_pairs]
        return sp.coo_matrix((np.ones(len(f1)), (f1, f2)), shape=(self.nf, self.nf)).tocsr()


def nxt(c):
    return c - c % 3 + (c % 3 + 1) % 3


class Charts:
    """Charts induced by labels + cut keys."""

    def __init__(self, mesh, labels, cut_keys=()):
        self.m = m = mesh
        self.labels = labels
        cut_keys = np.unique(np.asarray(list(cut_keys), dtype=np.int64))
        self.cut_keys = cut_keys
        valid = labels[m.adj_f1] == labels[m.adj_f2]
        if len(cut_keys):
            valid &= ~np.isin(m.adj_key, cut_keys)
        self.valid = valid
        c1 = m.adj_c1[valid]; c2 = m.adj_c2[valid]
        f1n = m.adj_f1[valid]; f2n = m.adj_f2[valid]
        # non-manifold edges: join two faces of the same label if they are the only such pair on that edge
        if len(m.nm_c1):
            n1 = m.nm_c1; n2 = m.nm_c2
            l1 = labels[n1 // 3]; l2 = labels[n2 // 3]
            ok = (l1 == l2) & (l1 >= 0)
            # only continue the surface across a non-manifold edge where it is (nearly) smooth
            ok &= (m.normal[n1 // 3] * m.normal[n2 // 3]).sum(1) > 0.5
            if len(cut_keys):
                ok &= ~np.isin(m.nm_key, cut_keys)
            kk = m.nm_key * (labels.max() + 2) + l1
            u, cnt = np.unique(kk[ok], return_counts=True)
            single = np.isin(kk, u[cnt == 1]) & ok
            # a face may only take part once per edge key -> already guaranteed by single pair
            c1 = np.r_[c1, n1[single]]; c2 = np.r_[c2, n2[single]]
            f1n = np.r_[f1n, n1[single] // 3]; f2n = np.r_[f2n, n2[single] // 3]
        # corner c1 = (a->b) in f1, c2 = (b->a) in f2 : a corners: c1 , nxt(c2) ; b corners: nxt(c1), c2
        ra = np.r_[c1, nxt(c1)]; rb = np.r_[nxt(c2), c2]
        nc = 3 * m.nf
        g = sp.coo_matrix((np.ones(len(ra)), (ra, rb)), shape=(nc, nc))
        _, self.uvv = cg.connected_components(g, directed=False)  # uv vertex per corner
        fg = sp.coo_matrix((np.ones(len(f1n)), (f1n, f2n)), shape=(m.nf, m.nf))
        self.n, self.chart = cg.connected_components(fg, directed=False)
        self.uvv_w = np.zeros(self.uvv.max() + 1, np.int64)
        self.uvv_w[self.uvv] = m.Fw.ravel()
        self.uvv_v = np.zeros(self.uvv.max() + 1, np.int64)
        self.uvv_v[self.uvv] = m.F.ravel()
        order = np.argsort(self.chart, kind='stable')
        cs = self.chart[order]
        st = np.r_[0, np.nonzero(np.diff(cs))[0] + 1, len(cs)]
        self.faces = [order[st[i]:st[i + 1]] for i in range(self.n)]

    def local(self, ci):
        """Local manifold mesh of chart ci: (faces, V3 (n,3), Fl (f,3), uv-vertex ids (n,))."""
        fs = self.faces[ci]
        corners = self.uvv.reshape(-1, 3)[fs]
        ids, Fl = np.unique(corners, return_inverse=True)
        Fl = Fl.reshape(-1, 3)
        V = self.m.co[self.uvv_v[ids]]
        return fs, V, Fl, ids


def topology(Fl, nv):
    e = np.sort(np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]), 1)
    ek = e[:, 0] * nv + e[:, 1]
    uk, cnt = np.unique(ek, return_counts=True)
    used = np.unique(Fl)
    V = len(used); E = len(uk); Fc = len(Fl)
    chi = V - E + Fc
    try:
        loops = igl.boundary_loop_all(Fl.astype(np.int64))
    except Exception:
        loops = []
    b = len(loops)
    genus = (2 - b - chi) // 2
    return dict(V=V, E=E, F=Fc, chi=chi, loops=loops, b=b, genus=genus, nonmanifold=int((cnt > 2).sum()))


def edge_graph(V, Fl, face_cost):
    e = np.sort(np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]), 1)
    fc = np.tile(np.asarray(face_cost, float), 3)
    n = len(V)
    k = e[:, 0] * n + e[:, 1]
    uk, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    cost = np.bincount(inv, fc) / cnt
    a = uk // n; b = uk % n
    L = np.linalg.norm(V[a] - V[b], axis=1)
    w = L * cost + 1e-9
    g = sp.coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return g


def shortest_path(g, sources, targets):
    """Multi-source Dijkstra; return vertex path from a source to the nearest target."""
    dist, pred, src = cg.dijkstra(g, directed=False, indices=np.asarray(sources), min_only=True, return_predecessors=True)
    t = np.asarray(targets)
    t = t[np.isfinite(dist[t])]
    if len(t) == 0:
        return None
    end = t[np.argmin(dist[t])]
    path = [end]
    while pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    return path[::-1]


def disk_cuts(V, Fl, ids_w, face_cost, nvw, closed_axis=None, max_iter=50):
    """Return extra global cut keys (welded) that turn the local mesh into a disk.

    face_cost : per-face multiplier for seam cost (low = good place for a seam).
    closed_axis : 3D direction used to pick the two ends of the cut on closed surfaces.
    """
    topo = topology(Fl, len(V))
    cuts = []
    g = edge_graph(V, Fl, face_cost)
    if topo['b'] == 0:
        # closed surface: slit between the two extreme cheap vertices along closed_axis
        vc = np.zeros(len(V)); cnt = np.zeros(len(V))
        np.add.at(vc, Fl.ravel(), np.repeat(face_cost, 3)); np.add.at(cnt, Fl.ravel(), 1)
        vc /= np.maximum(cnt, 1)
        cheap = vc <= np.quantile(vc, 0.3)
        ax = np.asarray(closed_axis if closed_axis is not None else _principal_axis(V), float)
        s = V @ ax
        cand = np.nonzero(cheap)[0]
        a = cand[np.argmin(s[cand])]; b = cand[np.argmax(s[cand])]
        p = shortest_path(g, [a], [b])
        if p is not None:
            cuts.append(p)
        return cuts, topo
    if topo['b'] >= 2:
        loops = [np.asarray(l) for l in topo['loops']]
        loops.sort(key=len, reverse=True)
        src = list(loops[0]); remaining = loops[1:]
        while remaining:
            tgt = np.concatenate(remaining)
            p = shortest_path(g, src, tgt)
            if p is None:
                break
            cuts.append(p)
            end = p[-1]
            hit = [i for i, l in enumerate(remaining) if end in set(l.tolist())][0]
            src += list(remaining[hit]) + p
            remaining.pop(hit)
    return cuts, topo


def _principal_axis(V):
    c = V - V.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    return v[:, -1]


def path_keys(path, ids_w, nvw):
    p = np.asarray(path)
    a = ids_w[p[:-1]]; b = ids_w[p[1:]]
    return np.minimum(a, b) * nvw + np.maximum(a, b)


# ---------------------------------------------------------------- parameterisation

def tri_frames(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1 = b - a; e2 = c - a
    x = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-30)
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    y = np.cross(n / nn, x)
    P = np.stack([np.stack([(e1 * x).sum(1), (e1 * y).sum(1)], 1), np.stack([(e2 * x).sum(1), (e2 * y).sum(1)], 1)], 2)
    return P, nn[:, 0] * 0.5  # 2x2 columns = edges in local frame, area


def singular_values(V, F, uv):
    P, A3 = tri_frames(V, F)
    a, b, c = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
    Q = np.stack([b - a, c - a], 2)  # 2x2 columns
    det = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    ok = np.abs(det) > 1e-18
    Pinv = np.zeros_like(P)
    Pinv[ok, 0, 0] = P[ok, 1, 1] / det[ok]; Pinv[ok, 1, 1] = P[ok, 0, 0] / det[ok]
    Pinv[ok, 0, 1] = -P[ok, 0, 1] / det[ok]; Pinv[ok, 1, 0] = -P[ok, 1, 0] / det[ok]
    J = Q @ Pinv
    s = np.linalg.svd(J, compute_uv=False)
    signed = np.sign(np.linalg.det(J))
    A2 = 0.5 * (Q[:, 0, 0] * Q[:, 1, 1] - Q[:, 0, 1] * Q[:, 1, 0])
    return s, signed, A3, A2, ok


def distortion_stats(V, F, uv):
    s, sg, A3, A2, ok = singular_values(V, F, uv)
    k = np.sqrt(np.abs(A2).sum() / max(A3[ok].sum(), 1e-30))
    s1 = s[:, 0] / k; s2 = s[:, 1] / k
    area = s1 * s2
    conf = s1 / np.maximum(s2, 1e-12)
    w = A3 / A3.sum()
    good = ok & np.isfinite(conf)
    def wq(x, q):
        xs = x[good]; ws = w[good]
        o = np.argsort(xs); cw = np.cumsum(ws[o])
        return float(xs[o][np.searchsorted(cw, q * cw[-1])]) if len(xs) else float('nan')
    return dict(
        flipped=int(((A2 < 0) & ok & (A3 > 1e-12 * A3.max())).sum()),
        conf_p95=wq(conf, 0.95), conf_p50=wq(conf, 0.5),
        area_p05=wq(area, 0.05), area_p95=wq(area, 0.95),
        area_ok=float(w[good & (area > 0.8) & (area < 1.25)].sum()),
        conf_ok=float(w[good & (conf < 1.5)].sum()),
        scale=float(k))


def fill_holes(V, F):
    """Cap every boundary loop except the longest one with a fan (for parameterisation only).
    Returns (V_filled, F_filled, n_original_vertices)."""
    try:
        loops = igl.boundary_loop_all(F.astype(np.int64))
    except Exception:
        return V, F, len(V)
    if len(loops) <= 1:
        return V, F, len(V)
    directed = set(zip(F[:, 0].tolist(), F[:, 1].tolist())) | set(zip(F[:, 1].tolist(), F[:, 2].tolist())) |         set(zip(F[:, 2].tolist(), F[:, 0].tolist()))
    per = [np.linalg.norm(V[np.asarray(l)] - V[np.roll(np.asarray(l), -1)], axis=1).sum() for l in loops]
    outer = int(np.argmax(per))
    Vn = [V]; Fn = [F]; nv = len(V)
    for i, l in enumerate(loops):
        if i == outer:
            continue
        l = np.asarray(l)
        if len(l) < 3:
            continue
        if (int(l[0]), int(l[1])) not in directed:
            l = l[::-1]
        Vn.append(V[l].mean(0)[None])
        Fn.append(np.c_[np.roll(l, -1), l, np.full(len(l), nv)])  # contains edge l[i+1] -> l[i]
        nv += 1
    return np.concatenate(Vn), np.concatenate(Fn), len(V)


def _signed2(u, F):
    a, b, c = u[F[:, 0]], u[F[:, 1]], u[F[:, 2]]
    return (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]


def _quality(V, F, uv):
    st = distortion_stats(V, F, uv)
    return (np.log(max(st['conf_p95'], 1.0)) + abs(np.log(max(st['area_p05'], 1e-3))) +
            abs(np.log(max(st['area_p95'], 1e-3))) + 5.0 * (st['flipped'] > 0))


def tutte_uniform(V, F, bnd):
    """Uniform-weight Tutte embedding (always injective for a disk with convex boundary)."""
    n = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    # boundary on circle with arc-length spacing
    bp = V[bnd]
    seg = np.linalg.norm(bp - np.roll(bp, -1, 0), axis=1)
    t = np.r_[0, np.cumsum(seg)[:-1]] / seg.sum() * 2 * np.pi
    R = seg.sum() / (2 * np.pi)
    uvb = np.c_[np.cos(t), np.sin(t)] * R
    deg = np.asarray(A.sum(1)).ravel()
    inner = np.setdiff1d(np.nonzero(deg > 0)[0], bnd)
    uv = np.zeros((n, 2)); uv[bnd] = uvb
    # vertices not reachable from the boundary (separate pieces) would make the system singular
    import scipy.sparse.csgraph as _cg
    _, cc = _cg.connected_components(A, directed=False)
    inner = inner[np.isin(cc[inner], np.unique(cc[bnd]))]
    if len(inner):
        Lii = L[inner][:, inner].tocsc(); Lib = L[inner][:, bnd]
        import scipy.sparse.linalg as sla
        solve = sla.factorized(Lii)
        rhs = -(Lib @ uvb)
        uv[inner, 0] = solve(rhs[:, 0]); uv[inner, 1] = solve(rhs[:, 1])
    return uv


def parameterize(V, Fl, slim_iters=30, log=None):
    """Chart -> UV.
    1. holes are temporarily capped (fans) so the initialisation sees one boundary
    2. LSCM; if it folds, uniform Tutte (guaranteed fold-free)
    3. SLIM (symmetric Dirichlet) on the real, non-degenerate faces only
    Degenerate faces are excluded from the energy; their isolated vertices follow neighbours."""
    nv0 = len(V)
    _, A3 = tri_frames(V, Fl)
    good = A3 > 1e-12 * max(A3.max(), 1e-30)
    Freal = Fl[good]
    # topology work (hole capping, boundary, Tutte) uses every face; energies skip degenerate ones
    Vf, Ff, _ = fill_holes(V, Fl)
    _, Af = tri_frames(Vf, Ff)
    Fe = Ff[Af > 1e-12 * max(Af.max(), 1e-30)]
    nv = len(Vf)
    used = np.zeros(nv, bool); used[Freal.ravel()] = True
    try:
        bnd = np.asarray(igl.boundary_loop(Ff.astype(np.int64)), np.int64)
    except Exception:
        bnd = np.array([], np.int64)
    uv = None; method = 'lscm'
    if len(bnd) >= 3:
        Bp = Vf[bnd]
        i0 = int(np.argmax(np.linalg.norm(Bp - Bp.mean(0), axis=1)))
        i1 = int(np.argmax(np.linalg.norm(Bp - Bp[i0], axis=1)))
        b = np.array([bnd[i0], bnd[i1]], np.int64)
        bc = np.array([[0.0, 0.0], [np.linalg.norm(Bp[i1] - Bp[i0]), 0.0]])
        ref = np.unique(Fe.ravel())
        try:
            if len(ref) == nv:
                uv, _ = igl.lscm(Vf, Fe.astype(np.int64), b, bc)
            elif len(ref) and np.isin(b, ref).all() and b[0] != b[1]:
                # a vertex used only by degenerate faces makes the LSCM system singular and libigl crashes (no
                # exception): solve on the referenced vertices; the rest follow their neighbours further down
                remap = -np.ones(nv, np.int64); remap[ref] = np.arange(len(ref))
                uvc, _ = igl.lscm(Vf[ref], remap[Fe].astype(np.int64), remap[b], bc)
                uv = np.zeros((nv, 2)); uv[ref] = uvc
            if uv is not None and not np.all(np.isfinite(uv[used])):
                uv = None
        except Exception:
            uv = None
    if uv is not None:
        cr = _signed2(uv, Freal)
        if (cr < 0).sum() > len(cr) / 2:
            uv[:, 0] *= -1; cr = -cr
        if (cr <= 0).any():
            uv = None
    if uv is None:
        method = 'tutte'
        if len(bnd) < 3:
            raise RuntimeError('chart is not a disk')
        uv = tutte_uniform(Vf, Ff, bnd)
        cr = _signed2(uv, Freal)
        if (cr < 0).sum() > len(cr) / 2:
            uv[:, 0] *= -1
        slim_iters = max(slim_iters, 120)
    nflip0 = int((_signed2(uv, Freal) <= 0).sum())
    if slim_iters > 0 and nflip0 == 0:
        uv = slim(Vf, Fe, uv, slim_iters)
    if method == 'lscm' and len(bnd) >= 3:
        sc = _quality(V, Freal, uv[:nv0])
        if sc > 0.8:  # poor result: compare with a Tutte start
            try:
                uv2 = tutte_uniform(Vf, Ff, bnd)
                cr = _signed2(uv2, Freal)
                if (cr < 0).sum() > len(cr) / 2:
                    uv2[:, 0] *= -1
                if (_signed2(uv2, Freal) > 0).all():
                    uv2 = slim(Vf, Fe, uv2, max(slim_iters, 120))
                    if _quality(V, Freal, uv2[:nv0]) < sc:
                        uv = uv2; method = 'tutte*'
            except Exception:
                pass
    uv = uv[:nv0].copy()
    miss = np.nonzero(~used[:nv0])[0]
    if len(miss):
        for _ in range(10):
            for vi in miss:
                fs = np.nonzero((Fl == vi).any(1))[0]
                nb = np.setdiff1d(np.unique(Fl[fs]), miss)
                if len(nb):
                    uv[vi] = uv[nb].mean(0)
    if log is not None:
        log.append(dict(method=method, init_flips=nflip0))
    return uv


def slim(V, F, uv, iters):
    ref = np.unique(F.ravel())
    if len(ref) < len(V):
        # vertices outside F (only on degenerate faces) make the system singular and libigl crashes on it:
        # solve on the referenced vertices, the others keep their position
        if not len(ref):
            return uv
        remap = -np.ones(len(V), np.int64); remap[ref] = np.arange(len(ref))
        out = np.array(uv, float, copy=True)
        out[ref] = slim(V[ref], remap[F], np.asarray(uv)[ref], iters)
        return out
    data = igl.slim_precompute(np.asfortranarray(V), np.asfortranarray(F.astype(np.int32)),
                               np.asfortranarray(uv), igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
                               np.zeros(0, np.int32), np.zeros((0, 2), order='F'), 0.0)
    out = igl.slim_solve(data, iters)
    out = np.asarray(out)[:, :2]
    if not np.all(np.isfinite(out)):
        return uv
    return out


# ---------------------------------------------------------------- orientation helpers

def orient_upright(V, uv, up, faces_area=None):
    """Rotate uv so that the 3D direction `up` points to +v. Returns rotated uv (centred)."""
    f = V @ np.asarray(up, float)
    A = np.c_[uv - uv.mean(0), np.ones(len(uv))]
    coef, *_ = np.linalg.lstsq(A, f - f.mean(), rcond=None)
    g = coef[:2]
    if np.linalg.norm(g) < 1e-12:
        return uv - uv.mean(0)
    ang = np.arctan2(g[1], g[0])
    rot = np.pi / 2 - ang
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])
    return (uv - uv.mean(0)) @ R.T


def uv_area(uv, F):
    a, b, c = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
    return 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])


# ---------------------------------------------------------------- UV outline smoothing
def _loop_smooth(P, sigma_edges, corner_deg):
    """Gaussian smoothing of a closed polyline by arc length; displacement fades out near real corners."""
    n = len(P)
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    h = max(float(np.median(seg)), 1e-12)
    s = sigma_edges * h
    L = float(seg.sum())
    cum = np.r_[0.0, np.cumsum(seg)[:-1]]
    K = int(min(n // 2 - 1, max(2, np.ceil(3.5 * s / h))))
    offs = np.arange(-K, K + 1)
    idx = (np.arange(n)[:, None] + offs[None, :]) % n
    d = cum[idx] - cum[:, None]
    d = (d + L / 2) % L - L / 2
    w = np.exp(-0.5 * (d / s) ** 2)
    Ps = (w[..., None] * P[idx]).sum(1) / w.sum(1)[:, None]
    # turning angle of the smoothed outline over +-3 sigma of arc length
    ext = np.r_[cum - L, cum, cum + L]
    Pe = np.r_[Ps, Ps, Ps]
    jp = np.searchsorted(ext, cum + 3 * s) ; jm = np.searchsorted(ext, cum - 3 * s) - 1
    jp = np.clip(jp, 0, 3 * n - 1); jm = np.clip(jm, 0, 3 * n - 1)
    a = Pe[jp] - Ps; b = Ps - Pe[jm]
    na = np.linalg.norm(a, axis=1) + 1e-30; nb = np.linalg.norm(b, axis=1) + 1e-30
    ang = np.degrees(np.arccos(np.clip((a * b).sum(1) / (na * nb), -1, 1)))
    corner = ang > corner_deg
    # keep only the local maxima of the turning angle as corner points
    cidx = np.nonzero(corner & (ang >= np.roll(ang, 1)) & (ang >= np.roll(ang, -1)))[0]
    fade = np.ones(n)
    if len(cidx):
        dc = np.abs(cum[:, None] - cum[cidx][None, :]); dc = np.minimum(dc, L - dc).min(1)
        fade = 1 - np.exp(-0.5 * (dc / s) ** 2)
    return (Ps - P) * fade[:, None], int(len(cidx))


def smooth_boundary(V, Fl, uv, sigma_edges=4.0, corner_deg=55.0, soft=100.0, iters=30, min_loop=8):
    """Smooth the staircase outline of a chart in UV space. 3D geometry and topology are untouched.
    Target outline = arc-length Gaussian smoothing of every boundary loop (real corners kept). The boundary is then
    pulled towards it with SLIM soft constraints starting from the valid map: the symmetric Dirichlet energy is an
    infinite barrier against folding, so the result never flips; distortion goes into the outer triangle rows."""
    nv = len(uv)
    info = dict(smoothed=False)
    try:
        loops = igl.boundary_loop_all(Fl.astype(np.int64))
    except Exception:
        return uv, info
    _, A3 = tri_frames(V, Fl)
    good = A3 > 1e-12 * max(A3.max(), 1e-30)
    Fg = Fl[good]
    tgt = uv.copy(); bmask = np.zeros(nv, bool); corners = 0
    for l in loops:
        l = np.asarray(l)
        bmask[l] = True
        if len(l) < min_loop or len(np.unique(l)) != len(l):
            continue
        dl, c = _loop_smooth(uv[l], sigma_edges, corner_deg)
        tgt[l] += dl; corners += c
    if np.abs(tgt - uv).max() == 0:
        return uv, info
    used = np.unique(Fg)
    remap = -np.ones(nv, np.int64); remap[used] = np.arange(len(used))
    Fc = remap[Fg]
    b = np.nonzero(bmask[used])[0]
    sign0 = np.sign(_signed2(uv, Fg).sum())
    try:
        data = igl.slim_precompute(np.asfortranarray(V[used]), np.asfortranarray(Fc.astype(np.int32)),
                                   np.asfortranarray(uv[used]), igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
                                   b.astype(np.int32), np.asfortranarray(tgt[used][b]), float(soft))
        out = np.asarray(igl.slim_solve(data, iters))[:, :2]
    except Exception:
        return uv, info
    if not np.all(np.isfinite(out)) or not (_signed2(out, Fc) * sign0 > 0).all():
        return uv, info
    uv2 = uv.copy(); uv2[used] = out
    # vertices only on degenerate faces follow their neighbours' displacement
    miss = np.setdiff1d(np.arange(nv), used)
    for vi in miss:
        fs = np.nonzero((Fl == vi).any(1))[0]
        nb = np.intersect1d(np.unique(Fl[fs]), used)
        if len(nb):
            uv2[vi] = uv[vi] + (uv2[nb] - uv[nb]).mean(0)
    dev = np.linalg.norm(uv2[bmask] - tgt[bmask], axis=1)
    info.update(smoothed=True, corners=corners, max_move=float(np.abs(uv2 - uv).max()),
                residual_mean=float(dev.mean()))
    return uv2, info
