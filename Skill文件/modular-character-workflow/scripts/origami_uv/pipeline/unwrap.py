"""Stage 3: semantic charts -> seams -> low-distortion flattening (per object).

Output npz per object:
  corner_uvv   (nf,3)  uv-vertex id per working corner (satellite corners point to their host uv-vertex)
  uvv_chart    (nuv,)  chart id per uv-vertex
  uvv_uv       (nuv,2) flattened coordinates in metres (UV area == 3D area), upright
  face_chart   (nf,)   chart id per face
  charts.json  per-chart metadata (region, area, distortion, topology, cut info)
"""
import sys, json, time
import numpy as np
import igl
from uvcore import (Mesh, Charts, topology, edge_graph, shortest_path, path_keys, parameterize, smooth_boundary,
                    distortion_stats, orient_upright, uv_area, _principal_axis)


def seam_cost(m, fs, region, E):
    n = m.normal[fs]
    c = 0.15 + 2.0 * E[fs] ** 1.2
    if region.get('seam') is not None:
        d = np.asarray(region['seam'])
        c = c * (1.0 + 3.0 * (1.0 - n @ d) / 2.0)
    return c


def loop_perimeter(V, loop):
    l = np.asarray(loop)
    return float(np.linalg.norm(V[l] - V[np.roll(l, -1)], axis=1).sum())


def plan_cuts(V, Fl, ids_w, nvw, cost, region, log):
    """Return list of global cut keys needed to make this chart a (holey) disk."""
    topo = topology(Fl, len(V))
    loops = [np.asarray(l) for l in topo['loops']]
    per = [loop_perimeter(V, l) for l in loops]
    maxper = max(per) if per else 0.0
    # a hole is "big" (an opening that needs a seam) if comparable to the main opening
    big = [l for l, p in zip(loops, per) if p >= 0.25 * maxper and p > 0.02]
    log.update(V=topo['V'], F=topo['F'], b=topo['b'], genus=int(topo['genus']), big_loops=len(big))
    g = edge_graph(V, Fl, cost)
    keys = []
    if topo['genus'] > 0:
        cuts = igl.cut_to_disk(Fl.astype(np.int64))
        bset = set()
        for l in loops:
            for a, b in zip(l, np.roll(l, -1)):
                bset.add((min(a, b), max(a, b)))
        for c in cuts:
            c = list(c)
            for a, b in zip(c[:-1], c[1:]):
                if (min(a, b), max(a, b)) not in bset:
                    keys.append(path_keys([a, b], ids_w, nvw)[0])
        log['cut'] = 'cut_to_disk'
        return keys
    if len(big) == 0:
        # closed (or only pin holes): slit between the two extreme cheap points along the up/closed axis
        vc = np.zeros(len(V)); cnt = np.zeros(len(V))
        np.add.at(vc, Fl.ravel(), np.repeat(cost, 3)); np.add.at(cnt, Fl.ravel(), 1)
        vc /= np.maximum(cnt, 1)
        ax = np.asarray(region.get('closed_axis') or region['up'], float)
        if abs((V - V.mean(0)) @ ax).max() < 1e-6:
            ax = _principal_axis(V)
        cheap = np.nonzero(vc <= np.quantile(vc, 0.35))[0]
        s = V[cheap] @ ax
        a = cheap[np.argmin(s)]; b = cheap[np.argmax(s)]
        p = shortest_path(g, [a], [b])
        if p is not None and len(p) > 1:
            keys += list(path_keys(p, ids_w, nvw))
        log['cut'] = 'slit'
        return keys
    if len(big) >= 2:
        big.sort(key=len, reverse=True)
        src = list(big[0]); remaining = big[1:]
        while remaining:
            tgt = np.concatenate(remaining)
            p = shortest_path(g, src, tgt)
            if p is None:
                break
            keys += list(path_keys(p, ids_w, nvw))
            end = p[-1]
            hit = [i for i, l in enumerate(remaining) if end in set(l.tolist())][0]
            src += list(remaining[hit]) + list(p)
            remaining.pop(hit)
        log['cut'] = f'connect {len(big)} openings'
    return keys


def relief_cut(V, Fl, ids_w, nvw, cost):
    """Cut from the boundary to the geodesically farthest interior point (opens cups / domes)."""
    topo = topology(Fl, len(V))
    if not topo['loops']:
        return []
    bnd = np.unique(np.concatenate([np.asarray(l) for l in topo['loops']]))
    g_len = edge_graph(V, Fl, np.ones(len(Fl)))
    import scipy.sparse.csgraph as cg
    d = cg.dijkstra(g_len, directed=False, indices=bnd, min_only=True)
    d[~np.isfinite(d)] = -1
    tgt = int(np.argmax(d))
    if d[tgt] <= 0:
        return []
    g = edge_graph(V, Fl, cost)
    p = shortest_path(g, bnd, [tgt])
    if p is None or len(p) < 2:
        return []
    return list(path_keys(p, ids_w, nvw))


ACCEPT = dict(conf95=2.0, area05=0.4, area95=2.5)


def boundary_overlaps(Fl, uv):
    """Exact global-injectivity test for a locally injective (flip free) map: the chart overlaps itself
    iff its boundary loops self-intersect or intersect each other."""
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    try:
        loops = igl.boundary_loop_all(Fl.astype(np.int64))
    except Exception:
        return False
    lines = [LineString(np.r_[uv[np.asarray(l)], uv[[l[0]]]]) for l in loops if len(l) >= 3]
    suspect = any(not ln.is_simple for ln in lines)
    if not suspect and len(lines) > 1:
        tree = STRtree(lines)
        for i, ln in enumerate(lines):
            for j in tree.query(ln):
                if j > i and ln.intersects(lines[j]):
                    suspect = True; break
            if suspect:
                break
    if not suspect:
        return False
    return overlap_fraction(Fl, uv) > 0.004


def overlap_fraction(Fl, uv):
    """(sum of triangle areas - area of their union) / sum, exact (shapely)."""
    from shapely import polygons
    from shapely.ops import unary_union
    t = uv[Fl]
    a = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]))
    keep = a > 1e-14 * max(a.max(), 1e-30)
    t = t[keep]
    if not len(t):
        return 0.0
    un = unary_union(list(polygons(np.concatenate([t, t[:, :1]], 1))))
    tot = a[keep].sum()
    return float(max(tot - un.area, 0) / tot)


def fiedler_split(V, Fl, ids_w, nvw):
    """Balanced bisection of a chart along its Fiedler vector; returns cut keys between the halves."""
    import scipy.sparse as sp_
    import scipy.sparse.linalg as sla
    nf = len(Fl)
    e = np.concatenate([np.c_[Fl[:, [0, 1]], np.arange(nf)], np.c_[Fl[:, [1, 2]], np.arange(nf)], np.c_[Fl[:, [2, 0]], np.arange(nf)]])
    a = np.minimum(e[:, 0], e[:, 1]); b = np.maximum(e[:, 0], e[:, 1])
    k = a * len(V) + b
    order = np.argsort(k); ks = k[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    f1 = e[order[same], 2]; f2 = e[order[same + 1], 2]
    L = np.linalg.norm(V[a[order[same]]] - V[b[order[same]]], axis=1) + 1e-9
    A = sp_.coo_matrix((np.r_[L, L], (np.r_[f1, f2], np.r_[f2, f1])), shape=(nf, nf)).tocsr()
    D = sp_.diags(np.asarray(A.sum(1)).ravel())
    Lap = (D - A).tocsc()
    try:
        vals, vecs = sla.eigsh(Lap + sp_.identity(nf) * 1e-8, k=2, sigma=0, which='LM')
        fv = vecs[:, np.argsort(vals)[1]]
    except Exception:
        c = V[Fl].mean(1); c = c - c.mean(0)
        w, v = np.linalg.eigh(c.T @ c); fv = c @ v[:, -1]
    # ratio cut along the Fiedler order: shortest cut relative to the smaller side
    area = np.linalg.norm(np.cross(V[Fl[:, 1]] - V[Fl[:, 0]], V[Fl[:, 2]] - V[Fl[:, 0]]), axis=1) * 0.5
    best = None
    for q in np.linspace(0.15, 0.85, 15):
        th = np.quantile(fv, q)
        sd = fv > th
        cl = L[sd[f1] != sd[f2]].sum()
        aa = area[sd].sum(); bb = area[~sd].sum()
        sc = cl / max(min(aa, bb), 1e-12) ** 0.5
        if best is None or sc < best[0]:
            best = (sc, sd)
    side = best[1]
    cut = same[side[f1] != side[f2]]
    ca = a[order[cut]]; cb = b[order[cut]]
    wa = ids_w[ca]; wb = ids_w[cb]
    return list(np.minimum(wa, wb) * nvw + np.maximum(wa, wb))


def unwrap_object(m, labels, regions, E, slim_iters=40, verbose=True, max_relief=3, accept=ACCEPT, smooth_outline=True):
    t0 = time.time()
    lab = labels.copy()
    cut_keys = np.zeros(0, np.int64)
    corner_uv = np.full((m.nf, 3, 2), np.nan)
    done = {}; relief = {}; plans = {}
    split_depth = np.zeros(m.nf, np.int32)
    for rnd in range(16):
        ch = Charts(m, lab, cut_keys)
        new = []
        for ci in range(ch.n):
            fs, V, Fl, ids = ch.local(ci)
            l = int(lab[fs[0]])
            if l < 0:
                continue
            ident = (l, int(fs.min()))
            if ident in done:
                continue
            reg = regions[l]
            cost = seam_cost(m, fs, reg, E)
            log = plans.setdefault(ident, {})
            keys = plan_cuts(V, Fl, ch.uvv_w[ids], m.nvw, cost, reg, log)
            keys = np.setdiff1d(np.asarray(keys, np.int64), cut_keys)
            if len(keys) and plans.get(('n', ident), 0) < 4:
                plans[('n', ident)] = plans.get(('n', ident), 0) + 1
                new += list(keys); continue
            plog = []
            t = time.time()
            try:
                uv = parameterize(V, Fl, slim_iters, plog)
                method = plog[0]['method']
            except Exception as e:
                n = m.normal[fs].mean(0); n /= np.linalg.norm(n) + 1e-12
                a_ = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0]); a_ /= np.linalg.norm(a_)
                uv = np.c_[V @ a_, V @ np.cross(n, a_)]
                method = 'planar'
            st = distortion_stats(V, Fl, uv)
            # global overlap of the chart with itself -> bisect and flatten the halves separately
            if boundary_overlaps(Fl, uv) and split_depth[fs].max() < 3 and len(fs) > 60:
                keys = [k for k in fiedler_split(V, Fl, ch.uvv_w[ids], m.nvw) if k not in set(cut_keys.tolist())]
                if keys:
                    new += keys
                    split_depth[fs] += 1
                    if verbose:
                        print(f"   overlap -> split {reg['name']} (F={len(fs)})", flush=True)
                    continue
            bad = (st['conf_p95'] > accept['conf95'] or st['area_p05'] < accept['area05'] or
                   st['area_p95'] > accept['area95'] or st['flipped'] > 0)
            nrel = relief.get(ident, 0)
            if bad and nrel < max_relief and len(fs) > 150:
                keys = relief_cut(V, Fl, ch.uvv_w[ids], m.nvw, cost)
                keys = [k for k in keys if k not in set(cut_keys.tolist())]
                if keys:
                    relief[ident] = nrel + 1
                    new += keys
                    if verbose:
                        print(f"   relief cut #{nrel+1} {reg['name']} (conf95={st['conf_p95']:.2f} area[{st['area_p05']:.2f},{st['area_p95']:.2f}])", flush=True)
                    continue
            uv = uv / st['scale']
            sm_info = {}
            if smooth_outline:
                uv_s, sm_info = smooth_boundary(V, Fl, uv)
                if sm_info.get('smoothed') and not boundary_overlaps(Fl, uv_s):
                    uv = uv_s
                    st = distortion_stats(V, Fl, uv); st['scale'] = 1.0
                else:
                    sm_info['smoothed'] = False
            uv = orient_upright(V, uv, reg['up'])
            if uv_area(uv, Fl).sum() < 0:
                uv[:, 0] *= -1
            corner_uv[fs] = uv[Fl]
            topo = topology(Fl, len(V))
            info = dict(region=l, name=reg['name'], faces=len(fs), area3d=float(m.area[fs].sum()),
                        loops=topo['b'], genus=int(topo['genus']), method=method, relief_cuts=nrel,
                        cut=log.get('cut', ''), t=round(time.time() - t, 2), outline_smoothed=bool(sm_info.get('smoothed')),
                        **{k: v for k, v in st.items() if k != 'scale'})
            done[ident] = info
            if verbose and (len(fs) > 300 or bad):
                d = info
                print(f"   {reg['name']:<26} F={len(fs):6d} b={topo['b']} {method:<6} rel={nrel} "
                      f"flip={d['flipped']} conf95={d['conf_p95']:.2f} area[{d['area_p05']:.2f},{d['area_p95']:.2f}] "
                      f"ok={d['area_ok']*100:.0f}%/{d['conf_ok']*100:.0f}% {d['t']}s{'  <-- BAD' if bad else ''}", flush=True)
        new = np.setdiff1d(np.asarray(new, np.int64), cut_keys)
        if verbose:
            print(f'  round {rnd}: charts={ch.n} new cut edges={len(new)}', flush=True)
        if len(new) == 0:
            break
        cut_keys = np.r_[cut_keys, new]
    ch = Charts(m, lab, cut_keys)
    nuv = ch.uvv.max() + 1
    uvv_uv = np.zeros((nuv, 2)); uvv_chart = np.full(nuv, -1, np.int64)
    flat = corner_uv.reshape(-1, 2)
    ok = np.isfinite(flat[:, 0])
    uvv_uv[ch.uvv[ok]] = flat[ok]
    infos = []
    for ci in range(ch.n):
        fs = ch.faces[ci]
        l = int(lab[fs[0]])
        ident = (l, int(fs.min()))
        uvv_chart[ch.uvv.reshape(-1, 3)[fs].ravel()] = ci
        if l < 0:
            infos.append(dict(chart=ci, region=-1, faces=len(fs)))
        else:
            infos.append(dict(chart=ci, **done[ident]))
    # consistency: all corners of one uv-vertex share the same uv
    dev = np.abs(uvv_uv[ch.uvv[ok]] - flat[ok]).max() if ok.any() else 0
    assert dev < 1e-9, f'uv-vertex inconsistency {dev}'
    corner_uvv = ch.uvv.reshape(-1, 3).copy()
    face_chart = ch.chart.copy()
    sat = np.nonzero(lab < 0)[0]
    unresolved = 0
    sat_uv = None
    if len(sat):
        # project each satellite corner onto the nearest host face and reuse its UV parameterisation
        from scipy.spatial import cKDTree
        host = np.nonzero(lab >= 0)[0]
        tree = cKDTree(m.cent[host])
        _, nn = tree.query(m.cent[sat], k=6)
        sat_uv = np.zeros((len(sat), 3, 2))
        for i, f in enumerate(sat):
            best = None
            for h in host[np.atleast_1d(nn[i])]:
                hv = m.co[m.F[h]]
                if best is None:
                    best = h
                # prefer a host face sharing a vertex
                if np.intersect1d(m.Fw[h], m.Fw[f]).size:
                    best = h; break
            h = best
            A, B, C = m.co[m.F[h]]
            e1 = B - A; e2 = C - A
            G = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]])
            for k in range(3):
                d = m.co[m.F[f, k]] - A
                try:
                    st = np.linalg.solve(G, [d @ e1, d @ e2])
                except np.linalg.LinAlgError:
                    st = np.array([1 / 3, 1 / 3])
                bary = np.clip(np.array([1 - st.sum(), st[0], st[1]]), 0, 1); bary /= bary.sum()
                sat_uv[i, k] = bary @ corner_uv[h]
            face_chart[f] = ch.chart[h]
    corner_uv_all = corner_uv.copy()
    if len(sat):
        corner_uv_all[sat] = sat_uv
    if verbose:
        print(f'  {m.name}: {sum(1 for i in infos if i["region"] >= 0)} charts, cut edges {len(cut_keys)}, '
              f'satellites {len(sat)} (unresolved {unresolved}), {time.time()-t0:.1f}s', flush=True)
    return dict(corner_uvv=corner_uvv, uvv_chart=uvv_chart, uvv_uv=uvv_uv, face_chart=face_chart,
                corner_uv=corner_uv_all, cut_keys=cut_keys, labels=lab, satellites=sat), infos
