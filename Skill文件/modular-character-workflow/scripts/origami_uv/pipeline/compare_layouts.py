"""Measure UV layouts with the same yardstick: the source UV (UV_orig, e.g. the AI generator's), optionally an earlier
layout stored in the input blend, and this version. Islands = connected UV components; coverage = rasterised union at 4K.
Usage: python compare_layouts.py <work> <out.json> <object> [<object> ...]"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.csgraph as cg


def islands(Fw, uv):
    """Fw (nf,3) welded vertex ids, uv (nf,3,2) corner uvs."""
    q = np.round(uv.reshape(-1, 2) * (1 << 20)).astype(np.int64)
    key = np.c_[Fw.ravel(), q]
    _, uid = np.unique(key, axis=0, return_inverse=True)
    uid = uid.reshape(-1, 3)
    n = uid.max() + 1
    r = np.r_[uid[:, 0], uid[:, 1]]; c = np.r_[uid[:, 1], uid[:, 2]]
    g = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    k, lab = cg.connected_components(g, directed=False)
    return int(k), lab[uid[:, 0]]


def outline_turning(Fw, uv, size=4096, min_faces=0, lab=None, area3d=None):
    """Jaggedness of island outlines: accumulated absolute turning angle (degrees) per 100 px of outline at `size`.
    Staircase outlines turn ~90 deg every few pixels; smooth outlines turn slowly."""
    q = np.round(uv.reshape(-1, 2) * (1 << 20)).astype(np.int64)
    _, uid = np.unique(np.c_[Fw.ravel(), q], axis=0, return_inverse=True)
    uid = uid.reshape(-1, 3)
    P = np.zeros((uid.max() + 1, 2)); P[uid.ravel()] = uv.reshape(-1, 2) * size
    a = uid.ravel(); b = uid[:, [1, 2, 0]].ravel()
    key = np.minimum(a, b) * (uid.max() + 1) + np.maximum(a, b)
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bd = cnt[inv] == 1
    nxt = {}
    for x, y in zip(a[bd].tolist(), b[bd].tolist()):
        nxt.setdefault(x, []).append(y)
    seen = set(); turn = 0.0; length = 0.0
    for s0 in list(nxt):
        for t0 in nxt[s0]:
            if (s0, t0) in seen:
                continue
            loop = [s0]; cur, nx = s0, t0
            while (cur, nx) not in seen:
                seen.add((cur, nx)); loop.append(nx)
                cands = [c for c in nxt.get(nx, []) if (nx, c) not in seen]
                if not cands:
                    break
                cur, nx = nx, cands[0]
            pts = P[loop]
            d = np.diff(pts, axis=0)
            L = np.linalg.norm(d, axis=1); keep = L > 1e-6
            d = d[keep]
            if len(d) < 3:
                continue
            ang = np.arctan2(d[:, 1], d[:, 0])
            da = np.diff(np.r_[ang, ang[:1]])
            da = (da + np.pi) % (2 * np.pi) - np.pi
            turn += np.degrees(np.abs(da)).sum(); length += L[keep].sum()
    return round(float(turn / max(length, 1e-9) * 100), 1)


def coverage(uv, size=4096):
    m = np.zeros((size, size), np.uint8)
    t = uv.copy() % 1.0 if (uv.min() < 0 or uv.max() > 1) else uv
    pts = np.round(np.stack([t[..., 0] * size - 0.5, (1 - t[..., 1]) * size - 0.5], -1) * 16).astype(np.int32)
    cv2.fillPoly(m, list(pts), 1, cv2.LINE_8, 4)
    return float(m.mean())


def tiny_islands(lab, area3d, frac=0.001):
    a = np.bincount(lab, area3d)
    return int((a < frac * a.sum()).sum())


def run(work, out, objects, previous_uv=None):
    """objects: meshes to measure; previous_uv: name of an earlier layout stored in the input blend (optional).
    Row keys: 'tripo' = source UV (UV_orig, kept for compatibility), 'previous', 'origami' = new layout."""
    sys.path.insert(0, str(Path(__file__).parent))
    from uvcore import Mesh
    rows = []
    for o in objects:
        m = Mesh(Path(work) / 'inspect' / f'{o}.npz')
        d = np.load(Path(work) / 'inspect' / f'{o}.npz')
        layouts = [('tripo', m.uv_orig)]
        if previous_uv and f'uv_{previous_uv}' in d.files:
            prev = d[f'uv_{previous_uv}'].reshape(-1, 3, 2)
            layouts.append(('previous', np.take_along_axis(prev, m.perm[:, :, None], 1)))
        new = pickle.load(open(Path(work) / 'pack' / f'{o}.pkl', 'rb'))['final']
        layouts.append(('origami', new))
        row = dict(object=o)
        for name, uv in layouts:
            k, lab = islands(m.Fw, uv)
            row[name] = dict(islands=k, tiny_islands=tiny_islands(lab, m.area), coverage=round(coverage(uv), 4),
                             outline_turn_deg_per_100px=outline_turning(m.Fw, uv))
        rows.append(row)
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    return rows


if __name__ == '__main__':
    for r in run(sys.argv[1], sys.argv[2], sys.argv[3:]):
        print(r)
