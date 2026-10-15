"""Per-face exposure (fraction of outside view directions from which the face's front side is visible).

Uses all character meshes together so hair occludes the scalp, shoes occlude feet, etc.
Deterministic: fixed Fibonacci directions, painter's algorithm ID buffers.
Usage: python exposure.py <inspect_dir> <out_npz> [n_dirs] [res]   (objects: every mesh listed in scene.json)
"""
import json, sys, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('\\', 1)[0] if '\\' in __file__ else __file__.rsplit('/', 1)[0])
from uvcore import Mesh



def fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def main(inspect_dir, out, n_dirs=64, res=2048, objects=None):
    """objects: the character meshes (all of them, so parts occlude each other)."""
    OBJECTS = list(objects) if objects else list(json.load(open(f'{inspect_dir}/scene.json', encoding='utf-8'))['objects'])
    meshes = [Mesh(f'{inspect_dir}/{o}.npz') for o in OBJECTS]
    co = []; F = []; N = []; off = 0; owner = []
    for k, m in enumerate(meshes):
        co.append(m.co); F.append(m.F + off); N.append(m.normal); off += len(m.co); owner.append(np.full(m.nf, k))
    co = np.concatenate(co); F = np.concatenate(F); N = np.concatenate(N); owner = np.concatenate(owner)
    nf = len(F)
    vis = np.zeros(nf, np.float32)
    D = fib_dirs(n_dirs)
    lo = co.min(0); hi = co.max(0); ctr = (lo + hi) / 2; R = np.linalg.norm(hi - lo) / 2
    for d in D:
        fwd = -d  # camera at +d looking toward -d
        up0 = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
        right = np.cross(fwd, up0); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        P = co - ctr
        x = (P @ right) / R * (res * 0.49) + res / 2
        y = res / 2 - (P @ up) / R * (res * 0.49)
        z = P @ fwd
        depth = z[F].mean(1)
        facing = (N @ d) > 0
        buf = np.full((res, res), -1, np.int32)
        pts = np.round(np.stack([x, y], 1)[F] * 16).astype(np.int32)
        for i in np.argsort(-depth):
            cv2.fillConvexPoly(buf, pts[i], int(i), lineType=cv2.LINE_8, shift=4)
        ids = np.unique(buf); ids = ids[ids >= 0]
        seen = np.zeros(nf, bool); seen[ids] = True
        vis += (seen & facing).astype(np.float32)
    vis /= n_dirs
    # normalise: a convex outward surface is visible from ~half the directions
    exposure = np.clip(vis * 2.0, 0, 1)
    res_d = {}
    for k, o in enumerate(OBJECTS):
        res_d[o] = exposure[owner == k]
    np.savez_compressed(out, **res_d)
    for o in OBJECTS:
        e = res_d[o]
        print(o, 'exposure quantiles', np.quantile(e, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3))


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 64, int(a[4]) if len(a) > 4 else 2048)
