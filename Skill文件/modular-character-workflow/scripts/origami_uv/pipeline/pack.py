"""Stage 4: semantic, shape-aware packing.

Charts are scaled by their region priority, then placed largest-first on a raster occupancy
grid. Each packing group (torso, sleeves, eyes ...) owns a zone of the atlas computed from a
treemap in a fixed semantic order; a chart is put at the free position closest to its group's zone
(ties: top-left first). The global texel density is the largest one for which everything fits.
"""
import numpy as np
import cv2
from scipy.signal import fftconvolve


def tri_mask(tris, R, shift=4):
    """Rasterise triangles (n,3,2) given in grid units (x right, y down)."""
    lo = np.floor(tris.reshape(-1, 2).min(0)).astype(int)
    hi = np.ceil(tris.reshape(-1, 2).max(0)).astype(int)
    w, h = hi - lo + 1
    m = np.zeros((h + 2, w + 2), np.uint8)
    pts = np.round((tris - lo + 1) * (1 << shift)).astype(np.int32)
    cv2.fillPoly(m, list(pts), 1, lineType=cv2.LINE_8, shift=shift)
    for t in pts:  # thin slivers: draw outlines too
        cv2.polylines(m, [t], True, 1, 1, cv2.LINE_8, shift)
    return m, lo - 1


def treemap(groups, weights, x0=0.0, y0=0.0, w=1.0, h=1.0):
    """Row-based treemap in reading order. Returns {group: (x0,y0,x1,y1)} in unit coords (y down)."""
    tot = sum(weights)
    rects = {}
    items = list(zip(groups, [wt / tot * w * h for wt in weights]))
    y = y0
    i = 0
    while i < len(items):
        # grow a row while it improves the worst aspect ratio
        best = None
        for j in range(i + 1, len(items) + 1):
            row = items[i:j]
            ra = sum(a for _, a in row)
            rh = ra / w
            worst = max(max((a / rh) / rh, rh / (a / rh)) for _, a in row)
            if best is None or worst <= best[0]:
                best = (worst, j)
            else:
                break
        j = best[1]
        row = items[i:j]
        ra = sum(a for _, a in row); rh = ra / w
        if j == len(items):
            rh = y0 + h - y
        x = x0
        for g, a in row:
            ww = a / ra * w
            rects[g] = (x, y, x + ww, y + rh)
            x += ww
        y += rh
        i = j
    return rects


class Island:
    def __init__(self, cid, tris_m, prio, group, rotate, name):
        self.cid = cid
        self.tris = tris_m * prio  # metres scaled by priority
        self.area = float(np.abs(0.5 * np.cross(self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0])).sum())
        self.group = group; self.rotate = rotate; self.name = name; self.prio = prio


def rot_tris(tris, k):
    """rotate by k*90 degrees (counter-clockwise in uv space)."""
    t = tris.copy()
    for _ in range(k % 4):
        t = np.stack([-t[..., 1], t[..., 0]], -1)
    return t


def try_pack(islands, scale, R, pad, zones, group_rank, order):
    occ = np.zeros((R, R), np.float32)
    occ[:pad, :] = 1; occ[-pad:, :] = 1; occ[:, :pad] = 1; occ[:, -pad:] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    placed = {}
    for idx in order:
        isl = islands[idx]
        zx0, zy0, zx1, zy1 = [v * R for v in zones[isl.group]]
        best = None
        for k in ((0, 1) if isl.rotate else (0,)):
            t = rot_tris(isl.tris * scale, k)
            # uv (v up) -> grid (y down): flip v
            g = np.stack([t[..., 0] * R, -t[..., 1] * R], -1)
            core, off = tri_mask(g, R)
            core = cv2.dilate(core, np.ones((3, 3), np.uint8))  # conservative
            mask = cv2.dilate(np.pad(core, pad), kernel)
            off = off - pad
            h, w = mask.shape
            if h >= R or w >= R:
                continue
            coll = fftconvolve(occ, mask[::-1, ::-1].astype(np.float32), mode='valid')
            free = coll < 0.5
            if not free.any():
                continue
            ys, xs = np.nonzero(free)
            cx = xs + w / 2; cy = ys + h / 2
            dx = np.maximum(np.maximum(zx0 - cx, 0), cx - zx1)
            dy = np.maximum(np.maximum(zy0 - cy, 0), cy - zy1)
            dist = np.hypot(dx, dy)
            wz = 4.0 if isl.area * scale * scale > 0.004 else 1.5
            cost = dist * wz + (ys - zy0) * 1.0 + (xs - zx0) * 0.35
            i = int(np.argmin(cost))
            c = cost[i] + (5.0 if k else 0.0)
            if best is None or c < best[0]:
                best = (c, ys[i], xs[i], k, mask, off)
        if best is None:
            return None
        _, y, x, k, mask, off = best
        occ[y:y + mask.shape[0], x:x + mask.shape[1]] += mask
        placed[isl.cid] = (y, x, k, off)
    return placed, occ


def pack_atlas(islands, group_order, R=2048, pad=4, target=0.70, steps=7, log=print):
    """Find the largest uniform scale that fits; return per-chart transforms (scale, rot, du, dv)."""
    groups = [g for g in group_order if any(i.group == g for i in islands)]
    groups += sorted({i.group for i in islands} - set(groups))
    garea = [sum(i.area for i in islands if i.group == g) for g in groups]
    zones = treemap(groups, garea)
    rank = {g: n for n, g in enumerate(groups)}
    # placement order: largest first; mirrored L/R pairs stay consecutive (character's right first,
    # i.e. on the left of the sheet as in a front view)
    import re
    def base(n):
        return re.sub(r'_(L|R)(?=_|$)', '_X', islands[n].name)
    seen = {}
    for n in sorted(range(len(islands)), key=lambda n: -islands[n].area):
        b = base(n); seen.setdefault(b, []).append(n)
    keyed = {}
    for b, ns in seen.items():
        # k-th largest chart of region pair b
        per_side = {}
        for n in ns:
            sd = 0 if re.search(r'_R(?=_|$)', islands[n].name) else 1
            per_side.setdefault(sd, []).append(n)
        for sd, lst in per_side.items():
            for k, n in enumerate(lst):
                keyed[n] = (b, k, sd)
    pair_area = {}
    for n, (b, k, sd) in keyed.items():
        pair_area[(b, k)] = max(pair_area.get((b, k), 0), islands[n].area)
    order = sorted(range(len(islands)), key=lambda n: (-pair_area[keyed[n][:2]], keyed[n][0], keyed[n][1], keyed[n][2]))
    tot = sum(garea)

    def search(R_, pad_, lo, hi, steps_, first=None):
        best = None
        if first is not None:
            res = try_pack(islands, first, R_, pad_, zones, rank, order)
            if res is None:
                hi = first
            else:
                best = (first, res); lo = first
            log(f'    [{R_}] scale {first:.4f} -> {"fit" if res else "overflow"} (util {tot*first*first*100:.1f}%)')
        for _ in range(steps_):
            mid = (lo + hi) / 2
            res = try_pack(islands, mid, R_, pad_, zones, rank, order)
            if res is None:
                hi = mid
            else:
                best = (mid, res); lo = mid
            log(f'    [{R_}] scale {mid:.4f} -> {"fit" if res else "overflow"} (util {tot*mid*mid*100:.1f}%)')
        return best

    # coarse search on a half-resolution grid, then refine on the final grid
    Rc = R // 2; padc = max(1, pad // 2)
    best = search(Rc, padc, np.sqrt(0.30 / tot), np.sqrt(0.92 / tot), steps, first=np.sqrt(target / tot))
    if best is None:
        raise RuntimeError('packing failed even at minimum scale')
    s0 = best[0]
    best = search(R, pad, s0 * 0.97, s0 * 1.06, 3, first=s0)
    while best is None:
        s0 *= 0.97
        best = search(R, pad, s0 * 0.97, s0, 0, first=s0)
    s, (placed, occ) = best
    xf = {}
    for isl in islands:
        y, x, k, off = placed[isl.cid]
        xf[isl.cid] = dict(scale=s * isl.prio, rot=k, gx=x - off[0], gy=y - off[1], R=R)
    return xf, zones, s, tot * s * s


def apply_transform(tris_m, xf):
    """tris in metres (chart local, upright) -> atlas uv [0,1]."""
    t = rot_tris(tris_m * xf['scale'], xf['rot'])
    R = xf['R']
    u = t[..., 0] + xf['gx'] / R
    v = t[..., 1] - xf['gy'] / R  # grid y down == -v
    return np.stack([u, 1.0 + v], -1)


# ============================================================================ two-level (block) packing
# Every packing group (e.g. "sleeves", "eye_L") is first packed into its own compact block, then the
# blocks are packed into the atlas. A region group therefore always stays one contiguous patch of the
# sheet, which is what makes the layout readable for texture artists.

def _island_mask(isl, scale, k, R, pad, kernel):
    t = rot_tris(isl.tris * scale, k)
    g = np.stack([t[..., 0] * R, -t[..., 1] * R], -1)
    core, off = tri_mask(g, R)
    core = cv2.dilate(core, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(np.pad(core, pad), kernel)
    return mask, off - pad


def _place(occ, mask, cost_fn, contact=0.0):
    """Best free position. cost_fn(ys, xs, h, w) is minimised; with contact>0 the number of mask
    border cells touching occupied cells / the sheet border is rewarded (maximal-contact heuristic)."""
    h, w = mask.shape
    H, W = occ.shape
    if h > H or w > W:
        return None
    occb = (occ > 0).astype(np.float32)
    coll = fftconvolve(occb, mask[::-1, ::-1].astype(np.float32), mode='valid')
    free = coll < 0.5
    if not free.any():
        return None
    ys, xs = np.nonzero(free)
    c = cost_fn(ys, xs, h, w)
    if contact > 0:
        ring = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32) - (mask > 0)).clip(0, 1)
        ringp = np.pad(ring, 1)
        padded = np.pad(occb, 1, constant_values=1.0)
        cont = fftconvolve(padded, ringp[::-1, ::-1], mode='valid')
        c = c - contact * cont[ys, xs]
    i = int(np.argmin(c))
    return ys[i], xs[i]


def pack_group(isls, order, scale, R, pad, width):
    """Strip-pack the islands of one group into a canvas of fixed width (cells). Returns block."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    masks = {}
    for i in order:
        for k in ((0, 1) if isls[i].rotate else (0,)):
            masks[(i, k)] = _island_mask(isls[i], scale, k, R, pad, kernel)
    width = max(width, max(m_[0].shape[1] for m_ in masks.values()))
    H = sum(max(masks[(i, 0)][0].shape) for i in order) + 4
    occ = np.zeros((H, width), np.float32)
    placed = {}
    for i in order:
        best = None
        for k in ((0, 1) if isls[i].rotate else (0,)):
            mask, off = masks[(i, k)]
            pos = _place(occ, mask, lambda ys, xs, h, w: (ys + h) * 1.0 + xs * 0.25, contact=0.6)
            if pos is None:
                continue
            c = (pos[0] + mask.shape[0]) + pos[1] * 0.25 + (3 if k else 0)
            if best is None or c < best[0]:
                best = (c, pos, k, mask, off)
        if best is None:
            return None
        _, (y, x), k, mask, off = best
        occ[y:y + mask.shape[0], x:x + mask.shape[1]] += mask
        placed[i] = (y, x, k, off)
    used = np.nonzero(occ.any(1))[0]
    h = used.max() + 1
    usedx = np.nonzero(occ.any(0))[0]
    w = usedx.max() + 1
    return dict(mask=(occ[:h, :w] > 0).astype(np.float32), placed=placed)


def _pair_order(isls):
    import re
    def base(n):
        return re.sub(r'_(L|R)(?=_|$)', '_X', isls[n].name)
    groups = {}
    for n in sorted(range(len(isls)), key=lambda n: -isls[n].area):
        groups.setdefault(base(n), []).append(n)
    keyed = {}
    for b, ns in groups.items():
        per = {}
        for n in ns:
            sd = 0 if re.search(r'_R(?=_|$)', isls[n].name) else 1
            per.setdefault(sd, []).append(n)
        for sd, lst in per.items():
            for k, n in enumerate(lst):
                keyed[n] = (b, k, sd)
    pa = {}
    for n, (b, k, sd) in keyed.items():
        pa[(b, k)] = max(pa.get((b, k), 0), isls[n].area)
    return sorted(range(len(isls)), key=lambda n: (-pa[keyed[n][:2]], keyed[n][0], keyed[n][1], keyed[n][2]))


def try_pack_blocks(islands, groups, scale, R, pad, log=None):
    variants = {}
    for g in groups:
        idx = [i for i, isl in enumerate(islands) if isl.group == g]
        sub = [islands[i] for i in idx]
        order = _pair_order(sub)
        area_cells = sum(isl.area for isl in sub) * (scale * R) ** 2
        vs = []
        seen = set()
        for aspect in (0.35, 0.6, 1.0, 1.6, 2.6, 4.0):
            w = int(np.sqrt(area_cells * aspect / 0.65)) + 2 * pad + 2
            if w > R:
                continue
            bl = pack_group(sub, order, scale, R, pad, w)
            if bl is None:
                continue
            h, w2 = bl['mask'].shape
            if h > R or w2 > R or (h, w2) in seen:
                continue
            seen.add((h, w2))
            bl['group'] = g; bl['idx'] = idx
            vs.append(bl)
        if not vs:
            return None
        variants[g] = vs
    occ = np.zeros((R, R), np.float32)
    occ[:pad, :] = 1; occ[-pad:, :] = 1; occ[:, :pad] = 1; occ[:, -pad:] = 1
    rank = {g: n for n, g in enumerate(groups)}
    order_g = sorted(groups, key=lambda g: (-max(v['mask'].sum() for v in variants[g]), rank[g]))
    placed = {}; blocks = []
    for g in order_g:
        best = None
        for bl in variants[g]:
            pos = _place(occ, bl['mask'], lambda ys, xs, h, w: (ys + h) * 1.0 + xs * 0.3, contact=0.5)
            if pos is None:
                continue
            c = pos[0] + bl['mask'].shape[0] + pos[1] * 0.3
            if best is None or c < best[0]:
                best = (c, pos, bl)
        if best is None:
            return None
        _, (y, x), bl = best
        occ[y:y + bl['mask'].shape[0], x:x + bl['mask'].shape[1]] += bl['mask']
        bl['pos'] = (y, x)
        blocks.append(bl)
        for i_local, (yy, xx, k, off) in bl['placed'].items():
            placed[islands[bl['idx'][i_local]].cid] = (y + yy, x + xx, k, off)
    return placed, occ, blocks


def pack_atlas_blocks(islands, group_order, R=2048, pad=4, target=0.68, steps=7, log=print):
    groups = [g for g in group_order if any(i.group == g for i in islands)]
    groups += sorted({i.group for i in islands} - set(groups))
    tot = sum(i.area for i in islands)

    def search(R_, pad_, lo, hi, n, first):
        best = None
        for it in range(n + 1):
            s = first if it == 0 else (lo + hi) / 2
            res = try_pack_blocks(islands, groups, s, R_, pad_)
            if res is None:
                hi = s
            else:
                best = (s, res); lo = s
            log(f'    [{R_}] scale {s:.4f} -> {"fit" if res else "overflow"} (util {tot*s*s*100:.1f}%)')
        return best
    best = search(R // 2, max(1, pad // 2), np.sqrt(0.25 / tot), np.sqrt(0.92 / tot), steps, np.sqrt(target / tot))
    if best is None:
        raise RuntimeError('block packing failed')
    s0 = best[0]
    best = search(R, pad, s0 * 0.97, s0 * 1.05, 3, s0)
    while best is None:
        s0 *= 0.97
        best = search(R, pad, s0, s0, 0, s0)
    s, (placed, occ, blocks) = best
    xf = {}
    for isl in islands:
        y, x, k, off = placed[isl.cid]
        xf[isl.cid] = dict(scale=s * isl.prio, rot=k, gx=x - off[0], gy=y - off[1], R=R)
    zones = {}
    for b in blocks:
        y, x = b['pos']; h, w = b['mask'].shape
        zones[b['group']] = (x / R, y / R, (x + w) / R, (y + h) / R)
    return xf, zones, s, tot * s * s


# ============================================================================ affinity packing
# Islands are placed one by one (group after group, biggest first inside a group). A position is scored by
#   + contact with islands of the same group (strong)   -> groups stay one connected patch
#   + contact with anything / the sheet border (weak)   -> tight packing
#   - distance to the centre of the already placed part of the group
#   - height of the filled area (top-left reading order)

def try_pack_affinity(islands, groups, scale, R, pad, order, P=None):
    P = P or {}
    w_any = P.get('w_any', 0.6); w_same = P.get('w_same', 2.0); w_dist = P.get('w_dist', 0.35)
    w_h = P.get('w_h', 0.15); w_x = P.get('w_x', 0.05)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    occ = np.zeros((R, R), np.float32)
    occ[:pad, :] = 1; occ[-pad:, :] = 1; occ[:, :pad] = 1; occ[:, -pad:] = 1
    gmap = np.full((R, R), -1, np.int16)
    gid = {g: n for n, g in enumerate(groups)}
    placed = {}
    gsum = {}
    for i in order:
        isl = islands[i]
        g = gid[isl.group]
        best = None
        same = (gmap == g).astype(np.float32)
        has_same = same.any()
        occp = np.pad(occ, 1, constant_values=1.0)
        samep = np.pad(same, 1)
        for k in ((0, 1, 3) if isl.rotate else (0,)):
            mask, off = _island_mask(isl, scale, k, R, pad, kernel)
            h, w = mask.shape
            if h >= R or w >= R:
                continue
            coll = fftconvolve(occ, mask[::-1, ::-1].astype(np.float32), mode='valid')
            free = coll < 0.5
            if not free.any():
                continue
            ys, xs = np.nonzero(free)
            ring = np.pad((cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(np.float32) - mask).clip(0, 1), 1)
            rr = ring[::-1, ::-1]
            c_any = fftconvolve(occp, rr, mode='valid')[ys, xs]
            per = ring.sum()
            cost = -w_any * c_any / per * (h + w)
            if has_same:
                c_same = fftconvolve(samep, rr, mode='valid')[ys, xs]
                cy, cx = gsum[g][0] / gsum[g][2], gsum[g][1] / gsum[g][2]
                d = np.hypot(ys + h / 2 - cy, xs + w / 2 - cx)
                cost = cost - w_same * c_same / per * (h + w) + w_dist * d
            cost = cost + w_h * (ys + h) + w_x * xs + (6 if k else 0)
            j = int(np.argmin(cost))
            if best is None or cost[j] < best[0]:
                best = (cost[j], ys[j], xs[j], k, mask, off)
        if best is None:
            return None
        _, y, x, k, mask, off = best
        occ[y:y + mask.shape[0], x:x + mask.shape[1]] += mask
        sub = gmap[y:y + mask.shape[0], x:x + mask.shape[1]]
        sub[mask > 0] = g
        a = float(mask.sum())
        s0 = gsum.get(g, [0.0, 0.0, 0.0])
        gsum[g] = [s0[0] + (y + mask.shape[0] / 2) * a, s0[1] + (x + mask.shape[1] / 2) * a, s0[2] + a]
        placed[isl.cid] = (y, x, k, off)
    return placed, occ, gmap


def pack_atlas_affinity(islands, group_order, R=2048, pad=4, target=0.70, steps=7, log=print, params=None):
    groups = [g for g in group_order if any(i.group == g for i in islands)]
    groups += sorted({i.group for i in islands} - set(groups))
    tot = sum(i.area for i in islands)
    # group sequence: by largest island, then semantic order; inside a group mirrored pairs largest first
    big = {g: max(i.area for i in islands if i.group == g) for g in groups}
    gseq = sorted(groups, key=lambda g: (-big[g], groups.index(g)))
    order = []
    for g in gseq:
        idx = [n for n, i in enumerate(islands) if i.group == g]
        sub = [islands[n] for n in idx]
        order += [idx[j] for j in _pair_order(sub)]
    if (params or {}).get('order') == 'size':
        # interleave groups: largest islands first overall (pairs kept together)
        order = sorted(order, key=lambda n: -round(islands[n].area / tot * 200))

    def search(R_, pad_, lo, hi, n, first):
        best = None
        for it in range(n + 1):
            s = first if it == 0 else (lo + hi) / 2
            res = try_pack_affinity(islands, groups, s, R_, pad_, order, params)
            if res is None:
                hi = s
            else:
                best = (s, res); lo = s
            log(f'    [{R_}] scale {s:.4f} -> {"fit" if res else "overflow"} (util {tot*s*s*100:.1f}%)')
        return best
    best = search(R // 2, max(1, pad // 2), np.sqrt(0.25 / tot), np.sqrt(0.92 / tot), steps, np.sqrt(target / tot))
    if best is None:
        raise RuntimeError('affinity packing failed')
    s0 = best[0]
    best = search(R, pad, s0 * 0.97, s0 * 1.05, 3, s0)
    while best is None:
        s0 *= 0.97
        best = search(R, pad, s0, s0, 0, s0)
    s, (placed, occ, gmap) = best
    xf = {}
    for isl in islands:
        y, x, k, off = placed[isl.cid]
        xf[isl.cid] = dict(scale=s * isl.prio, rot=k, gx=x - off[0], gy=y - off[1], R=R)
    zones = {}
    for g in groups:
        ys, xs = np.nonzero(gmap == groups.index(g))
        if len(ys):
            zones[g] = (xs.min() / R, ys.min() / R, (xs.max() + 1) / R, (ys.max() + 1) / R)
    return xf, zones, s, tot * s * s
