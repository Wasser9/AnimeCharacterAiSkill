"""Targeted texture touch-ups that run after the clean-up (stage 4), configured in config.touchup:
  - {object: <mesh>, regions: [<region names from guides/*_regions.json>], op: <op>, params: {...}}
Every op edits <tex_dir>/<object>_basecolor.png inside the texels of the named regions, then the gutter around those
regions is re-padded from the edited colours (repad).

remove_stripe  a grey vertical line/smear baked into a garment by the generator (e.g. a shadow streak on a sock).
               mask: low-saturation mid-bright texels with vertical structure, plus small compact blots, away from a
               light top band; fill: clone the material from both sides, blended by position, shifted vertically
               (sub-pixel) to follow a band edge (hem) if there is one; a broken hem is bridged with a cubic.
seam_hem       a tube-shaped part (leg, sleeve) whose band edge (hem) does not meet across the UV seam cut inside the
               island: redraws the hem near the cut on both sides to one curve in model height; lighter rim texels
               along the cut are replaced. Needs the mesh (ctx), regions must be whole tubes.
white_to_skin  white areas that are really bare skin (e.g. the top of the legs above the socks): every texel is
               re-tinted by its share of "neighbour colour -> white" towards a skin colour, keeping shading and edges.
               params.skin_rgb, or params.skin_from: {object, regions} to take the median colour of another region.
despeckle      isolated specks much brighter/darker than their surroundings (<= max_size texels).
"""
import json
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage


def masked_blur(img, mask, sigma):
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-6)


def region_mask_work(work, obj, names=None, size=4096):
    """Texel mask of the named regions (all faces when names is None), rasterised from the unwrap labels and the packed UVs."""
    import pickle
    u = pickle.load(open(Path(work) / 'unwrap' / f'{obj}.pkl', 'rb'))
    p = pickle.load(open(Path(work) / 'pack' / f'{obj}.pkl', 'rb'))
    ids = [i for i, r in enumerate(u['regions']) if names is None or r['name'] in names]
    reg_of_chart = {i['chart']: i['region'] for i in u['info']}
    fr = np.array([reg_of_chart.get(c, -1) for c in u['res']['face_chart']])
    fs = np.nonzero(np.isin(fr, ids))[0]
    pts = np.round(np.stack([p['final'][fs, :, 0] * size - 0.5, (1 - p['final'][fs, :, 1]) * size - 0.5], -1) * 16).astype(np.int32)
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, list(pts), 1, cv2.LINE_8, 4)
    return m > 0


def region_mask(guides, obj, names):
    regs = json.loads((Path(guides) / f'{obj}_regions.json').read_text(encoding='utf-8'))['regions']
    idm = cv2.imread(str(Path(guides) / f'{obj}_UV_RegionID.png'))[:, :, ::-1]
    mk = np.zeros(idm.shape[:2], bool)
    for r in regs:
        if r['name'] in names:
            mk |= np.all(idm == np.array(r['color'], np.uint8), 2)
    return mk


def remove_stripe(tex_bgr, region, sat_max=70, v_min=105, v_max=232, band_v=232, min_len=60, grow=7, shift=40):
    hsv = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.int32); V = hsv[..., 2].astype(np.int32)
    inner = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    band = inner & (S < 40) & (V >= band_v)
    band_core = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
    band = cv2.dilate(band_core.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    cand = inner & (S < sat_max) & (V >= v_min) & (V < v_max) & ~band
    # keep vertical structures only (the stripe runs along the leg = along v in the upright island)
    vert = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((min_len, 1), np.uint8)) > 0
    lab, n = ndimage.label(cv2.dilate(vert.astype(np.uint8), np.ones((3, 3), np.uint8)))
    keep = np.zeros_like(vert)
    for k in range(1, n + 1):
        c = lab == k
        ys, xs = np.nonzero(c)
        if ys.max() - ys.min() > 4 * max(xs.max() - xs.min(), 1) and c.sum() > 400:
            keep |= c
    stripe = keep.copy()
    # small grey/light blots inside the sock (e.g. a light square left by the original texture)
    lab2, n2 = ndimage.label(cand)
    for k in range(1, n2 + 1):
        c = lab2 == k
        a_ = int(c.sum())
        if 40 <= a_ <= 4000:
            ys_, xs_ = np.nonzero(c)
            hh = ys_.max() - ys_.min() + 1; ww = xs_.max() - xs_.min() + 1
            if max(hh, ww) <= 3 * min(hh, ww) and not (c & band).any():   # compact blots only, not hem lines
                keep |= c
    mask = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))) > 0
    # run the stripe on to the band edge and to the island border (the ends were excluded by the band/erosion margins)
    mask |= (cv2.dilate(stripe.astype(np.uint8), np.ones((121, 2 * grow + 1), np.uint8)) > 0) & ~band_core
    mask &= region
    # soft grey fringe right next to the stripe (its edges fade out and widen towards the band)
    fringe_cand = region & (S < 90) & (V >= 95) & (V < 240) & ~band_core
    near = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    mask |= cv2.dilate((fringe_cand & near).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mask &= region
    if not mask.any():
        return tex_bgr, dict(texels=0)
    H, W = mask.shape
    # hem = lower edge of the white top band, per column; the stripe broke it, so it is re-interpolated across the
    # stripe columns from a quadratic fit of the intact hem on both sides
    rows = np.arange(H)[:, None]
    has = band_core.any(0)
    hem = np.where(has, (band_core * rows).max(0), -1).astype(np.float64)
    near_hem = mask & (rows <= hem[None, :] + 60) & has[None, :]
    bad = cv2.dilate(near_hem.any(0).astype(np.uint8)[None, :], np.ones((1, 13), np.uint8))[0] > 0
    # hem_c: the measured hem lightly smoothed (it is integer and jaggy) and, across each stripe gap, a cubic that
    # meets both sides with their height and slope (the hem may step across the stripe, so no global fit)
    xs_all = np.arange(W)
    val = (has & ~bad).astype(np.float64)
    hem_c = ndimage.gaussian_filter1d(np.where(val > 0, hem, 0), 3) / np.maximum(ndimage.gaussian_filter1d(val, 3), 1e-6)
    hem_c[val == 0] = np.nan
    lab, n = ndimage.label(bad & has)
    for k in range(1, n + 1):
        gap = np.nonzero(lab == k)[0]
        a0, b0 = gap[0] - 1, gap[-1] + 1
        lw = np.nonzero((xs_all >= a0 - 30) & (xs_all <= a0) & (val > 0))[0]
        rw = np.nonzero((xs_all >= b0) & (xs_all <= b0 + 30) & (val > 0))[0]
        if len(lw) < 5 or len(rw) < 5:
            continue
        sl, il = np.polyfit(lw, hem_c[lw], 1); sr, ir = np.polyfit(rw, hem_c[rw], 1)
        h0, h1 = sl * a0 + il, sr * b0 + ir; L = float(b0 - a0)
        t = (gap - a0) / L
        hem_c[gap] = ((2 * t ** 3 - 3 * t ** 2 + 1) * h0 + (t ** 3 - 2 * t ** 2 + t) * L * sl
                      + (-2 * t ** 3 + 3 * t ** 2) * h1 + (t ** 3 - t ** 2) * L * sr)
    hem_s = np.where(np.isnan(hem_c), hem, hem_c)
    # the band edge inside the stripe columns (grey notch into the white, possible step of the hem) is redrawn as well
    mask |= bad[None, :] & has[None, :] & (rows >= hem_s[None, :] - 30) & (rows <= hem_s[None, :] + 50) & region
    img = tex_bgr.astype(np.float32) / 255
    known = region & ~mask
    sock = known & ~band_core & (S > 60)
    out = img.copy()
    ys, xs = np.nonzero(mask)
    # clone from beside the stripe, shifted vertically (sub-pixel, bilinear) so the source keeps the same distance to the hem
    lp_known = masked_blur(img, sock, 12)           # low-frequency colour of the clean sock around
    lp_t = lp_known[ys, xs]
    # tone correction only well below the hem (near the hem the clone already sits at the same relative position)
    below = np.where(np.isnan(hem_c[xs]), 1e3, ys - hem_c[xs])
    w = np.clip((below - 10) / 30, 0, 1)[:, None]

    def clone(shifts):
        src = np.zeros((len(ys), 3), np.float32); ok = np.zeros(len(ys), bool)
        for d in shifts:
            xx = np.clip(xs + d, 0, W - 1)
            dy = np.nan_to_num(hem_c[xx] - hem_c[xs], nan=0.0)
            yf = np.clip(ys + dy, 0, H - 2); y0 = np.floor(yf).astype(int); fy = (yf - y0)[:, None].astype(np.float32)
            good = ~ok & known[y0, xx] & known[y0 + 1, xx]
            s = img[y0, xx] * (1 - fy) + img[y0 + 1, xx] * fy
            ratio = lp_t / np.maximum(lp_known[y0, xx], 1e-3)
            src[good] = (s * (1 + w * (ratio - 1)))[good]
            ok |= good
        return src, ok

    srcL, okL = clone([-shift, -2 * shift, -3 * shift])
    srcR, okR = clone([shift, 2 * shift, 3 * shift])
    # blend the two sides by the position across the masked run of the row, so the patch has no switch-over line
    col = np.arange(W, dtype=np.int32)[None, :]
    left = np.maximum.accumulate(np.where(mask, -1, col), axis=1)[ys, xs]
    right = np.minimum.accumulate(np.where(mask, 2 * W, col)[:, ::-1], axis=1)[:, ::-1][ys, xs]
    t = np.clip((xs - left) / np.maximum(right - left, 1), 0, 1)
    wr = (t * t * (3 - 2 * t))[:, None]
    wr = np.where(okL[:, None] & okR[:, None], wr, np.where(okR[:, None], 1.0, 0.0))
    fill = np.where((okL | okR)[:, None], srcL * (1 - wr) + srcR * wr, lp_t)
    out[ys, xs] = np.clip(fill, 0, 1)
    # feather the patch border
    a = np.clip(ndimage.distance_transform_edt(mask) / 5.0, 0, 1)[..., None]
    out = img * (1 - a) + out * a
    return (np.clip(out * 255 + 0.5, 0, 255)).astype(np.uint8), dict(texels=int(mask.sum()))


def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def white_to_skin(tex_bgr, region, skin_rgb, white_s_max=40, white_v_min=200):
    """The white part of the leg panels is bare thigh skin. Every texel is written as a mix between the sock colour
    and the white (t = projection on the sock->white line in linear light, so anti-aliased edges keep their blend),
    and the white share is re-tinted to the skin colour with its relative shading kept."""
    img = srgb2lin(tex_bgr[..., ::-1].astype(np.float32) / 255)
    hsv = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1]; V = hsv[..., 2]
    inner = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    white = inner & (S < white_s_max) & (V >= white_v_min)
    sock = inner & (S > 90) & (V < 140)
    w_ref = np.median(img[white], 0)
    # dirt specks inside the white (skin) area: small holes of the white mask become white first
    holes = ndimage.binary_fill_holes(white) & ~white
    lab, n = ndimage.label(holes)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        small = np.isin(lab, np.nonzero(sizes <= 600)[0] + 1)
        small = cv2.dilate(small.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        img[small & region] = w_ref
    n_ref = np.median(img[sock], 0) if sock.any() else np.zeros(3, np.float32)
    skin = srgb2lin(np.asarray(skin_rgb, np.float32) / 255)
    d = w_ref - n_ref
    t = np.clip(((img - n_ref) * d).sum(2) / max(float((d * d).sum()), 1e-9), 0, 1)
    t = np.where(region, t, 0)[..., None]
    gain = skin / np.maximum(w_ref, 1e-4)
    out = img * (1 + t * (gain - 1))
    res = (lin2srgb(out)[..., ::-1] * 255 + 0.5).astype(np.uint8)
    return res, dict(white_ref=[round(float(x), 4) for x in w_ref], sock_ref=[round(float(x), 4) for x in n_ref],
                     skin_rgb=[int(x) for x in skin_rgb], texels=int((t[..., 0] > 0.5).sum()))


def intra_island_seams(work, obj, names, size=4096):
    """Seams cut inside one island of the named regions (e.g. the back of a leg tube). Per chart: the two sides of the
    cut as point lists (x, y in texels, z = model height), side 0 = the one left of the chart centre."""
    import pickle
    u = pickle.load(open(Path(work) / 'unwrap' / f'{obj}.pkl', 'rb'))
    p = pickle.load(open(Path(work) / 'pack' / f'{obj}.pkl', 'rb'))
    m = np.load(Path(work) / 'inspect' / f'{obj}.npz')
    co = m['co']; F = m['loop_vert'].reshape(-1, 3)
    ids = [i for i, r in enumerate(u['regions']) if r['name'] in names]
    reg_of_chart = {i['chart']: i['region'] for i in u['info']}
    chart = np.asarray(u['res']['face_chart'])
    fsel = np.nonzero(np.isin([reg_of_chart.get(c, -1) for c in chart], ids))[0]
    pix = np.stack([p['final'][..., 0] * size - 0.5, (1 - p['final'][..., 1]) * size - 0.5], -1)
    edges = {}
    for f in fsel:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va, vb = int(F[f, a]), int(F[f, b])
            key = (va, vb) if va < vb else (vb, va)
            edges.setdefault(key, []).append((f, a if va < vb else b, b if va < vb else a))
    out = {}
    for (va, vb), lst in edges.items():
        if len(lst) != 2 or chart[lst[0][0]] != chart[lst[1][0]]:
            continue
        (f1, a1, b1), (f2, a2, b2) = lst
        if np.abs(pix[f1, a1] - pix[f2, a2]).max() < 0.5 and np.abs(pix[f1, b1] - pix[f2, b2]).max() < 0.5:
            continue                                  # welded in UV: not a seam
        out.setdefault(int(chart[f1]), []).append(((pix[f1, a1], pix[f1, b1]), (pix[f2, a2], pix[f2, b2]), (co[va, 2], co[vb, 2])))
    res = {}
    for c, lst in out.items():
        cx = pix[chart == c][..., 0].mean()
        sides = ([], [])
        for e1, e2, z in lst:
            l, r = (e1, e2) if (e1[0][0] + e1[1][0]) < (e2[0][0] + e2[1][0]) else (e2, e1)
            if (l[0][0] + l[1][0]) / 2 > cx or (r[0][0] + r[1][0]) / 2 < cx:
                continue                              # not a cut running through the chart centre line
            for k in range(2):
                sides[0].append((l[k][0], l[k][1], z[k])); sides[1].append((r[k][0], r[k][1], z[k]))
        if len(sides[0]) >= 20:
            res[c] = [np.unique(np.array(s, np.float64), axis=0) for s in sides]
    return res


def _hem_rows(lum, xs_cols, y0, y1):
    """Sub-pixel row of the bright-band -> sock transition in each of the given columns (searched in rows y0..y1)."""
    out = np.full(len(xs_cols), np.nan)
    for i, x in enumerate(xs_cols):
        prof = ndimage.gaussian_filter1d(lum[y0:y1, x], 1.0)
        top, bottom = np.median(prof[:15]), np.median(prof[-15:])
        if top - bottom < 60:
            continue
        thr = (top + bottom) / 2
        below = np.nonzero(prof < thr)[0]
        if len(below) and below[0] > 0:
            j = below[0]
            out[i] = y0 + j - 1 + (prof[j - 1] - thr) / max(prof[j - 1] - prof[j], 1e-3)
    return out


def seam_hem(tex_bgr, region, ctx, anchor=18, win=24, fade=8, rim=5, search=80, blend=4):
    """Redraw the sock hem where it meets the back seam of each leg tube. The painted hem hooks up or down in the last
    texels before the cut, has stray specks there, and the texels along the cut are a little lighter. Per side of the
    cut: measure the hem row column by column, convert it to model height through the seam, bridge the two sides with a
    cubic that meets each side `anchor` texels away from the cut with its height and slope, and replace the hem zone of
    the columns in between by columns cloned further inside, shifted vertically so their hem lands on the bridge.
    Below the hem, the `rim` texels next to the cut are replaced by the sock texture a few texels further in."""
    seams = intra_island_seams(ctx['work'], ctx['obj'], ctx['names'])
    H, W = region.shape
    img = tex_bgr.astype(np.float32)
    lum = img.mean(2)
    out = img.copy()
    log = []
    for c, sides in seams.items():
        geo = []
        for s, pts in enumerate(sides):
            pts = pts[np.argsort(pts[:, 1])]
            inward = 1 if s == 0 else -1
            xs_of = lambda y, pts=pts: np.interp(y, pts[:, 1], pts[:, 0])
            # rough hem row at the cut, from columns 15-30 texels in
            ys = np.arange(int(pts[0, 1]) + 2, min(int(pts[-1, 1]) - 2, int(pts[0, 1]) + 1200))
            cols = np.clip(np.round(xs_of(ys) + inward * 20).astype(int), 0, W - 1)
            prof = ndimage.gaussian_filter1d(lum[ys, cols], 1.5)
            thr = (np.median(prof[:30]) + np.median(prof[-200:])) / 2
            h0 = float(ys[np.nonzero(prof < thr)[0][0]])
            x0 = float(xs_of(h0))
            u = np.arange(0, anchor + 3 * anchor)
            xc = np.clip(np.round(x0 + inward * u).astype(int), 0, W - 1)
            h = _hem_rows(lum, xc, int(h0 - search), int(h0 + search))
            ok = ~np.isnan(h) & (u >= anchor)
            hs = np.interp(u, u[ok], ndimage.median_filter(h[ok], 7))
            hs = ndimage.gaussian_filter1d(hs, 2.0)
            z_of = lambda y, pts=pts: np.interp(y, pts[:, 1], pts[:, 2])
            o = np.argsort(pts[:, 2])
            y_of = lambda z, pts=pts, o=o: np.interp(z, pts[o, 2], pts[o, 1])
            zz = z_of(hs)
            k = (u >= anchor) & (u <= anchor + 20)
            slope = np.polyfit(u[k], zz[k], 1)[0]
            geo.append(dict(pts=pts, inward=inward, x0=x0, h0=h0, u=u, hs=hs, z_anchor=float(z_of(hs[anchor])), slope=slope,
                            y_of=y_of, xs_of=xs_of))
        if len(geo) != 2:
            continue
        # cubic across the cut in model height; t runs from side 1's anchor (-anchor) through the cut (0) to side 0's (+anchor)
        L = 2.0 * anchor
        za, zb = geo[1]['z_anchor'], geo[0]['z_anchor']
        ma, mb = -geo[1]['slope'], geo[0]['slope']           # dz/dt at both ends
        def bridge(t):
            q = (t + anchor) / L
            return ((2 * q ** 3 - 3 * q ** 2 + 1) * za + (q ** 3 - 2 * q ** 2 + q) * L * ma
                    + (-2 * q ** 3 + 3 * q ** 2) * zb + (q ** 3 - q ** 2) * L * mb)
        info = dict(chart=c)
        for s, g in enumerate(geo):
            inward, u = g['inward'], g['u']
            t_of_u = u if s == 0 else -u
            target = np.where(u <= anchor, g['y_of'](bridge(np.clip(t_of_u, -anchor, anchor))), g['hs'])
            D = anchor + 8                                     # clone source: D columns further inside
            changed = 0
            for i in range(anchor + blend):
                x = int(round(g['x0'] + inward * i))
                xsrc = int(round(g['x0'] + inward * (i + D)))
                if not (0 <= x < W and 0 <= xsrc < W):
                    continue
                yt = target[i]; hsrc = g['hs'][i + D]
                wu = 1.0 if i < anchor else max(0.0, (anchor + blend - i) / (blend + 1.0))
                for y in range(int(yt - win - fade), int(yt + win + fade) + 1):
                    if not (0 <= y < H) or not region[y, x]:
                        continue
                    wv = min(1.0, (win + fade - abs(y - yt)) / fade)
                    if wv <= 0:
                        continue
                    ys_ = y - yt + hsrc
                    y0 = int(np.floor(ys_)); fy = ys_ - y0
                    if not (0 <= y0 < H - 1) or not (region[y0, xsrc] and region[y0 + 1, xsrc]):
                        continue
                    src = img[y0, xsrc] * (1 - fy) + img[y0 + 1, xsrc] * fy
                    a = wu * wv
                    out[y, x] = out[y, x] * (1 - a) + src * a
                    changed += 1
            # the texels right at the cut, below the hem zone: same row, a few texels further in
            ys = np.arange(int(target[0] + win), int(g['pts'][-1, 1]) - 1)
            xcut = g['xs_of'](ys)
            for j in range(rim + 2):
                xt = np.round(xcut + inward * j).astype(int); xs2 = np.round(xcut + inward * (j + rim + 3)).astype(int)
                okr = (xt >= 0) & (xt < W) & (xs2 >= 0) & (xs2 < W)
                okr[okr] &= region[ys[okr], xt[okr]] & region[ys[okr], xs2[okr]]
                a = 1.0 if j < rim else 0.5
                out[ys[okr], xt[okr]] = out[ys[okr], xt[okr]] * (1 - a) + img[ys[okr], xs2[okr]] * a
            info[f'side{s}'] = dict(hem_at_cut=round(float(g['hs'][0]), 1), hem_at_anchor=round(float(g['hs'][anchor]), 1),
                                    target_at_cut=round(float(target[0]), 1), texels=changed)
        info['z_anchor_mm'] = [round(za * 1000, 1), round(zb * 1000, 1)]
        log.append(info)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), dict(seams=log)


def despeckle(tex_bgr, region, max_size=16, thr=55.0, k=7):
    """Isolated specks (a few texels much brighter or darker than their surroundings, e.g. leftover highlight texels on
    the sock hem). Only small outlier components are replaced by the local median, so lines and the knit grain stay."""
    img = tex_bgr.astype(np.float32)
    med = np.stack([cv2.medianBlur(tex_bgr[..., i], k) for i in range(3)], -1).astype(np.float32)
    dev = np.abs(img - med).max(2)
    out_m = region & (dev > thr)
    lab, n = ndimage.label(out_m, np.ones((3, 3)))
    if not n:
        return tex_bgr, dict(specks=0)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    small = np.isin(lab, np.nonzero(sizes <= max_size)[0] + 1)
    out = tex_bgr.copy()
    out[small] = med[small].astype(np.uint8)
    return out, dict(specks=int((sizes <= max_size).sum()), texels=int(small.sum()))


OPS = {'remove_stripe': remove_stripe, 'white_to_skin': white_to_skin, 'seam_hem': seam_hem, 'despeckle': despeckle}
NEEDS_CTX = {'seam_hem'}


def repad(tex_bgr, region, cov, pad=32, inset=2):
    """The gutter around an edited region still holds the old colours (padding was built before the edit) and would
    bleed into the island edge through mip-mapping. Gutter texels (and a thin rim inside the island border, where
    rasterisation is partial) whose nearest island texel belongs to the region take the colour of the nearest edited texel."""
    k = np.ones((2 * inset + 1, 2 * inset + 1), np.uint8)
    cov_in = cv2.erode(cov.astype(np.uint8), k) > 0
    src = cov_in & region
    if not src.any():
        return tex_bgr
    d, (oy, ox) = ndimage.distance_transform_edt(~cov_in, return_indices=True)
    tgt = ~cov_in & (d <= pad) & region[oy, ox]
    _, (sy, sx) = ndimage.distance_transform_edt(~src, return_indices=True)
    out = tex_bgr.copy()
    out[tgt] = tex_bgr[sy[tgt], sx[tgt]]
    return out


def median_rgb(work, tex_dir, obj, names):
    tex = cv2.imread(str(Path(tex_dir) / f'{obj}_basecolor.png'))
    mk = cv2.erode(region_mask_work(work, obj, names).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    return [int(x) for x in np.median(tex[mk][:, ::-1], 0)]


def run(items, work, tex_dir):
    """items: list of {object, regions, op, params}. Edits tex_dir/<object>_basecolor.png in place."""
    log = []
    for it in items or []:
        p = Path(tex_dir) / f"{it['object']}_basecolor.png"
        tex = cv2.imread(str(p))
        reg = region_mask_work(work, it['object'], it['regions'])
        params = dict(it.get('params') or {})
        if 'skin_from' in params:            # colour sampled from another atlas (e.g. the neck)
            sf = params.pop('skin_from')
            params['skin_rgb'] = median_rgb(work, tex_dir, sf['object'], sf['regions'])
        if it['op'] in NEEDS_CTX:            # ops that need the mesh (seams, heights)
            params['ctx'] = dict(work=work, obj=it['object'], names=it['regions'])
        new, info = OPS[it['op']](tex, reg, **params)
        new = repad(new, reg, region_mask_work(work, it['object']))
        cv2.imwrite(str(p), new)
        log.append(dict(object=it['object'], op=it['op'], regions=it['regions'], **info))
    return log
