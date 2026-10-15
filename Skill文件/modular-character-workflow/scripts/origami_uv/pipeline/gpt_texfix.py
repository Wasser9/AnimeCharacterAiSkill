"""Texture cleanup with an external image model (ChatGPT) without breaking the UV layout.

prepare: cut every atlas into per-group crops (one packing group = one body-part zone), pad each crop to an aspect ratio the
         image model returns (1:1, 3:2, 2:3) and save crop + island mask + manifest.
apply:   for every crop that has a returned image: resize back, register it to the original crop (global affine ECC + dense
         DIS optical flow, so small shifts/warps of the model are undone), keep only texels inside the islands, feather the
         island border, write a new atlas and redo the padding (pull-push band + flat background).
Usage:
  python gpt_texfix.py prepare <output_dir> <work_texfix_dir>
  python gpt_texfix.py apply   <output_dir> <work_texfix_dir> <result_dir> <new_texture_dir>
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import srgb_to_linear, linear_to_srgb, push_pull
from scipy import ndimage

SIZES = {(1, 1): (1024, 1024), (3, 2): (1536, 1024), (2, 3): (1024, 1536)}


def run_objects(out):
    """Objects of the last run (output/run_config.resolved.json), for command-line use."""
    return json.loads((Path(out) / 'run_config.resolved.json').read_text(encoding='utf-8'))['objects']


def group_masks(out, obj, size, merge):
    work = Path(out) / 'work'
    u = pickle.load(open(work / 'unwrap' / f'{obj}.pkl', 'rb'))
    p = pickle.load(open(work / 'pack' / f'{obj}.pkl', 'rb'))
    reg_of_chart = {i['chart']: i['region'] for i in u['info']}
    fc = u['res']['face_chart']
    groups = {}
    pts = np.round(np.stack([p['final'][..., 0] * size - 0.5, (1 - p['final'][..., 1]) * size - 0.5], -1) * 16).astype(np.int32)
    for f_chart in np.unique(fc):
        r = reg_of_chart.get(int(f_chart), -1)
        if r < 0:
            continue
        g = u['regions'][r]['group']; g = merge.get(g, g)
        groups.setdefault(g, []).append(np.nonzero(fc == f_chart)[0])
    masks = {}
    for g, lst in groups.items():
        fs = np.concatenate(lst)
        mk = np.zeros((size, size), np.uint8)
        cv2.fillPoly(mk, list(pts[fs]), 255, cv2.LINE_8, 4)
        masks[g] = mk
    return masks


def best_aspect(w, h):
    r = w / h
    return min(SIZES, key=lambda k: abs(np.log(r / (k[0] / k[1]))))


def fits(w, h, max_scale_down):
    """True if a w x h window fits one of the model's output sizes with at most max_scale_down reduction."""
    return any(w * max_scale_down <= W * 1.0 / 1.0 * 1 and h * max_scale_down <= H for W, H in
               [(sz[0] / max_scale_down / max_scale_down, sz[1] / max_scale_down / max_scale_down) for sz in SIZES.values()])         if False else any(w <= W / max_scale_down and h <= H / max_scale_down for W, H in SIZES.values())


def prepare(out, tdir, objects=None, src_dir=None, max_down=0.50, margin=24):
    """Cluster islands spatially into crops; every island lands whole in exactly one crop.
    src_dir: folder with <object>_migrated.png (the texture before any clean-up); default output/final/textures."""
    out = Path(out); tdir = Path(tdir); (tdir / 'crops').mkdir(parents=True, exist_ok=True)
    man = []
    for obj in objects or run_objects(out):
        tp = Path(src_dir) / f'{obj}_migrated.png' if src_dir else out / 'final' / 'textures' / f'{obj}_basecolor.png'
        tex = cv2.imread(str(tp))
        size = tex.shape[0]
        cov = np.load(out / 'work' / 'transfer' / f'{obj}_coverage.npy')
        lab, n = ndimage.label(cov)
        objs = ndimage.find_objects(lab)
        isl = []
        for i, sl in enumerate(objs):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            area = int((lab[sl] == i + 1).sum())
            if area < 30:
                continue
            isl.append([i + 1, max(x0 - margin, 0), max(y0 - margin, 0), min(x1 + margin, size), min(y1 + margin, size), area])
        isl.sort(key=lambda r: -r[5])
        crops = []  # [x0,y0,x1,y1,[labels]]
        for lid, x0, y0, x1, y1, area in isl:
            best = None
            for c in crops:
                ux0, uy0, ux1, uy1 = min(c[0], x0), min(c[1], y0), max(c[2], x1), max(c[3], y1)
                if not fits(ux1 - ux0, uy1 - uy0, max_down):
                    continue
                grow = (ux1 - ux0) * (uy1 - uy0) - (c[2] - c[0]) * (c[3] - c[1])
                if best is None or grow < best[0]:
                    best = (grow, c, (ux0, uy0, ux1, uy1))
            if best is None:
                crops.append([x0, y0, x1, y1, [lid]])
            else:
                c = best[1]; c[0], c[1], c[2], c[3] = best[2]; c[4].append(lid)
        for k, (x0, y0, x1, y1, lids) in enumerate(sorted(crops, key=lambda c: (c[1] // 512, c[0]))):
            w, h = x1 - x0, y1 - y0
            a = best_aspect(w, h)
            tw = max(w, int(np.ceil(h * a[0] / a[1]))); th = max(h, int(np.ceil(w * a[1] / a[0])))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            X0 = int(round(cx - tw / 2)); Y0 = int(round(cy - th / 2))
            crop = np.full((th, tw, 3), 128, np.uint8); cm = np.zeros((th, tw), np.uint8)
            sx0, sy0 = max(X0, 0), max(Y0, 0); sx1, sy1 = min(X0 + tw, size), min(Y0 + th, size)
            win = tex[sy0:sy1, sx0:sx1]
            own = np.isin(lab[sy0:sy1, sx0:sx1], lids)
            dst = crop[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0]
            dst[own] = win[own]
            cm[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0][own] = 255
            gw, gh = SIZES[a]
            send = cv2.resize(crop, (gw, gh), interpolation=cv2.INTER_AREA)
            name = f'{obj}__{k:02d}'
            cv2.imwrite(str(tdir / 'crops' / f'{name}.png'), send)
            cv2.imwrite(str(tdir / 'crops' / f'{name}_mask.png'), cm)
            man.append(dict(name=name, object=obj, window=[int(X0), int(Y0), int(tw), int(th)], send_size=[gw, gh],
                            scale=round(gw / tw, 4), islands=len(lids), island_px=int((cm > 0).sum())))
    (tdir / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    return man


def register(src, ref, mask):
    """Warp src onto ref (both uint8 BGR, same size). Returns warped src and a quality score."""
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g2 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    # 1) global affine on edge maps (colours may change a lot, outlines should not)
    e1 = cv2.GaussianBlur(cv2.Canny((g1 * 255).astype(np.uint8), 30, 90).astype(np.float32) / 255, (0, 0), 3)
    e2 = cv2.GaussianBlur(cv2.Canny((g2 * 255).astype(np.uint8), 30, 90).astype(np.float32) / 255, (0, 0), 3)
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(e1, e2, M, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        M = np.eye(2, 3, dtype=np.float32)
    h, w = g1.shape
    src_a = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    # 2) dense residual flow on edge maps, strongly smoothed (only small local corrections)
    g2a = cv2.cvtColor(src_a, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), g2a, None)
    flow = cv2.GaussianBlur(flow, (0, 0), 6)
    mag = np.linalg.norm(flow, axis=2)
    flow[mag > 12] *= (12 / mag[mag > 12])[:, None]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    src_f = cv2.remap(src_a, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # quality: edge agreement inside the islands before / after
    def score(a):
        ea = cv2.GaussianBlur(cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), 30, 90).astype(np.float32) / 255, (0, 0), 3)
        mm = mask > 0
        return float(np.corrcoef(ea[mm], e1[mm])[0, 1]) if mm.sum() > 100 else 0.0
    return src_f, dict(affine=M.round(4).tolist(), edge_corr_raw=round(score(cv2.resize(src, (w, h))), 3),
                       edge_corr_aligned=round(score(src_f), 3), flow_px_p95=round(float(np.quantile(mag, 0.95)), 2))


def guided_filter(guide, src, r, eps):
    """He et al. guided filter, grey guide (luminance of `guide`), colour source. float32 HxWx3 in, out."""
    I = (guide * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2)
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)
    mI = box(I); vI = box(I * I) - mI * mI
    out = np.empty_like(src)
    for c in range(3):
        p = src[..., c]
        mp = box(p); cIp = box(I * p) - mI * mp
        a = cIp / (vI + eps); b = mp - a * mI
        out[..., c] = box(a) * I + box(b)
    return out


def _fg(img, tol=14):
    return np.abs(img.astype(np.int16) - 128).max(2) > tol


def _iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def register_silhouette(res, tw, th, mask):
    """Place the model's image onto the crop window by matching silhouettes (non-grey foreground) to the
    island mask. Several initial placements are tried because the model sometimes returns another aspect
    ratio (stretch / fit / fill), each refined by an affine ECC on blurred silhouettes and optionally by a
    smooth dense flow on silhouette distance fields. The candidate with the highest IoU wins (the raw
    stretch is always a candidate, so alignment can never make things worse)."""
    h0, w0 = res.shape[:2]
    m = (mask > 0)
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 4)
    cands = {}
    stretch = cv2.resize(res, (tw, th), interpolation=cv2.INTER_CUBIC)
    cands['stretch'] = stretch
    for mode in ('fit', 'fill'):
        sc = (min if mode == 'fit' else max)(tw / w0, th / h0)
        if abs(w0 / h0 - tw / th) < 0.02:
            continue
        r2 = cv2.resize(res, (int(round(w0 * sc)), int(round(h0 * sc))), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((th, tw, 3), 128, np.uint8)
        ox = (tw - r2.shape[1]) // 2; oy = (th - r2.shape[0]) // 2
        sx0, sy0 = max(0, -ox), max(0, -oy); dx0, dy0 = max(0, ox), max(0, oy)
        wv = min(r2.shape[1] - sx0, tw - dx0); hv = min(r2.shape[0] - sy0, th - dy0)
        canvas[dy0:dy0 + hv, dx0:dx0 + wv] = r2[sy0:sy0 + hv, sx0:sx0 + wv]
        cands[mode] = canvas
    best = None
    for name, img in list(cands.items()):
        variants = [(name, img)]
        sf = cv2.GaussianBlur(_fg(img).astype(np.float32), (0, 0), 4)
        M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(mf, sf, M, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7), None, 5)
            a_img = cv2.warpAffine(img, M, (tw, th), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
            variants.append((name + '+affine', a_img))
            # residual local warp from silhouette signed distance fields (smooth, max 10 px)
            def sdf(b):
                return (ndimage.distance_transform_edt(~b) - ndimage.distance_transform_edt(b)).astype(np.float32)
            d1 = np.clip(sdf(m), -40, 40); d2 = np.clip(sdf(_fg(a_img)), -40, 40)
            u8 = lambda d: np.clip((d + 40) * 3.18, 0, 255).astype(np.uint8)
            flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(u8(d1), u8(d2), None)
            flow = cv2.GaussianBlur(flow, (0, 0), 10)
            mag = np.linalg.norm(flow, axis=2); big = mag > 10
            flow[big] *= (10 / mag[big])[:, None]
            gx, gy = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
            f_img = cv2.remap(a_img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
            variants.append((name + '+affine+flow', f_img))
        except cv2.error:
            pass
        for vn, vi in variants:
            sc_ = _iou(_fg(vi), m)
            if best is None or sc_ > best[0]:
                best = (sc_, vn, vi)
    return best[2], dict(method=best[1], iou_raw=round(_iou(_fg(stretch), m), 4), iou=round(best[0], 4))


def apply(out, tdir, rdir, newdir, feather=3, pad_px=16, detail=0.7, detail_override=None,
          color_match=None, level_seams=True, rim_snap=24, gf_radius=10, gf_eps=2e-3, objects=None):
    """color_match: crop-name prefixes that get the per-island colour gain (None = every crop)."""
    detail_override = detail_override or {}
    out = Path(out); tdir = Path(tdir); rdir = Path(rdir); newdir = Path(newdir); newdir.mkdir(parents=True, exist_ok=True)
    man = json.loads((tdir / 'manifest.json').read_text(encoding='utf-8'))
    log = []
    for obj in objects or run_objects(out):
        items = [m for m in man if m['object'] == obj and (rdir / f"{m['name']}.png").exists()]
        tex = cv2.imread(str(out / 'final' / 'textures' / f'{obj}_basecolor.png'))
        if not items:
            cv2.imwrite(str(newdir / f'{obj}_basecolor.png'), tex)
            continue
        size = tex.shape[0]
        lin = srgb_to_linear(tex[..., ::-1].astype(np.float32) / 255).astype(np.float32)
        ref_lin = lin.copy()
        cov = np.load(out / 'work' / 'transfer' / f'{obj}_coverage.npy')
        for m in items:
            X0, Y0, tw, th = m['window']
            ref = cv2.imread(str(tdir / 'crops' / f"{m['name']}.png"))
            ref = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_CUBIC)
            res = cv2.imread(str(rdir / f"{m['name']}.png"), cv2.IMREAD_COLOR)
            cm = cv2.imread(str(tdir / 'crops' / f"{m['name']}_mask.png"), cv2.IMREAD_GRAYSCALE)
            # original pixels of this window (full resolution) for the reference
            orig = np.zeros((th, tw, 3), np.uint8)
            sx0, sy0 = max(X0, 0), max(Y0, 0); sx1, sy1 = min(X0 + tw, size), min(Y0 + th, size)
            orig[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = tex[sy0:sy1, sx0:sx1]
            warped, info = register_silhouette(res, tw, th, cm)
            inside = (cm > 0)
            # islands the model deleted (painted as background grey) or recoloured into background keep the original
            lab_i, n_i = ndimage.label(inside)
            dropped = []
            if n_i:
                wf = warped.astype(np.float32); of = orig.astype(np.float32)
                for k in range(1, n_i + 1):
                    mk = lab_i == k
                    if mk.sum() < 50:
                        continue
                    d_bg = np.abs(wf[mk] - 128).max(1)
                    o_bg = np.abs(of[mk] - 128).max(1)
                    if np.median(d_bg) < 12 and np.median(o_bg) > 20:
                        inside[mk] = False; dropped.append(int(mk.sum()))
            info['islands_kept_original'] = dropped
            r = int(np.ceil(max(3.0, 2.0 / max(m['scale'], 1e-3))))
            core = cv2.erode(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            core &= inside
            # areas where the aligned model image is still background grey (misalignment) are treated like the rim
            # (extrapolated). Only connected grey areas count: isolated dark knit texels near grey are kept.
            grey = inside & ~_fg(warped)
            grey = cv2.morphologyEx(grey.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
            core &= ~grey
            wl = srgb_to_linear(warped[..., ::-1].astype(np.float32) / 255).astype(np.float32)
            ol = srgb_to_linear(orig[..., ::-1].astype(np.float32) / 255).astype(np.float32)
            if color_match is None or any(m['name'].startswith(pfx) for pfx in color_match):
                # keep the original overall hue/brightness of every island (the model tends to re-saturate and
                # separately cleaned crops drift apart): ONE gain per island, from medians over its interior
                lab_isl, n_isl = ndimage.label(inside)
                gains = []
                for k in range(1, n_isl + 1):
                    isl = lab_isl == k
                    mk = isl & core
                    if mk.sum() < 30:
                        continue
                    g = np.median(ol[mk], 0) / np.maximum(np.median(wl[mk], 0), 1e-4)
                    # only a hue/brightness drift is corrected; a big difference means the original median
                    # itself was a stain (e.g. dark blotches on pale fur) and the model colour is kept
                    if np.all((g > 0.75) & (g < 1.33)):
                        wl[isl] = np.clip(wl[isl] * g, 0, 1)
                        gains.append([round(float(x), 3) for x in g])
                info['island_gains'] = gains
            ext = push_pull(np.where(core[..., None], wl, 0).astype(np.float32), core.astype(np.float32))
            wl = np.where(core[..., None], wl, ext)
            # keep the original fine texture (knit, plaid, line work) that the model cannot reproduce at its
            # lower resolution: add back the original high-pass above the model's resolution limit (interior only)
            sig = max(0.8, 0.9 / max(m['scale'], 1e-3))
            hp = ol - cv2.GaussianBlur(ol, (0, 0), sig)
            dk = detail_override.get(m['name'], detail)
            dcore = np.clip(ndimage.distance_transform_edt(inside) / (2.0 * r), 0, 1).astype(np.float32)[..., None]
            blend = np.clip(wl + dk * hp * dcore, 0, 1)
            if rim_snap > 0:
                # near island borders (= 3D seams) the structure must agree on both sides of the seam: a guided
                # filter with the ORIGINAL as guide keeps the cleaned colours but puts edges that exist in both
                # images (e.g. a sock band) back where the original had them; stains (absent in the cleaned
                # image) are not brought back. Blended in over the outer `rim_snap` texels only.
                gfil = guided_filter(ol, blend, gf_radius, gf_eps)
                dr = ndimage.distance_transform_edt(inside)
                wr = np.clip(1.0 - dr / rim_snap, 0, 1).astype(np.float32)[..., None]
                blend = np.clip(blend * (1 - wr) + gfil * wr, 0, 1)
            sub = lin[sy0:sy1, sx0:sx1]
            ins = inside[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0]
            sub[ins] = blend[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0][ins]
            log.append(dict(name=m['name'], **info))
        if level_seams:
            from seam_level import seam_level
            lin, sl = seam_level(out / 'work', obj, lin, ref_lin, cov)
            log.append(dict(name=f'{obj}__seam_level', **sl))
        # padding again from the island coverage
        bgc = lin[cov].mean(0)
        filled = push_pull(np.where(cov[..., None], lin, 0).astype(np.float32), cov.astype(np.float32))
        dist = ndimage.distance_transform_edt(~cov)
        dist[ndimage.binary_fill_holes(cov) & ~cov] = 0      # holes inside islands: filled completely
        w = np.clip((dist - pad_px) / pad_px, 0, 1).astype(np.float32)[..., None]
        filled = filled * (1 - w) + bgc[None, None, :] * w
        srgb = linear_to_srgb(filled)
        cv2.imwrite(str(newdir / f'{obj}_basecolor.png'),
                    cv2.cvtColor(np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    (newdir / 'apply_log.json').write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding='utf-8')
    return log


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'prepare':
        for m in prepare(a[2], a[3]):
            print(m['name'], m['window'], m['send_size'], m['scale'])
    else:
        for l in apply(a[2], a[3], a[4], a[5]):
            print(l)
