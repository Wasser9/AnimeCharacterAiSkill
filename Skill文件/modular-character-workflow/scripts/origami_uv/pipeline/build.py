"""Python-side stages of the origami UV pipeline (run through run.py)."""
import json, pickle, time, hashlib
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage

from uvcore import Mesh, uv_area
from segment import segmenter_for
from unwrap import unwrap_object
from pack import Island, pack_atlas_affinity, pack_atlas_blocks, apply_transform
from transfer import transfer, load_srgb, save_png
from render_util import palette


def _p(work, *a):
    p = Path(work).joinpath(*a); p.parent.mkdir(parents=True, exist_ok=True); return p


# ------------------------------------------------------------------ stage 2+3: segment, cut, flatten
def stage_unwrap(cfg, work, obj):
    t = time.time()
    m = Mesh(Path(work) / 'inspect' / f'{obj}.npz')
    E = np.load(Path(work) / 'exposure.npz')[obj]
    labels, regions = segmenter_for(cfg, obj)(m, E)
    u = cfg['unwrap']
    res, info = unwrap_object(m, labels, regions, E, slim_iters=u['slim_iterations'], verbose=False,
                              max_relief=u['max_relief_cuts'], accept=u['accept'])
    pickle.dump(dict(res=res, info=info, regions=regions), open(_p(work, 'unwrap', f'{obj}.pkl'), 'wb'))
    return dict(object=obj, charts=sum(1 for i in info if i['region'] >= 0), regions=len(regions), seconds=round(time.time() - t, 1))


# ------------------------------------------------------------------ stage 4: semantic packing
PACK_VARIANTS = [
    dict(),                                                                   # v0 default
    dict(w_any=0.9, w_same=2.0, w_dist=0.35, w_h=0.10, w_x=0.03),
    dict(w_any=0.6, w_same=3.0, w_dist=0.50, w_h=0.15, w_x=0.05),
    dict(w_any=0.4, w_same=1.5, w_dist=0.25, w_h=0.30, w_x=0.10),
    dict(w_any=0.8, w_same=2.5, w_dist=0.35, w_h=0.15, w_x=0.05, order='size'),
    dict(w_any=1.2, w_same=2.0, w_dist=0.30, w_h=0.05, w_x=0.02),
]


def stage_pack(cfg, work, obj, variant=0):
    t = time.time()
    cached = Path(work) / 'pack' / f'{obj}_v{variant}.pkl'
    src = Path(work) / 'unwrap' / f'{obj}.pkl'
    if cached.exists() and cached.stat().st_mtime > src.stat().st_mtime:
        c = pickle.load(open(cached, 'rb'))
        if c.get('cfg_pack') == cfg['pack']:
            return dict(object=obj, variant=variant, utilisation=round(c['util'], 4), scale=round(c['scale'], 4), seconds=0.0, cached=True)
    d = pickle.load(open(Path(work) / 'unwrap' / f'{obj}.pkl', 'rb'))
    res, info, regs = d['res'], d['info'], d['regions']
    pk = cfg['pack']
    merge = (pk.get('merge') or {}).get(obj, {}) or {}
    cuv, fc = res['corner_uv'], res['face_chart']
    isl = []
    for it in info:
        if it['region'] < 0:
            continue
        fs = np.nonzero(fc == it['chart'])[0]
        r = regs[it['region']]
        isl.append(Island(it['chart'], cuv[fs], r['prio'], merge.get(r['group'], r['group']), r['rotate'], r['name']))
    fn = pack_atlas_affinity if pk['mode'] == 'affinity' else pack_atlas_blocks
    log = []
    kw = dict(params=PACK_VARIANTS[variant]) if pk['mode'] == 'affinity' else {}
    # group order on the sheet (earlier = placed first, top-left); default: order in which the segmenter created them
    order = list((pk.get('groups') or {}).get(obj) or [])
    for g in [i.group for i in isl]:
        if g not in order:
            order.append(g)
    xf, zones, s, util = fn(isl, order, R=pk['grid'], pad=pk['pad_cells'], target=pk['target_utilisation'],
                            log=lambda x: log.append(x), **kw)
    final = np.zeros_like(cuv)
    for c, x in xf.items():
        fs = np.nonzero(fc == c)[0]
        final[fs] = apply_transform(cuv[fs], x)
    pickle.dump(dict(final=final, xf=xf, zones=zones, scale=s, util=util, log=log, variant=variant, cfg_pack=cfg['pack']),
                open(_p(work, 'pack', f'{obj}_v{variant}.pkl'), 'wb'))
    return dict(object=obj, variant=variant, utilisation=round(util, 4), scale=round(s, 4), seconds=round(time.time() - t, 1))


def choose_pack(work, obj, results):
    import shutil
    mine = [r for r in results if r['object'] == obj]
    best = max(mine, key=lambda r: (r['utilisation'], -r['variant']))
    shutil.copyfile(Path(work) / 'pack' / f"{obj}_v{best['variant']}.pkl", Path(work) / 'pack' / f'{obj}.pkl')
    return dict(object=obj, chosen_variant=best['variant'], utilisation=best['utilisation'],
                variants={r['variant']: r['utilisation'] for r in mine})


# ------------------------------------------------------------------ stage 5: texture migration
def stage_transfer(cfg, work, out, obj):
    t = time.time()
    m = Mesh(Path(work) / 'inspect' / f'{obj}.npz')
    u = pickle.load(open(Path(work) / 'unwrap' / f'{obj}.pkl', 'rb'))
    p = pickle.load(open(Path(work) / 'pack' / f'{obj}.pkl', 'rb'))
    src = load_srgb(Path(work) / 'inspect' / 'src_tex' / f'{obj}.png')
    size = cfg['texture_size']
    tex, cov, ids = transfer(p['final'], m.uv_orig, src, size, cfg['supersample'], satellites=u['res']['satellites'])
    save_png(_p(out, 'final', 'textures', f'{obj}_basecolor.png'), tex)
    np.save(_p(work, 'transfer', f'{obj}_coverage.npy'), cov)
    # fidelity: compare original vs migrated colour at 4 interior points of every face
    new = load_srgb(Path(out) / 'final' / 'textures' / f'{obj}_basecolor.png')
    from transfer import bilinear
    W = np.array([[1 / 3, 1 / 3, 1 / 3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])

    def samp(img, uv):
        H, Wd = img.shape[:2]
        return np.mean([bilinear(img, (uv * w[None, :, None]).sum(1)[:, 0] * Wd - 0.5,
                                 (1 - (uv * w[None, :, None]).sum(1)[:, 1]) * H - 0.5) for w in W], 0)
    co = samp(src, m.uv_orig); cn = samp(new, p['final'])
    dd = np.abs(co - cn).max(1)
    wa = m.area / m.area.sum()
    order = np.argsort(dd); cw = np.cumsum(wa[order])
    fid = dict(mean_abs=float(np.abs(co - cn).mean()), p95_max_channel=float(dd[order][np.searchsorted(cw, 0.95)]),
               area_over_0p1=float(wa[dd > 0.1].sum()), coverage=float(cov.mean()))
    return dict(object=obj, seconds=round(time.time() - t, 1), **fid)


# ------------------------------------------------------------------ UV gate (hard checks + metrics)
def uv_gate(cfg, work, obj):
    m = Mesh(Path(work) / 'inspect' / f'{obj}.npz')
    u = pickle.load(open(Path(work) / 'unwrap' / f'{obj}.pkl', 'rb'))
    p = pickle.load(open(Path(work) / 'pack' / f'{obj}.pkl', 'rb'))
    final = p['final']; fc = u['res']['face_chart']; info = u['info']; regs = u['regions']
    sat = np.zeros(m.nf, bool); sat[u['res']['satellites']] = True
    size = cfg['texture_size']
    oob = int(((final < 0) | (final > 1)).any(2).any(1).sum())
    a2 = uv_area(final.reshape(-1, 2), np.arange(3 * m.nf).reshape(-1, 3))
    nondeg = m.area > 1e-12 * m.area.max()
    flipped = int(((a2 < 0) & nondeg & ~sat).sum())
    # overlap between different charts: rasterise charts at 4K, count texels claimed by two charts
    chart_map = np.full((size, size), -1, np.int32)
    overlap = 0
    pts = np.round(np.stack([final[..., 0] * size - 0.5, (1 - final[..., 1]) * size - 0.5], -1) * 16).astype(np.int32)
    charts = sorted(set(fc[~sat].tolist()))
    self_overlap = {}
    for c in charts:
        fs = np.nonzero((fc == c) & ~sat)[0]
        mk = np.zeros((size, size), np.uint8)
        cv2.fillPoly(mk, list(pts[fs]), 1, cv2.LINE_8, 4)
        inside = mk.astype(bool)
        overlap += int((inside & (chart_map >= 0)).sum())
        chart_map[inside & (chart_map < 0)] = c
        # a chart folding over itself covers fewer texels than its analytic area
        analytic = float(np.abs(a2[fs]).sum()) * size * size
        union = float(inside.sum())
        if analytic > 400 and union < 0.97 * analytic:
            self_overlap[int(c)] = round(1 - union / analytic, 3)
    cov = chart_map >= 0
    dist, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
    L = chart_map[iy, ix]
    hl = L[:, 1:] != L[:, :-1]
    vl = L[1:, :] != L[:-1, :]
    g1 = (dist[:, 1:] + dist[:, :-1] + 1)[hl]
    g2 = (dist[1:, :] + dist[:-1, :] + 1)[vl]
    min_gap = float(min(g1.min() if g1.size else 1e9, g2.min() if g2.size else 1e9))
    # texel density per region (px per metre at the atlas size)
    dens = {}
    for ri, r in enumerate(regs):
        fs = np.nonzero((u['res']['labels'] == ri))[0]
        if not len(fs):
            continue
        a_uv = np.abs(a2[fs]).sum(); a3 = m.area[fs].sum()
        dens[r['name']] = round(float(np.sqrt(a_uv / max(a3, 1e-12)) * size), 1)
    util = float(np.abs(a2[~sat]).sum())
    dist_stats = [i for i in info if i['region'] >= 0]
    w = np.array([i['area3d'] for i in dist_stats]); w /= w.sum()
    return dict(object=obj, islands=len(charts), regions=len(regs), utilisation=round(util, 4),
                out_of_bounds_faces=oob, flipped_faces=flipped, overlap_texels_between_islands=overlap,
                self_overlapping_islands=self_overlap,
                min_gap_px=round(min_gap, 1),
                distortion=dict(conf95_area_weighted=round(float((w * np.array([i['conf_p95'] for i in dist_stats])).sum()), 3),
                                area_ok_area_weighted=round(float((w * np.array([i['area_ok'] for i in dist_stats])).sum()), 4),
                                worst=sorted([(i['name'], round(i['conf_p95'], 2)) for i in dist_stats], key=lambda x: -x[1])[:5]),
                texel_density_px_per_m=dens)


# ------------------------------------------------------------------ guides for artists
def stage_guides(cfg, work, out, obj):
    m = Mesh(Path(work) / 'inspect' / f'{obj}.npz')
    u = pickle.load(open(Path(work) / 'unwrap' / f'{obj}.pkl', 'rb'))
    p = pickle.load(open(Path(work) / 'pack' / f'{obj}.pkl', 'rb'))
    final = p['final']; fc = u['res']['face_chart']; regs = u['regions']
    reg_of_chart = {i['chart']: i['region'] for i in u['info']}
    freg = np.array([reg_of_chart.get(c, -1) for c in fc])
    size = cfg['texture_size']
    # chart boundary edges (in final uv)
    pts = np.stack([final[..., 0] * size, (1 - final[..., 1]) * size], -1)
    e = np.concatenate([pts[:, [0, 1]], pts[:, [1, 2]], pts[:, [2, 0]]])
    q = np.round(e * 64).astype(np.int64)
    k = np.sort(np.stack([q[:, 0, 0] * 10 ** 7 + q[:, 0, 1], q[:, 1, 0] * 10 ** 7 + q[:, 1, 1]], 1), 1)
    k = np.c_[k, np.tile(fc, 3)]
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    bd = e[cnt[inv.ravel()] == 1]
    # 1) transparent overlay (4K, same pixel grid as the texture)
    ov = np.zeros((size, size, 4), np.uint8)
    for a_, b_ in bd:
        cv2.line(ov, tuple(np.round(a_ * 16).astype(int)), tuple(np.round(b_ * 16).astype(int)), (255, 255, 255, 255), 3, cv2.LINE_AA, 4)
    for a_, b_ in bd:
        cv2.line(ov, tuple(np.round(a_ * 16).astype(int)), tuple(np.round(b_ * 16).astype(int)), (40, 40, 40, 255), 1, cv2.LINE_AA, 4)
    # region labels (Chinese labels need a TrueType font: use PIL)
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for fpath in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            font = ImageFont.truetype(fpath, 34); fsmall = ImageFont.truetype(fpath, 26); break
        except Exception:
            continue
    regions_json = []
    pal = palette(len(regs) + 1, 11)
    idmap = np.zeros((size, size, 3), np.uint8)
    ppts = np.round(pts * 16).astype(np.int32)
    for ri, r in enumerate(regs):
        fs = np.nonzero(freg == ri)[0]
        if not len(fs):
            continue
        col = tuple(int(c * 255) for c in pal[ri])
        cv2.fillPoly(idmap, list(ppts[fs]), col[::-1], cv2.LINE_8, 4)
        uvs = final[fs].reshape(-1, 2)
        charts = sorted(set(fc[fs].tolist()))
        bb = [float(uvs[:, 0].min()), float(uvs[:, 1].min()), float(uvs[:, 0].max()), float(uvs[:, 1].max())]
        regions_json.append(dict(id=ri, name=r['name'], cn=r['cn'], group=r['group'], islands=len(charts),
                                 uv_bbox=[round(v, 4) for v in bb], texel_priority=r['prio'], color=list(col)))
    cv2.imwrite(str(_p(out, 'guides', f'{obj}_UV_RegionID.png')), idmap)
    ovi = Image.fromarray(cv2.cvtColor(ov, cv2.COLOR_BGRA2RGBA))
    dr = ImageDraw.Draw(ovi)
    for rj in regions_json:
        ri = rj['id']
        fs = np.nonzero(freg == ri)[0]
        ch = np.bincount(fc[fs]).argmax(); f2 = fs[fc[fs] == ch]
        c = (final[f2].reshape(-1, 2) * [1, -1] + [0, 1]).mean(0) * size
        txt = f"{ri:02d} {rj['cn']}"
        dr.text((c[0], c[1]), txt, font=font, fill=(255, 255, 255, 255), anchor='mm', stroke_width=4, stroke_fill=(20, 20, 20, 255))
    ovi.save(_p(out, 'guides', f'{obj}_UV_Overlay.png'))
    # 2) guide: texture + outlines + labels, half size
    tex = Image.open(Path(out) / 'final' / 'textures' / f'{obj}_basecolor.png').convert('RGBA')
    covm = np.load(Path(work) / 'transfer' / f'{obj}_coverage.npy')
    t = np.array(tex); t[~covm] = (t[~covm] * 0.25 + np.array([235, 236, 240, 255]) * 0.75).astype(np.uint8)
    g = Image.alpha_composite(Image.fromarray(t), ovi).resize((size // 2, size // 2), Image.LANCZOS)
    g.convert('RGB').save(_p(out, 'guides', f'{obj}_UV_Guide.png'))
    json.dump(dict(object=obj, texture=f'final/textures/{obj}_basecolor.png', uv_layer='UV_clean',
                   face_attribute='UV_panel', regions=regions_json),
              open(_p(out, 'guides', f'{obj}_regions.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return dict(object=obj, regions=len(regions_json))


# ------------------------------------------------------------------ data for the Blender stage
def stage_apply_data(cfg, work, obj):
    m = Mesh(Path(work) / 'inspect' / f'{obj}.npz')
    u = pickle.load(open(Path(work) / 'unwrap' / f'{obj}.pkl', 'rb'))
    p = pickle.load(open(Path(work) / 'pack' / f'{obj}.pkl', 'rb'))
    final = p['final']
    # working corner (f,k) == source loop 3f + perm[f,k] (all polygons are triangles)
    loop_uv = np.zeros((3 * m.nf, 2), np.float32)
    idx = (3 * np.arange(m.nf)[:, None] + m.perm).ravel()
    loop_uv[idx] = final.reshape(-1, 2)
    fc = u['res']['face_chart']
    reg_of_chart = {i['chart']: i['region'] for i in u['info']}
    panel = np.array([reg_of_chart.get(c, -1) for c in fc], np.int32)
    names = [r['name'] for r in u['regions']]
    cn = [r['cn'] for r in u['regions']]
    np.savez(_p(work, 'apply', f'{obj}.npz'), loop_uv=loop_uv, panel=panel, names=np.array(names), cn=np.array(cn))
    h = hashlib.sha256(loop_uv.tobytes()).hexdigest()
    return dict(object=obj, uv_sha256=h)
