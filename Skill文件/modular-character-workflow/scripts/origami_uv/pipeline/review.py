"""Review sheets: atlas overview, before/after checker and texture renders, texture clean-up close-ups."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for f in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/arial.ttf',
              '/System/Library/Fonts/PingFang.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _grid(paths, captions, cols, cell, title=None, bg=(238, 239, 242)):
    rows = (len(paths) + cols - 1) // cols
    top = 56 if title else 0
    W = cols * cell + (cols + 1) * 12; H = top + rows * (cell + 40) + 12
    sheet = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((14, 12), title, font=_font(30), fill=(30, 30, 35))
    for i, (p, c) in enumerate(zip(paths, captions)):
        r, k = divmod(i, cols)
        x = 12 + k * (cell + 12); y = top + 12 + r * (cell + 40)
        im = Image.open(p).convert('RGB'); im.thumbnail((cell, cell), Image.LANCZOS)
        sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        d.text((x, y + cell + 6), c, font=_font(20), fill=(40, 40, 45))
    return sheet


def make_sheets(cfg, work, out):
    out = Path(out); rv = out / 'review'; rv.mkdir(exist_ok=True)
    rd = rv / 'renders'
    names = cfg.get('display_names') or {}
    res = {}
    for kind, label in (('checker', '棋盘格'), ('texture', '贴图')):
        paths = [rd / f'{kind}_before_{i}.png' for i in range(1, 9)] + [rd / f'{kind}_after_{i}.png' for i in range(1, 9)]
        if all(p.exists() for p in paths):
            caps = [f'原 UV · 视角 {i}' for i in range(1, 9)] + [f'新 UV · 视角 {i}' for i in range(1, 9)]
            s = _grid(paths, caps, 8, 360, f'{label}对比：上排原始 UV，下排新的按部位 UV')
            s.save(rv / f'Compare_{kind}.png'); res[f'compare_{kind}'] = str(rv / f'Compare_{kind}.png')
    paths = [rd / f'checker_{k}_{i}.png' for k in ('before', 'previous', 'after') for i in range(1, 9)]
    if cfg.get('compare_previous_uv') and all(p.exists() for p in paths):
        caps = [f'{t} · 视角 {i}' for t in ('原始 UV', '上一版 UV', '本版按部位 UV') for i in range(1, 9)]
        s = _grid(paths, caps, 8, 360, '棋盘格三方对比：第一排原始，第二排上一版，第三排本版')
        s.save(rv / 'Compare_checker_3way.png'); res['compare_checker_3way'] = str(rv / 'Compare_checker_3way.png')
    objs = cfg['objects']
    g = [out / 'guides' / f'{o}_UV_Guide.png' for o in objs]
    if all(p.exists() for p in g):
        caps = [f'{names.get(o, o)}（{o}）' if o in names else o for o in objs]
        s = _grid(g, caps, 3, 900, '新 UV 图集总览（贴图 + 岛屿轮廓 + 部位编号）')
        s.save(out / 'UV_Atlas_Overview.png'); res['overview'] = str(out / 'UV_Atlas_Overview.png')
    ids = [out / 'guides' / f'{o}_UV_RegionID.png' for o in objs]
    if all(p.exists() for p in ids):
        s = _grid(ids, [f'{names.get(o, o)} 部位色块' for o in objs], 5, 520, '部位色块图（每种颜色 = 一个部位，可作选区蒙版）')
        s.save(rv / 'Region_ID_Sheet.png'); res['region_sheet'] = str(rv / 'Region_ID_Sheet.png')
    return res


# close-up views for judging texture edits, for a standing humanoid facing -Y (Blender), feet at the bottom of the
# bounding box. Units are the model height: camera location, look-at target (x, y relative to the box centre, z above
# the feet) and orthographic scale. config review.closeup_views overrides: {name: [loc, target, scale, label]}.
CLOSEUP_VIEWS = {
    'upper_front': ([0.147, -1.76, 0.675], [0.147, 0, 0.616], 0.44, '上身正面'),
    'back': ([0.176, 1.76, 0.587], [0.0, 0, 0.557], 0.763, '背面'),
    'thighs_front': ([0.352, -1.76, 0.364], [0.0, 0, 0.323], 0.352, '大腿正面'),
    'legs_side': ([1.467, -1.056, 0.205], [0, 0, 0.176], 0.528, '腿 · 侧面'),
    'legs_back': ([0.528, 1.525, 0.235], [0, 0, 0.205], 0.528, '腿 · 背面'),
    'shoes_low': ([0.939, -1.291, 0.352], [0, 0, 0.029], 0.352, '鞋 / 脚'),
    'side_hair': ([1.76, 0.117, 0.733], [0, 0.059, 0.704], 0.587, '侧面头发'),
}


def closeup_views(cfg):
    return (cfg.get('review') or {}).get('closeup_views') or CLOSEUP_VIEWS


def model_box(work, objects):
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for o in objects:
        co = np.load(Path(work) / 'inspect' / f'{o}.npz')['co']
        lo = np.minimum(lo, co.min(0)); hi = np.maximum(hi, co.max(0))
    return lo, hi


def closeup_job(cfg, work, out_dir, tex_dir=None):
    lo, hi = model_box(work, cfg['objects'])
    H = float(hi[2] - lo[2]); c = (lo + hi) / 2
    origin = np.array([c[0], c[1], lo[2]])
    views = {k: [list(map(float, origin + H * np.asarray(v[0], float))), list(map(float, origin + H * np.asarray(v[1], float))),
                 float(H * v[2])] for k, v in closeup_views(cfg).items()}
    return dict(objects=cfg['objects'], out_dir=str(out_dir), tex_dir=str(tex_dir) if tex_dir else None, views=views)


def make_closeup_sheet(cfg, rd, dst, notes):
    """Two columns per view: before the clean-up (migrated texture) / final texture."""
    rd = Path(rd); cv = closeup_views(cfg); views = list(cv)
    pairs = [(rd / 'closeup_before' / f'{v}.png', rd / 'closeup_after' / f'{v}.png') for v in views]
    if not all(a.exists() and b.exists() for a, b in pairs):
        return None
    cw, ch, gap, cap, top = 700, 500, 10, 30, 64
    sheet = Image.new('RGB', (2 * cw + 3 * gap, top + len(views) * (ch + cap + gap)), (238, 239, 242))
    d = ImageDraw.Draw(sheet)
    d.text((14, 14), '贴图清理前后对比（同一模型、同一 UV，只换了颜色贴图）', font=_font(30), fill=(30, 30, 35))
    for r, (v, (a, b)) in enumerate(zip(views, pairs)):
        y = top + r * (ch + cap + gap)
        for k, (p, lab) in enumerate(((a, '清理前（迁移后的原贴图）'), (b, notes))):
            x = gap + k * (cw + gap)
            im = Image.open(p).convert('RGB'); im.thumbnail((cw, ch), Image.LANCZOS)
            d.text((x, y + 2), f'{cv[v][3]} · {lab}', font=_font(19), fill=(40, 40, 45))
            sheet.paste(im, (x + (cw - im.width) // 2, y + cap))
    sheet.save(dst)
    return str(dst)
