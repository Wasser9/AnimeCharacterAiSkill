"""Stage 5: migrate the original basecolor (via UV_orig) onto the new UV layout.

Pure resampling of the source texture, no repainting:
  * rasterise the new UV triangles at `ss`x supersampling into a triangle-id buffer
  * per covered sample: barycentrics in the new triangle -> point in the same triangle of UV_orig
  * bilinear sample of the source texture in linear light
  * box-filter down to the target size using coverage weights (no background bleed)
  * island-aware padding: empty texels take the colour of the nearest island texel
"""
import numpy as np
import cv2
from scipy import ndimage


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def load_srgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = np.stack([im] * 3, -1)
    alpha = None
    if im.shape[2] == 4:
        alpha = im[..., 3]
        im = im[..., :3]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    scale = 65535.0 if im.dtype == np.uint16 else 255.0
    return im.astype(np.float32) / scale


def bilinear(img, x, y):
    H, W = img.shape[:2]
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)[:, None]; fy = (y - y0).astype(np.float32)[:, None]
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y0 + 1, 0, H - 1)
    return ((img[y0c, x0c] * (1 - fx) + img[y0c, x1c] * fx) * (1 - fy) +
            (img[y1c, x0c] * (1 - fx) + img[y1c, x1c] * fx) * fy)


def push_pull(img, w):
    """Pull-push hole filling (Gortler et al.): empty texels get a smooth, streak-free blend of the nearby island
    colours; covered texels are returned unchanged. img (H,W,C) linear, w (H,W) coverage weight 0..1."""
    levels = [(img * w[..., None], w)]
    while min(levels[-1][1].shape) > 1:
        c, ww = levels[-1]
        H, W = ww.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        cp = np.zeros((H2 * 2, W2 * 2, c.shape[2]), np.float32); cp[:H, :W] = c
        wp = np.zeros((H2 * 2, W2 * 2), np.float32); wp[:H, :W] = ww
        c2 = cp.reshape(H2, 2, W2, 2, -1).sum((1, 3))
        w2 = wp.reshape(H2, 2, W2, 2).sum((1, 3))
        # normalise to keep weights in [0,1] (premultiplied colour)
        sc = np.minimum(w2, 1.0) / np.maximum(w2, 1e-12)
        levels.append((c2 * sc[..., None], np.minimum(w2, 1.0)))
    c, ww = levels[-1]
    col = c / np.maximum(ww, 1e-12)[..., None]
    for c_l, w_l in reversed(levels[:-1]):
        H, W = w_l.shape
        up = cv2.resize(col, (col.shape[1] * 2, col.shape[0] * 2), interpolation=cv2.INTER_LINEAR)[:H, :W]
        if up.ndim == 2:
            up = up[..., None]
        here = c_l / np.maximum(w_l, 1e-12)[..., None]
        a = w_l[..., None]
        col = here * a + up * (1 - a)
    out = img.copy()
    hole = w <= 0
    out[hole] = col[hole]
    return out


def rasterize_ids(uv_new, size, draw_order):
    """uv_new (nf,3,2) in [0,1]; returns int32 id buffer (size x size), -1 = empty."""
    buf = np.full((size, size), -1, np.int32)
    pts = np.stack([uv_new[..., 0] * size - 0.5, (1 - uv_new[..., 1]) * size - 0.5], -1)
    pts = np.round(pts * 16).astype(np.int32)
    for f in draw_order:
        cv2.fillConvexPoly(buf, pts[f], int(f), lineType=cv2.LINE_8, shift=4)
    return buf


def transfer(uv_new, uv_old, src_srgb, size=4096, ss=2, pad_mode='pushpull', satellites=None, tile=512, pad_px=16):
    nf = len(uv_new)
    S = size * ss
    a = uv_new[:, 0]; e1 = uv_new[:, 1] - a; e2 = uv_new[:, 2] - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    good = np.abs(det) > 1e-14
    sat = np.zeros(nf, bool)
    if satellites is not None and len(satellites):
        sat[satellites] = True
    order = np.r_[np.nonzero(good & sat)[0], np.nonzero(good & ~sat)[0]]
    ids = rasterize_ids(uv_new, S, order)
    inv = np.zeros((nf, 2, 2))
    inv[good, 0, 0] = e2[good, 1] / det[good]; inv[good, 0, 1] = -e2[good, 0] / det[good]
    inv[good, 1, 0] = -e1[good, 1] / det[good]; inv[good, 1, 1] = e1[good, 0] / det[good]
    oa = uv_old[:, 0]; o1 = uv_old[:, 1] - oa; o2 = uv_old[:, 2] - oa
    src_lin = srgb_to_linear(src_srgb).astype(np.float32)
    Hs, Ws = src_lin.shape[:2]
    acc = np.zeros((3, size * size), np.float32)
    wgt = np.zeros(size * size, np.float32)
    assert tile % ss == 0
    for y0 in range(0, S, tile):
        blk = ids[y0:y0 + tile]
        yy, xx = np.nonzero(blk >= 0)
        if not len(yy):
            continue
        f = blk[yy, xx]
        pu = (xx + 0.5) / S; pv = 1.0 - (y0 + yy + 0.5) / S
        d = np.stack([pu - a[f, 0], pv - a[f, 1]], 1)
        b1 = inv[f, 0, 0] * d[:, 0] + inv[f, 0, 1] * d[:, 1]
        b2 = inv[f, 1, 0] * d[:, 0] + inv[f, 1, 1] * d[:, 1]
        # clamp to the triangle (border samples of an island)
        b1 = np.clip(b1, 0, 1); b2 = np.clip(b2, 0, 1)
        s_ = b1 + b2; over = s_ > 1
        b1[over] /= s_[over]; b2[over] /= s_[over]
        ou = oa[f, 0] + b1 * o1[f, 0] + b2 * o2[f, 0]
        ov = oa[f, 1] + b1 * o1[f, 1] + b2 * o2[f, 1]
        mx = (ou * Ws - 0.5).astype(np.float32); my = ((1 - ov) * Hs - 0.5).astype(np.float32)
        col = bilinear(src_lin, mx, my)
        ty = (y0 + yy) // ss; tx = xx // ss
        lo_ = (y0 // ss) * size
        idx = ty * size + tx - lo_
        n = min((tile // ss) * size, size * size - lo_)
        wgt[lo_:lo_ + n] += np.bincount(idx, minlength=n)
        for c in range(3):
            acc[c, lo_:lo_ + n] += np.bincount(idx, weights=col[:, c], minlength=n)
    cov = (wgt > 0).reshape(size, size)
    out = np.zeros((size, size, 3), np.float32)
    w_ = np.maximum(wgt, 1e-12)
    for c in range(3):
        out[..., c] = (acc[c] / w_).reshape(size, size)
    out[~cov] = 0
    # island-aware padding: nearest covered texel
    if pad_mode == 'pushpull':
        bgc = out[cov].mean(0) if cov.any() else np.zeros(3, np.float32)
        out = push_pull(out, cov.astype(np.float32))
        if pad_px is not None:
            # smooth fill for pad_px texels around every island, then a soft fade into one flat background colour
            dist = ndimage.distance_transform_edt(~cov)
            w = np.clip((dist - pad_px) / max(pad_px, 1), 0, 1).astype(np.float32)[..., None]
            out = out * (1 - w) + bgc[None, None, :] * w
    elif pad_mode == 'nearest':
        dist, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
        bgc = out[cov].mean(0) if cov.any() else np.zeros(3, np.float32)
        out = out[iy, ix]
        if pad_px is not None:
            # islands are extended by pad_px texels (mip safety); the rest of the sheet is one flat colour
            out[dist > pad_px] = bgc
    del ids, acc, wgt, src_lin
    return linear_to_srgb(out).astype(np.float32), cov, None


def save_png(path, srgb, bits=8):
    if bits == 16:
        a = np.clip(srgb * 65535 + 0.5, 0, 65535).astype(np.uint16)
    else:
        a = np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
