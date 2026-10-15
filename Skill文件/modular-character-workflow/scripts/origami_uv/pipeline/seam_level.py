"""Seam levelling (after Lempitsky & Ivanov 2007): a smooth additive colour correction per uv-vertex that makes
colours agree across 3D seams, applied to a texture whose islands were cleaned independently.

Constraints (least squares, one solve per channel, linear light):
  * seams: for every welded mesh vertex shared by two charts, (c_i + g_i) - (c_j + g_j) ~ 0, weighted by how
    continuous the ORIGINAL texture was there (real material borders such as skin|sleeve get ~0 weight)
  * smoothness: g_a - g_b ~ 0 along every uv edge inside a chart (so the correction spreads smoothly)
  * a tiny pull of g towards 0 (fixes the free offset)
The per-uv-vertex correction is interpolated over the triangles and added to every covered texel.
"""
import pickle
from pathlib import Path
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from uvcore import Mesh
from transfer import rasterize_ids


def masked_blur(img, mask, sigma):
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-6)


def bilinear(img, x, y):
    H, W = img.shape[:2]
    x = np.clip(x, 0, W - 1.001); y = np.clip(y, 0, H - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
            (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def seam_level(work, obj, lin, ref_lin, cov, sigma=4.0, w_seam=40.0, w_smooth=1.0, w_anchor_visible=3e-3,
               w_anchor_hidden=1e-4, visible_exposure=0.12, tol=0.04, inner=0.35):
    """lin, ref_lin: (S,S,3) linear float32 textures (cleaned, original). Returns corrected copy and stats."""
    work = Path(work)
    m = Mesh(work / 'inspect' / f'{obj}.npz')
    u = pickle.load(open(work / 'unwrap' / f'{obj}.pkl', 'rb'))
    p = pickle.load(open(work / 'pack' / f'{obj}.pkl', 'rb'))
    S = lin.shape[0]
    final = p['final']; uvv = u['res']['corner_uvv']; fc = u['res']['face_chart']
    sat = np.zeros(m.nf, bool); sat[u['res']['satellites']] = True
    fs = np.nonzero(~sat)[0]
    ids, Fl = np.unique(uvv[fs], return_inverse=True); Fl = Fl.reshape(-1, 3)
    n = len(ids)
    P = np.zeros((n, 2)); P[Fl.ravel()] = final[fs].reshape(-1, 2)
    W = np.zeros(n, np.int64); W[Fl.ravel()] = m.Fw[fs].ravel()
    C = np.zeros(n, np.int64); C[Fl.ravel()] = np.repeat(fc[fs], 3)
    # visible surfaces keep their colour (correction stays local to the seam, ~15 edges wide);
    # hidden ones (inside shoes / clothes / hair) adapt to their visible neighbours
    E = np.load(work / 'exposure.npz')[obj]
    vexp = np.zeros(n); np.maximum.at(vexp, Fl.ravel(), np.repeat(E[fs], 3))
    w_anchor = np.where(vexp > visible_exposure, w_anchor_visible, w_anchor_hidden)
    # sample slightly inside the island: move each uv-vertex towards the centroid of its incident triangles
    cen = np.zeros((n, 2)); cnt = np.zeros(n)
    tc = final[fs].mean(1)
    np.add.at(cen, Fl.ravel(), np.repeat(tc, 3, 0)); np.add.at(cnt, Fl.ravel(), 1)
    cen /= np.maximum(cnt, 1)[:, None]
    Q = P + inner * (cen - P)
    px = Q[:, 0] * S - 0.5; py = (1 - Q[:, 1]) * S - 0.5
    cs = bilinear(masked_blur(lin, cov, sigma), px, py)
    os_ = bilinear(masked_blur(ref_lin, cov, sigma), px, py)
    # seam pairs: uv-vertices of the same welded vertex in different charts
    order = np.argsort(W, kind='stable'); Ws = W[order]
    st = np.r_[0, np.nonzero(np.diff(Ws))[0] + 1, len(Ws)]
    pi, pj = [], []
    for a, b in zip(st[:-1], st[1:]):
        if b - a < 2:
            continue
        grp = order[a:b]
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                if C[grp[x]] != C[grp[y]]:
                    pi.append(grp[x]); pj.append(grp[y])
    pi = np.asarray(pi, np.int64); pj = np.asarray(pj, np.int64)
    dorig = np.abs(os_[pi] - os_[pj]).max(1)
    ws = w_seam * np.exp(-(dorig / tol) ** 2)
    # smoothness edges
    e = np.sort(np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]), 1)
    e = np.unique(e, axis=0)
    rows_i = np.r_[pi, e[:, 0], np.arange(n)]
    rows_j = np.r_[pj, e[:, 1], np.full(n, -1)]
    wts = np.r_[ws, np.full(len(e), w_smooth), w_anchor]
    R = len(wts)
    r_idx = np.arange(R)
    A = sp.coo_matrix((np.r_[np.ones(R), -np.ones(R - n)], (np.r_[r_idx, r_idx[:R - n]], np.r_[rows_i, rows_j[:R - n]])),
                      shape=(R, n)).tocsr()
    Wd = sp.diags(wts)
    M = (A.T @ Wd @ A).tocsc()
    solve = sla.factorized(M)
    G = np.zeros((n, 3))
    for ch in range(3):
        b = np.zeros(R)
        b[:len(pi)] = -(cs[pi, ch] - cs[pj, ch])
        G[:, ch] = solve(A.T @ (wts * b))
    before = float((np.sqrt(ws / w_seam) * np.abs(cs[pi] - cs[pj]).max(1)).sum() / max(np.sqrt(ws / w_seam).sum(), 1e-9))
    after = float((np.sqrt(ws / w_seam) * np.abs(cs[pi] + G[pi] - cs[pj] - G[pj]).max(1)).sum() / max(np.sqrt(ws / w_seam).sum(), 1e-9))
    # rasterise the correction
    tri = np.stack([P[Fl[:, 0]], P[Fl[:, 1]], P[Fl[:, 2]]], 1)
    idb = rasterize_ids(tri, S, np.arange(len(Fl)))
    out = lin.copy()
    a0 = tri[:, 0]; e1 = tri[:, 1] - a0; e2 = tri[:, 2] - a0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    for y0 in range(0, S, 512):
        blk = idb[y0:y0 + 512]
        yy, xx = np.nonzero(blk >= 0)
        if not len(yy):
            continue
        f = blk[yy, xx]
        pu = (xx + 0.5) / S; pv = 1.0 - (y0 + yy + 0.5) / S
        dx = pu - a0[f, 0]; dy = pv - a0[f, 1]
        b1 = np.clip((dx * e2[f, 1] - dy * e2[f, 0]) / det[f], 0, 1)
        b2 = np.clip((e1[f, 0] * dy - e1[f, 1] * dx) / det[f], 0, 1)
        sm = b1 + b2; over = sm > 1; b1[over] /= sm[over]; b2[over] /= sm[over]
        g = (1 - b1 - b2)[:, None] * G[Fl[f, 0]] + b1[:, None] * G[Fl[f, 1]] + b2[:, None] * G[Fl[f, 2]]
        out[y0 + yy, xx] = np.clip(out[y0 + yy, xx] + g.astype(np.float32), 0, 1)
    return out, dict(seam_pairs=int(len(pi)), constrained=int((ws > 0.1 * w_seam).sum()),
                     visible_vertices=int((vexp > visible_exposure).sum()), vertices=int(n),
                     seam_diff_before=round(before, 4), seam_diff_after=round(after, 4),
                     max_correction=round(float(np.abs(G).max()), 4))
