"""Layout / guide drawing helpers."""
import numpy as np, cv2
from render_util import palette


def draw_layout(path, uv_tris, face_region, regions, face_chart, size=2048, zones=None, title='', tex=None):
    """uv_tris (nf,3,2) in [0,1]."""
    if tex is not None:
        img = cv2.resize(tex, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        img = img * 0.85 + 0.15
    else:
        img = np.full((size, size, 3), 0.97, np.float32)
    pal = palette(len(regions) + 1, 11)
    pts = np.round(np.stack([uv_tris[..., 0] * size, (1 - uv_tris[..., 1]) * size], -1) * 16).astype(np.int32)
    if tex is None:
        for f in range(len(pts)):
            r = face_region[f]
            cv2.fillConvexPoly(img, pts[f], (pal[r] * 0.85 + 0.15).tolist() if r >= 0 else (1, 0, 0), cv2.LINE_8, 4)
    # chart outlines: draw boundary edges (edges used once within a chart)
    if zones is not None:
        for g, (x0, y0, x1, y1) in zones.items():
            cv2.rectangle(img, (int(x0 * size), int(y0 * size)), (int(x1 * size), int(y1 * size)), (0.75, 0.75, 0.8), 1)
            cv2.putText(img, g, (int(x0 * size) + 4, int(y0 * size) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0.5, 0.5, 0.6), 1, cv2.LINE_AA)
    e = np.concatenate([np.stack([pts[:, 0], pts[:, 1]], 1), np.stack([pts[:, 1], pts[:, 2]], 1), np.stack([pts[:, 2], pts[:, 0]], 1)])
    fc = np.tile(face_chart, 3)
    key = np.sort(np.stack([e[:, 0, 0] * 100003 + e[:, 0, 1], e[:, 1, 0] * 100003 + e[:, 1, 1]], 1), 1)
    k2 = np.c_[key, fc]
    _, inv, cnt = np.unique(k2, axis=0, return_inverse=True, return_counts=True)
    bd = cnt[inv.ravel()] == 1
    for a, b in e[bd]:
        cv2.line(img, tuple(a), tuple(b), (0.1, 0.1, 0.15), 1, cv2.LINE_AA, 4)
    # labels at region centroid of the largest chart
    for ri, r in enumerate(regions):
        fs = np.nonzero(face_region == ri)[0]
        if not len(fs):
            continue
        ch = np.bincount(face_chart[fs]).argmax()
        f2 = fs[face_chart[fs] == ch]
        c = uv_tris[f2].reshape(-1, 2).mean(0)
        x, y = int(c[0] * size), int((1 - c[1]) * size)
        txt = r['name']
        cv2.putText(img, txt, (x - 4 * len(txt), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x - 4 * len(txt), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0.05, 0.05, 0.1), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.1, 0.1, 0.1), 2, cv2.LINE_AA)
    a = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
