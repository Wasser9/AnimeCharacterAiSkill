"""Quick orthographic painter's-algorithm renderer for previews (labels, checker, texture).

Not used for final deliverable renders; only for fast iteration and CP previews.
"""
import numpy as np, cv2

PALETTE = None


def palette(n, seed=7):
    rng = np.random.default_rng(seed)
    h = (np.arange(n) * 0.61803398875 + rng.random()) % 1.0
    s = 0.45 + 0.4 * rng.random(n)
    v = 0.75 + 0.25 * rng.random(n)
    hsv = np.stack([h * 179, s * 255, v * 255], 1).astype(np.uint8)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0].astype(np.float32) / 255


def view_matrix(yaw_deg, pitch_deg=0.0):
    """Camera looking at the character. yaw 0 = front (camera at -Y looking +Y)."""
    y = np.radians(yaw_deg); p = np.radians(pitch_deg)
    # camera right / up / forward in world coords (Z up)
    fwd = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), -np.sin(p)])  # from camera toward scene
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def render(co, F, face_rgb, yaw=0, pitch=0, size=800, bounds=None, light=None, shade=True, bg=(0.93, 0.94, 0.95)):
    right, up, fwd = view_matrix(yaw, pitch)
    P = np.stack([co @ right, co @ up, co @ fwd], 1)
    if bounds is None:
        lo = P[:, :2].min(0); hi = P[:, :2].max(0)
    else:
        lo, hi = bounds
    scale = (size * 0.92) / (hi - lo).max()
    c = (lo + hi) / 2
    xy = (P[:, :2] - c) * scale
    px = np.stack([size / 2 + xy[:, 0], size / 2 - xy[:, 1]], 1)
    tri = px[F]
    v0, v1, v2 = co[F[:, 0]], co[F[:, 1]], co[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    depth = P[F, 2].mean(1)
    if shade:
        L = -fwd + up * 0.6 + right * 0.3 if light is None else np.asarray(light)
        L = L / np.linalg.norm(L)
        lam = np.abs(n @ L)  # two-sided
        k = 0.35 + 0.65 * lam
        col = face_rgb * k[:, None]
    else:
        col = face_rgb
    img = np.empty((size, size, 3), np.float32); img[:] = bg
    order = np.argsort(-depth)  # far first
    pts = np.round(tri * 16).astype(np.int32)
    c8 = col
    for i in order:
        cv2.fillConvexPoly(img, pts[i], c8[i].tolist(), lineType=cv2.LINE_8, shift=4)
    return img


def save(path, img):
    a = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def grid(imgs, captions=None, cols=4, pad=6):
    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    out = np.ones((rows * (h + pad + 22), cols * (w + pad), 3), np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y0 = r * (h + pad + 22); x0 = c * (w + pad)
        out[y0:y0 + h, x0:x0 + w] = im
        if captions:
            cv2.putText(out, captions[i], (x0 + 6, y0 + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0.1, 0.1, 0.1), 1, cv2.LINE_AA)
    return out


def checker_colors(uv_face_centroid, n=24):
    """Flat per-face checker colour from per-face UV centroid."""
    a = np.floor(uv_face_centroid * n).astype(np.int64)
    k = (a[:, 0] + a[:, 1]) & 1
    c = np.where(k[:, None] == 1, np.array([0.27, 0.31, 0.35]), np.array([0.86, 0.91, 0.97]))
    return c.astype(np.float32)


def label_text(img, co, F, labels, names, yaw, pitch, size, bounds, min_faces=30):
    """Write label names at the projected centroid of each label (if facing camera)."""
    right, up, fwd = view_matrix(yaw, pitch)
    P = np.stack([co @ right, co @ up, co @ fwd], 1)
    lo, hi = bounds
    scale = (size * 0.92) / (hi - lo).max(); c = (lo + hi) / 2
    fc = co[F].mean(1)
    v0, v1, v2 = co[F[:, 0]], co[F[:, 1]], co[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    for l in np.unique(labels):
        m = (labels == l) & ((n @ fwd) < 0)
        if m.sum() < min_faces:
            continue
        p = fc[m].mean(0)
        q = np.array([p @ right, p @ up])
        x = size / 2 + (q[0] - c[0]) * scale; y = size / 2 - (q[1] - c[1]) * scale
        cv2.putText(img, str(names[l]), (int(x) - 20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, str(names[l]), (int(x) - 20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (1, 1, 1), 1, cv2.LINE_AA)
    return img
