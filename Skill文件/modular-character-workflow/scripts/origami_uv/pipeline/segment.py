"""Semantic segmentation of each character mesh into UV panels ("origami" regions).

Each object function returns (labels per face, regions list). A region is a dict:
  name   : stable English id (used in guides / UV_panel attribute)
  cn     : Chinese display name
  group  : packing zone key (regions of one group are packed next to each other)
  prio   : texel density multiplier (1 = atlas average)
  up     : 3D direction that should point up (+V) in the UV island
  seam   : 3D direction where seams are preferred (hidden side), or None (use exposure)
  rotate : allow 90 degree rotation while packing (small parts only)
Faces with label -1 are "satellites": tiny non-manifold debris that inherits the host's UVs.

Which rules run for an object: config `segmenters: {<object>: <key>}` with a key of SEGMENTERS. 'generic' (the default)
works for any character (one panel per connected piece x body part of the dominant deform bone, left/right split);
'legs', 'body', 'head', 'front_hair', 'back_hair' are the hand-written rules of the example character (five modules,
thresholds tuned to that model) and serve as templates for character-specific rules.
"""
import re
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg

Z = np.array([0, 0, 1.0]); Y = np.array([0, 1.0, 0]); X = np.array([1.0, 0, 0])


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


class Seg:
    def __init__(self, mesh, exposure):
        self.m = mesh
        self.E = exposure
        self.regions = []
        self.index = {}
        self.labels = np.full(mesh.nf, -2, np.int64)
        A = mesh.face_adjacency()
        self.A = A + A.T
        self.ncomp, self.comp = cg.connected_components(self.A, directed=False)
        self.csize = np.bincount(self.comp)
        self.carea = np.bincount(self.comp, mesh.area)

    def R(self, name, cn, group, prio=1.0, up=Z, seam=None, rotate=False, closed_axis=None):
        if name not in self.index:
            self.index[name] = len(self.regions)
            self.regions.append(dict(name=name, cn=cn, group=group, prio=float(prio), up=list(map(float, up)),
                                     seam=None if seam is None else list(map(float, unit(seam))), rotate=bool(rotate),
                                     closed_axis=None if closed_axis is None else list(map(float, closed_axis))))
        return self.index[name]

    def comp_faces(self, k):
        return np.nonzero(self.comp == k)[0]

    def comps_by_size(self):
        return list(np.argsort(-self.csize))

    def dom_bone(self, faces):
        m = self.m
        w = m.W[m.F[faces]].mean(1)
        return w.argmax(1)

    def bone_weight(self, faces, names):
        m = self.m
        idx = [m.groups.index(n) for n in names if n in m.groups]
        return m.W[m.F[faces]][:, :, idx].sum(2).mean(1)

    def field(self, fs, score, iters=40, keep=0.0):
        """Diffuse a per-face score over the surface (restricted to faces fs) -> smooth iso-line splits.
        keep: weight of the original score in each Jacobi step (data term)."""
        m = self.m
        fs = np.asarray(fs)
        loc = -np.ones(m.nf, np.int64); loc[fs] = np.arange(len(fs))
        A = self.A.tocoo()
        ok = (loc[A.row] >= 0) & (loc[A.col] >= 0)
        r = loc[A.row[ok]]; c = loc[A.col[ok]]
        w = m.area[fs]
        W = sp.coo_matrix((w[c], (r, c)), shape=(len(fs), len(fs))).tocsr()
        deg = np.asarray(W.sum(1)).ravel()
        x0 = np.asarray(score, float).copy(); x = x0.copy()
        for _ in range(iters):
            x = (W @ x + (w + keep * deg) * x + keep * deg * x0) / (deg + w + 2 * keep * deg + 1e-30)
        return x

    def assign(self, fs, scores, labels, iters=60):
        """Multi-label smooth assignment: diffuse each label's score and take argmax."""
        S = np.stack([self.field(fs, scores[:, k], iters) for k in range(scores.shape[1])], 1)
        self.labels[fs] = np.asarray(labels)[S.argmax(1)]

    def split(self, fs, score, a, b, iters=40):
        """label faces fs with a where the diffused score > 0 else b."""
        x = self.field(fs, np.clip(score, -1, 1), iters)
        self.labels[fs] = np.where(x > 0, a, b)

    # ---------------------------------------------------------------- cleanup
    def satellites(self, max_faces=40, max_frac=2e-4):
        """Tiny components that only touch other geometry through shared vertices."""
        m = self.m
        tot = m.area.sum()
        big_verts = np.zeros(m.nvw, bool)
        small = (self.csize <= max_faces) & (self.carea <= max_frac * tot)
        bigf = ~small[self.comp]
        big_verts[m.Fw[bigf].ravel()] = True
        # any tiny piece (shared vertices or floating debris) borrows UVs from the nearest host face
        sat = small[self.comp]
        self.labels[sat] = -1
        return sat

    def smooth(self, iters=6, mask=None):
        """Area-weighted majority filter on the manifold face graph (labels >= 0 only)."""
        m = self.m
        A = self.A.tocoo()
        for _ in range(iters):
            lab = self.labels
            ok = (lab[A.row] >= 0) & (lab[A.col] >= 0)
            if mask is not None:
                ok &= mask[A.row] & mask[A.col]
            r = A.row[ok]; c = A.col[ok]
            nl = len(self.regions)
            votes = sp.coo_matrix((m.area[c], (r, lab[c])), shape=(m.nf, nl)).tocsr()
            votes = votes + sp.coo_matrix((m.area * 1.5, (np.arange(m.nf), np.maximum(lab, 0))), shape=(m.nf, nl)).tocsr()
            new = np.asarray(votes.argmax(1)).ravel()
            upd = lab >= 0
            if mask is not None:
                upd &= mask
            self.labels = np.where(upd, new, lab)

    def denotch(self, iters=6):
        """Remove staircase teeth: a face whose 2+ edge-neighbours share another label switches to it.
        Each flip replaces two boundary edges by one, so boundary length strictly decreases."""
        m = self.m
        A = self.A.tocoo()
        r, c = A.row, A.col
        for _ in range(iters):
            lab = self.labels
            ok = (lab[r] >= 0) & (lab[c] >= 0) & (lab[r] != lab[c])
            if not ok.any():
                return
            rr = r[ok]; cl = lab[c[ok]]
            key = rr * (len(self.regions) + 1) + cl
            uk, cnt = np.unique(key, return_counts=True)
            hit = cnt >= 2
            f = uk[hit] // (len(self.regions) + 1); nl = uk[hit] % (len(self.regions) + 1)
            if not len(f):
                return
            # update half of the candidates per pass (checkerboard by face index parity) to avoid oscillation
            sel = (f + _) % 2 == 0
            new = self.labels.copy(); new[f[sel]] = nl[sel]
            self.labels = new

    def enforce_connectivity(self, min_frac=0.08, min_faces=60):
        """Every region keeps its big connected pieces; small islands join the neighbour they share most edges with."""
        m = self.m
        for _ in range(8):
            lab = self.labels
            A = self.A.tocoo()
            same = lab[A.row] == lab[A.col]
            g = sp.coo_matrix((np.ones(same.sum()), (A.row[same], A.col[same])), shape=(m.nf, m.nf))
            n, pc = cg.connected_components(g, directed=False)
            parea = np.bincount(pc, m.area)
            pfaces = np.bincount(pc)
            plabel = np.zeros(n, np.int64); plabel[pc] = lab
            # largest piece per label
            best = {}
            for p in np.argsort(-parea):
                l = plabel[p]
                if l < 0:
                    continue
                best.setdefault(l, parea[p])
            small = np.array([plabel[p] >= 0 and parea[p] < min_frac * best[plabel[p]] and pfaces[p] < 4000 or
                              (plabel[p] >= 0 and pfaces[p] < min_faces and parea[p] < best[plabel[p]])
                              for p in range(n)])
            if not small.any():
                return
            # neighbour labels across boundary edges
            diff = ~same & (lab[A.row] >= 0) & (lab[A.col] >= 0)
            r = A.row[diff]; c = A.col[diff]
            changed = False
            sp_r = pc[r]
            for p in np.nonzero(small)[0]:
                sel = sp_r == p
                if not sel.any():
                    continue  # isolated component: keep own label
                nb = lab[c[sel]]
                vals, cnt = np.unique(nb, return_counts=True)
                self.labels[pc == p] = vals[np.argmax(cnt)]
                changed = True
            if not changed:
                return

    def open_regions(self, width=0.012, min_area=0.02):
        """Morphological opening per region on the surface: peninsulas narrower than ~2*width are
        handed to the neighbouring region (keeps big panels free of thin spikes)."""
        m = self.m
        A = self.A.tocsr()
        lab = self.labels
        nf = m.nf
        for l in range(len(self.regions)):
            fs = np.nonzero(lab == l)[0]
            if not len(fs) or m.area[fs].sum() < min_area:
                continue
            inr = np.zeros(nf, bool); inr[fs] = True
            # distance (metres, via face centroids) from the region border
            sub = A[fs][:, fs].tocoo()
            w = np.linalg.norm(m.cent[fs[sub.row]] - m.cent[fs[sub.col]], axis=1) + 1e-9
            G = sp.coo_matrix((w, (sub.row, sub.col)), shape=(len(fs), len(fs))).tocsr()
            deg_in = np.asarray((A[fs][:, fs] > 0).sum(1)).ravel()
            deg_all = np.asarray((A[fs] > 0).sum(1)).ravel()
            border = np.nonzero(deg_in < deg_all)[0]
            if not len(border):
                continue
            d = cg.dijkstra(G, directed=False, indices=border, min_only=True)
            core = np.nonzero(d >= width)[0]
            if not len(core):
                continue
            d2 = cg.dijkstra(G, directed=False, indices=core, min_only=True)
            lost = fs[d2 > width * 1.05]
            if not len(lost):
                continue
            # hand lost faces to the most common foreign neighbour label (iteratively, outside-in)
            for _ in range(50):
                if not len(lost):
                    break
                keep = []
                for f in lost:
                    nb = A.indices[A.indptr[f]:A.indptr[f + 1]]
                    nl = self.labels[nb]
                    nl = nl[(nl >= 0) & (nl != l)]
                    if len(nl):
                        vals, cnt = np.unique(nl, return_counts=True)
                        self.labels[f] = vals[np.argmax(cnt)]
                    else:
                        keep.append(f)
                if len(keep) == len(lost):
                    break
                lost = np.array(keep, np.int64)

    def finish(self, denotch=6, straight=True, opening=True):
        assert (self.labels >= -1).all(), 'unlabelled faces remain'
        if opening:
            self.open_regions()
        if denotch:
            self.denotch(denotch)
            self.enforce_connectivity()
        if straight:
            from straighten import straighten
            self.labels = straighten(self.m, self.labels)
            self.denotch(2)
            self.enforce_connectivity()
        return self.labels, self.regions


def radial_out(m, faces, center, axis=Z):
    """dot(normal, direction from axis through center to face) ; >0 means facing outward."""
    p = m.cent[faces] - center
    p = p - np.outer(p @ axis, axis)
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-12
    return (m.normal[faces] * p).sum(1)


def side(x):
    return 'L' if x > 0 else 'R'


def cn_side(x):
    return '左' if x > 0 else '右'


# ============================================================================ objects

def seg_legs(m, E):
    s = Seg(m, E)
    s.satellites()
    # heel collar height of each shoe (largest low components)
    rim = {}
    for k in s.comps_by_size()[:6]:
        fs = s.comp_faces(k); c = m.cent[fs].mean(0)
        if c[2] < 0.2 and len(fs) > 5000:
            p = m.cent[fs]
            heel = p[:, 1] > np.quantile(p[:, 1], 0.85)
            rim[side(c[0])] = float(np.quantile(p[heel, 2], 0.97))
    for k in s.comps_by_size():
        fs = s.comp_faces(k)
        if (s.labels[fs] == -1).all():
            continue
        c = m.cent[fs].mean(0); sd = side(c[0]); cs = cn_side(c[0]); sx = np.sign(c[0])
        if c[2] > 0.2 and len(fs) > 1000:  # leg (thigh skin + sock)
            top_c = m.cent[fs][m.cent[fs, 2] > 0.5].mean(0)
            inward = radial_out(m, fs, np.array([top_c[0], top_c[1], 0])) < 0
            cap = (m.cent[fs, 2] > 0.55) & (inward | (m.normal[fs, 2] > 0.6))
            leg = s.R(f'Leg_{sd}', f'{cs}腿（大腿+袜子）', 'legs', 1.0, Z, seam=-sx * 0.35 * X + Y)
            foot = s.R(f'Foot_{sd}', f'{cs}脚（鞋内）', 'feet', 0.6, -Y, seam=Z, rotate=True)
            capr = s.R(f'LegTop_{sd}', f'{cs}腿 顶部封口（裙内）', 'feet', 0.3, Y, rotate=True)
            # leg / foot: horizontal cut just below the heel collar of the matching shoe
            zz = m.cent[fs, 2]
            zcut = rim[sd] - 0.006
            s.labels[fs] = np.where(zz < zcut, foot, np.where(cap, capr, leg))
        elif len(fs) < 3000 and len(fs) > 500:  # platform sole slab (bottom / side band / hidden top)
            nz = m.normal[fs, 2]
            bot = s.R(f'SoleBottom_{sd}', f'{cs}鞋底 底面', 'soles', 0.6, -Y)
            top = s.R(f'SoleTop_{sd}', f'{cs}鞋底 顶面（被鞋遮住）', 'soles', 0.25, -Y, rotate=True)
            band = s.R(f'SoleSide_{sd}', f'{cs}鞋底 侧边', 'soles', 1.0, Z, seam=Y, rotate=True)
            s.labels[fs] = np.where(nz < -0.5, bot, np.where(nz > 0.5, top, band))
        else:
            p = m.cent[fs]; z = p[:, 2]; nz = m.normal[fs, 2]
            # shoe frame: long axis of the footprint
            cxy = p[:, :2].mean(0)
            cov = np.cov((p[:, :2] - cxy).T); w_, v_ = np.linalg.eigh(cov)
            across = v_[:, 0] * np.sign(v_[0, 0] * sx)  # points to the outer (lateral) side
            lateral = (p[:, :2] - cxy) @ across
            outR = s.R(f'ShoeOut_{sd}', f'{cs}鞋 外侧鞋面', 'shoes', 1.0, Z)
            inR = s.R(f'ShoeIn_{sd}', f'{cs}鞋 内侧鞋面', 'shoes', 1.0, Z)
            lining = s.R(f'ShoeLining_{sd}', f'{cs}鞋 内壁', 'shoes_inner', 0.45, Z, seam=Y, rotate=True)
            ins = s.R(f'Insole_{sd}', f'{cs}鞋垫', 'shoes_inner', 0.45, -Y, rotate=True)
            r = p[:, :2] - cxy; r = r - np.outer(r @ v_[:, 1], v_[:, 1]) * 0.6
            r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
            facing_in = -(m.normal[fs, :2] * r).sum(1)
            score = np.clip((0.12 - E[fs]) / 0.12, -1, 1) + 0.8 * np.clip(facing_in, -1, 1) * (nz < 0.5)
            inner_x = s.field(fs, np.clip(score, -1, 1), 150) > 0
            s.labels[fs] = np.where(s.field(fs, np.clip(lateral / 0.01, -1, 1), 20) > 0, outR, inR)
            fi = fs[inner_x]
            sole_s = np.clip((nz[inner_x] - 0.55) * 4, -1, 1) * (z[inner_x] < 0.115) - (z[inner_x] >= 0.115)
            s.labels[fi] = np.where(s.field(fi, sole_s, 20) > 0, ins, lining)
    s.smooth(8)
    s.enforce_connectivity()
    return s.finish()


def seg_body(m, E):
    s = Seg(m, E)
    s.satellites()
    bi = {n: i for i, n in enumerate(m.groups)}
    for k in s.comps_by_size():
        fs = s.comp_faces(k)
        if (s.labels[fs] == -1).all():
            continue
        c = m.cent[fs].mean(0); n = len(fs); sx = np.sign(c[0]); sd = side(c[0]); cs = cn_side(c[0])
        hb = s.bone_weight(fs, ['hand_l', 'hand_r']).mean()
        if n > 10000:  # sweater body + sleeves
            p = m.cent[fs]
            arm = np.abs(p[:, 0]) > 0.215
            # front/back split along the mid plane of each height slice
            zb = np.clip(((p[:, 2] - p[:, 2].min()) / 0.02).astype(int), 0, 200)
            ymid = np.zeros(zb.max() + 1)
            for b in np.unique(zb):
                sel = (zb == b) & ~arm
                if sel.sum() > 10:
                    ymid[b] = 0.5 * (p[sel, 1].min() + p[sel, 1].max())
                else:
                    ymid[b] = np.nan
            idx = np.arange(len(ymid)); okb = np.isfinite(ymid)
            ymid = np.interp(idx, idx[okb], ymid[okb])
            sl_l = s.R('Sleeve_L', '左袖', 'sleeves', 1.0, -X, seam=-Z)
            sl_r = s.R('Sleeve_R', '右袖', 'sleeves', 1.0, X, seam=-Z)
            tf = s.R('Sweater_Front', '毛衣 前片', 'torso', 1.0, Z, seam=X)
            tb = s.R('Sweater_Back', '毛衣 后片', 'torso', 0.8, Z, seam=X)
            ax_ = np.abs(p[:, 0])
            s_arm = np.clip((ax_ - 0.215) / 0.015, -1, 1)
            s_fb = np.clip((ymid[zb] - p[:, 1]) / 0.01, -1, 1)
            xs_ = np.clip(p[:, 0] / 0.05, -1, 1)
            sc = np.stack([s_arm + xs_, s_arm - xs_, -s_arm + s_fb, -s_arm - s_fb], 1)
            s.assign(fs, sc, [sl_l, sl_r, tf, tb], iters=150)
            # closed end caps hidden inside the cuffs
            cap = (ax_ > 0.40) & (m.normal[fs, 0] * np.sign(p[:, 0]) > 0.75)
            for sd, cs, sg in (('L', '左', 1), ('R', '右', -1)):
                sel = cap & (np.sign(p[:, 0]) == sg)
                if sel.sum():
                    s.labels[fs[sel]] = s.R(f'SleeveEnd_{sd}', f'{cs}袖 内端封口', 'misc', 0.3, Z, rotate=True)
        elif hb > 0.5 and n > 4000:  # hand
            nz = m.normal[fs, 2]
            top = s.R(f'Hand_{sd}_Back', f'{cs}手 手背', 'hands', 0.9, -sx * X, rotate=False)
            bot = s.R(f'Hand_{sd}_Palm', f'{cs}手 手心', 'hands', 0.9, -sx * X, rotate=False)
            s.labels[fs] = np.where(nz >= 0, top, bot)
        elif hb > 0.3 or (abs(c[0]) > 0.4 and n > 1000):  # ribbed cuff
            s.labels[fs] = s.R(f'Cuff_{sd}', f'{cs}袖口罗纹', 'sleeves', 1.0, -sx * X, seam=-Z)
        elif c[2] < 0.72 and n > 1000:  # pleated skirt
            out = radial_out(m, fs, np.array([0, 0.0, 0]))
            p = m.cent[fs]
            outer = out > -0.2
            front = p[:, 1] < 0.0
            sf = s.R('Skirt_Front', '裙子 前片', 'skirt', 1.0, Z, seam=X)
            sb = s.R('Skirt_Back', '裙子 后片', 'skirt', 0.85, Z, seam=X)
            si = s.R('Skirt_Inner', '裙子 内侧', 'skirt', 0.45, Z, seam=Y)
            s.labels[fs] = np.where(outer, np.where(front, sf, sb), si)
        elif abs(c[2] - 0.735) < 0.02 and n < 400:
            s.labels[fs] = s.R('Sweater_Bottom', '毛衣 底面', 'misc', 0.35, -Y, rotate=True)
        elif c[2] > 1.12 and n > 3000:  # scarf ring around the neck
            out = radial_out(m, fs, np.array([0, -0.01, 0]))
            o = s.R('Scarf_Wrap_Outer', '围巾 绕颈 外侧', 'scarf', 1.0, Z, seam=Y)
            i = s.R('Scarf_Wrap_Inner', '围巾 绕颈 内侧', 'scarf', 0.5, Z, seam=Y)
            s.labels[fs] = np.where(out > -0.1, o, i)
        elif c[2] > 1.05 and n > 1500:  # shirt collar / neck opening
            s.labels[fs] = s.R('Collar', '衬衫领口', 'collar', 1.0, Z, seam=Y)
        elif s.bone_weight(fs, ['scarf_01', 'scarf_02']).mean() > 0.3 and n > 900:  # hanging scarf end
            fr = m.normal[fs, 1] < 0.15
            a = s.R('Scarf_Tail_Front', '围巾 垂片 正面', 'scarf', 1.0, Z)
            b = s.R('Scarf_Tail_Back', '围巾 垂片 背面', 'scarf', 0.6, Z)
            s.labels[fs] = np.where(fr, a, b)
        elif c[2] < 0.9 and 400 < n < 900:  # tassels
            order = sorted([kk for kk in range(s.ncomp) if 400 < s.csize[kk] < 900 and m.cent[s.comp == kk, 2].mean() < 0.9],
                           key=lambda kk: m.cent[s.comp == kk, 0].mean())
            t = order.index(k) + 1
            s.labels[fs] = s.R(f'Tassel_{t}', f'围巾流苏 {t}', 'scarf_small', 1.0, Z, seam=Y)
        elif 800 < n < 900 or (abs(c[0]) < 0.02 and c[2] > 1.05):
            s.labels[fs] = s.R('Shirt_V', '衬衫 V 领内', 'collar', 1.0, Z, seam=Y)
        elif abs(c[0]) < 0.02 and n > 300:  # tie
            fr = m.normal[fs, 1] < 0.2
            a = s.R('Tie_Front', '领带 正面', 'collar', 1.0, Z)
            b = s.R('Tie_Back', '领带 背面', 'collar', 0.5, Z)
            s.labels[fs] = np.where(fr, a, b)
        else:
            s.labels[fs] = s.R(f'Body_Detail_{k}', f'衣物细件 {k}', 'misc', 1.0, Z, rotate=True)
    s.smooth(8)
    s.enforce_connectivity()
    return s.finish()


def seg_head(m, E):
    s = Seg(m, E)
    s.satellites()
    ks = s.comps_by_size()
    main = ks[0]
    fs = s.comp_faces(main)
    p = m.cent[fs]
    face_r = s.R('Face', '脸 正面', 'face', 1.6, Z)
    sk_l = s.R('Skull_L', '后脑 左', 'skull', 0.45, Z)
    sk_r = s.R('Skull_R', '后脑 右', 'skull', 0.45, Z)
    nk = s.R('Neck', '脖子', 'neck', 0.8, Z, seam=Y)
    el = s.R('Ear_L', '左耳', 'ears', 0.8, Z)
    er = s.R('Ear_R', '右耳', 'ears', 0.8, Z)
    # cranium ellipsoid (axis aligned) fitted on faces away from ears / neck
    earbox = (np.abs(p[:, 0]) > 0.12) & (p[:, 2] > 1.19) & (p[:, 2] < 1.37) & (p[:, 1] > -0.07) & (p[:, 1] < 0.09)
    fitm = ~earbox & (p[:, 2] > 1.22)
    q = p[fitm]
    A_ = np.c_[q[:, 0] ** 2, q[:, 1] ** 2, q[:, 2] ** 2, q[:, 1], q[:, 2]]
    coef, *_ = np.linalg.lstsq(A_, np.ones(len(q)), rcond=None)
    rho = np.c_[p[:, 0] ** 2, p[:, 1] ** 2, p[:, 2] ** 2, p[:, 1], p[:, 2]] @ coef
    r_neck = np.hypot(p[:, 0], p[:, 1] - 0.005)
    th = np.degrees(np.arctan2(p[:, 0], -(p[:, 1] - 0.0)))
    s_neck = np.clip((1.185 - p[:, 2]) / 0.01, -1, 1) * 0.5 + np.clip((0.062 - r_neck) / 0.01, -1, 1) * 0.5
    s_ear = np.where(earbox, np.clip((rho - 1.03) / 0.02, -1, 1), -1.0)
    s_face = np.clip((72 - np.abs(th)) / 10, -1, 1) * 0.6 + np.clip((1.47 - p[:, 2]) / 0.02, -1, 1) * 0.4
    s_skull = -s_face
    sx_ = np.clip(p[:, 0] / 0.01, -1, 1)
    scores = np.stack([s_face, s_skull + sx_, s_skull - sx_, s_neck * 2,
                       s_ear * 2 + sx_ - 1, s_ear * 2 - sx_ - 1], 1)
    s.assign(fs, scores, [face_r, sk_l, sk_r, nk, el, er], iters=200)
    # eye / mouth / brow details: each connected component is one small panel
    names = {}
    for k in ks[1:]:
        f2 = s.comp_faces(k)
        if (s.labels[f2] == -1).all():
            continue
        c = m.cent[f2].mean(0)
        if c[2] < 1.25 and abs(c[0]) < 0.04:
            base, cn, grp = 'Mouth', '嘴', 'mouth'
        elif c[2] > 1.345:
            base, cn, grp = f'Brow_{side(c[0])}', f'{cn_side(c[0])}眉', 'brows'
        else:
            base, cn, grp = f'Eye_{side(c[0])}', f'{cn_side(c[0])}眼', 'eye_' + side(c[0])
        names[base] = names.get(base, 0) + 1
        i = names[base]
        prio = 2.2 if base.startswith('Eye') else 1.6
        s.labels[f2] = s.R(f'{base}_{i:02d}', f'{cn} 部件{i}', grp, prio, Z, rotate=False)
    s.enforce_connectivity()
    return s.finish()


def seg_front_hair(m, E):
    s = Seg(m, E)
    s.satellites()
    hc = np.array([0.0, 0.0, 1.33])
    ks = s.comps_by_size()
    for k in ks:
        fs = s.comp_faces(k)
        if (s.labels[fs] == -1).all():
            continue
        c = m.cent[fs].mean(0); n = len(fs)
        out = (m.normal[fs] * unit_rows(m.cent[fs] - hc)).sum(1)
        sc_out = np.clip(out * 2, -1, 1) * 0.7 + np.clip((E[fs] - 0.2) / 0.2, -1, 1) * 0.3
        if n > 4000 and n < 6000 and c[0] > 0.1:
            s.labels[fs] = s.R('Hair_Clip', '发夹', 'clip', 1.2, Z, rotate=True)
            continue
        if n > 15000 or (n > 6000 and c[1] > -0.12):
            # arch-shaped shells spanning both sides: split at the parting (x = 0) and outer / inner
            base, cn, grp = ('Hair_Shell', '顶发外层', 'shell') if n > 15000 else ('Hair_Arch', '顶发中层', 'arch')
            sxx = np.clip(m.cent[fs, 0] / 0.01, -1, 1)
            labs = []; cols = []
            if base == 'Hair_Shell':
                # each half of the shell is a U around the side of the head: split at its lowest point (front / back)
                yc = np.median(m.cent[fs, 1])
                syy = np.clip((yc - m.cent[fs, 1]) / 0.01, -1, 1)
                for sd, cs, sgn in (('L', '左', 1), ('R', '右', -1)):
                    for fb, fcn, sg2 in (('Front', '前段', 1), ('Back', '后段', -1)):
                        labs.append(s.R(f'{base}_{sd}_{fb}_Outer', f'{cn} {cs}{fcn} 外侧', grp, 1.0, Z))
                        cols.append(sgn * sxx + sg2 * syy + sc_out)
                        labs.append(s.R(f'{base}_{sd}_{fb}_Inner', f'{cn} {cs}{fcn} 内侧', grp + '_inner', 0.5, Z, rotate=True))
                        cols.append(sgn * sxx + sg2 * syy - sc_out)
            else:
                for sd, cs, sgn in (('L', '左', 1), ('R', '右', -1)):
                    labs.append(s.R(f'{base}_{sd}_Outer', f'{cn} {cs} 外侧', grp, 1.0, Z))
                    cols.append(sgn * sxx + sc_out)
                    labs.append(s.R(f'{base}_{sd}_Inner', f'{cn} {cs} 内侧', grp + '_inner', 0.5, Z, rotate=True))
                    cols.append(sgn * sxx - sc_out)
            s.assign(fs, np.stack(cols, 1), labs, iters=150)
            continue
        if n > 6000:
            base, cn, grp, pr = 'Bang_Center', '刘海 中', 'bangs', 1.2
        elif c[1] < -0.1:
            base, cn, grp, pr = f'Bang_{side(c[0])}', f'刘海 {cn_side(c[0])}', 'bangs', 1.2
        else:
            base, cn, grp, pr = f'SideLock_{side(c[0])}', f'鬓发 {cn_side(c[0])}', 'sidelocks', 1.0
        o = s.R(base + '_Outer', cn + ' 外侧', grp, pr, Z)
        i = s.R(base + '_Inner', cn + ' 内侧', grp + '_inner', 0.5, Z, rotate=True)
        s.split(fs, sc_out, o, i, iters=120)
    s.enforce_connectivity()
    return s.finish()


def unit_rows(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def seg_back_hair(m, E):
    s = Seg(m, E)
    s.satellites()
    ks = s.comps_by_size()
    locks = ['outer_r', 'inner_r', 'center', 'inner_l', 'outer_l']
    lock_cn = {'outer_r': '右外', 'inner_r': '右内', 'center': '中', 'inner_l': '左内', 'outer_l': '左外'}
    for k in ks:
        fs = s.comp_faces(k)
        if (s.labels[fs] == -1).all():
            continue
        c = m.cent[fs].mean(0); n = len(fs)
        if n > 20000:  # main back hair mass
            p = m.cent[fs]
            W = m.W[m.F[fs]].mean(1)
            score = np.stack([W[:, [m.groups.index(f'hair_back_{l}_0{j}') for j in (1, 2, 3)]].sum(1) for l in locks], 1)
            Ssm = np.stack([s.field(fs, score[:, k], 150) for k in range(5)], 1)
            lock = Ssm.argmax(1)
            # outward = away from body axis (vertical line through head / body), plus exposure
            out = radial_out(m, fs, np.array([0, 0.03, 0]))
            sc = np.clip(out * 2, -1, 1) * 0.6 + np.clip((E[fs] - 0.15) / 0.15, -1, 1) * 0.4
            outer = s.field(fs, sc, 150) > 0
            labs = np.zeros(len(fs), np.int64)
            for li, l in enumerate(locks):
                o = s.R(f'BackHair_{l}_Outer', f'后发 {lock_cn[l]} 外侧', 'back_outer', 1.0, Z)
                ii = s.R(f'BackHair_{l}_Inner', f'后发 {lock_cn[l]} 内侧', 'back_inner', 0.4, Z)
                labs[lock == li] = np.where(outer[lock == li], o, ii)
            s.labels[fs] = labs
        elif n > 5000 and c[2] > 1.4 and abs(c[0]) < 0.05:  # hood / crown shell
            out = (m.normal[fs] * unit_rows(m.cent[fs] - np.array([0, 0.0, 1.33]))).sum(1)
            o = s.R('Hood_Outer', '头顶后发壳 外侧', 'hood', 1.0, Z)
            ii = s.R('Hood_Inner', '头顶后发壳 内侧', 'hood_inner', 0.4, Z)
            s.split(fs, np.clip(out * 2, -1, 1), o, ii, iters=80)
        elif n > 4000 and c[2] > 1.6:  # cat ear shell
            sd = side(c[0]); cs = cn_side(c[0])
            a = s.R(f'CatEar_{sd}_Front', f'{cs}猫耳 正面', 'ears', 1.2, Z)
            b = s.R(f'CatEar_{sd}_Back', f'{cs}猫耳 背面', 'ears', 1.0, Z)
            s.split(fs, np.clip(-m.normal[fs, 1] * 3, -1, 1), a, b, iters=60)
        elif n > 4000:  # ear fur tuft
            sd = side(c[0]); cs = cn_side(c[0])
            s.labels[fs] = s.R(f'EarFur_{sd}', f'{cs}耳内绒毛', 'ears', 1.2, Z)
        else:
            s.labels[fs] = s.R(f'Hair_Detail_{k}', f'发片细件 {k}', 'misc', 1.0, Z, rotate=True)
    s.enforce_connectivity()
    return s.finish()


# ============================================================================ generic (any character)

# body part of a deform bone, by name tokens (camelCase / snake_case / mixamo prefixes all split into tokens);
# checked in order, a token matches when it starts or ends with a pattern (upperarm, LeftForeArm, ball_l, hair_front_01)
PART_RULES = [
    ('hair', '头发', ('hair', 'bang', 'fringe', 'ponytail', 'twintail', 'braid', 'ahoge')),
    ('ears', '耳朵', ('ear',)),
    ('tail', '尾巴', ('tail',)),
    ('head', '头部', ('head', 'neck', 'face', 'jaw', 'eye', 'brow', 'mouth', 'tongue')),
    ('skirt', '裙摆', ('skirt', 'dress', 'coat')),
    ('accessory', '配饰', ('scarf', 'ribbon', 'tie', 'bow', 'cape')),
    ('hands', '手', ('hand', 'finger', 'thumb', 'index', 'middle', 'ring', 'pinky', 'palm', 'wrist')),
    ('arms', '手臂', ('arm', 'clavicle', 'shoulder', 'elbow')),
    ('feet', '脚', ('foot', 'ball', 'toe', 'ankle', 'heel')),
    ('legs', '腿', ('thigh', 'calf', 'leg', 'knee', 'shin')),
    ('torso', '躯干', ('spine', 'chest', 'pelvis', 'hip', 'root', 'waist', 'breast', 'belly', 'body')),
]
PART_CN = {k: cn for k, cn, _ in PART_RULES}
PART_CN.update(misc='其他', part='部件')
PAIRED = {'ears', 'hands', 'arms', 'feet', 'legs'}
PART_PRIO = {'head': 1.4, 'hands': 1.1}            # texel density multipliers (face reads best with more pixels)


def bone_tokens(name):
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    return [t for t in re.split(r'[^a-z]+', s.lower()) if t]


def bone_part(name):
    toks = bone_tokens(name)
    for key, _, pats in PART_RULES:
        if any(t.startswith(p) or t.endswith(p) for t in toks for p in pats):
            return key
    return 'misc'


def bone_side(name):
    toks = bone_tokens(name)
    if 'l' in toks or 'left' in toks:
        return 'L'
    if 'r' in toks or 'right' in toks:
        return 'R'
    return None


def seg_generic(m, E, min_piece=0.02):
    """Any character mesh. Every connected piece is split by the body part of each face's dominant deform bone
    (PART_RULES), paired parts (arms, legs ...) into left / right; the split is diffused over the surface so panel
    borders are smooth, then the usual clean-up (connectivity, opening, de-notching, straightened borders) runs.
    Without skin weights every connected piece is one panel. Seams come from the exposure map (hidden side)."""
    s = Seg(m, E)
    s.satellites()
    has_w = m.W.shape[1] == len(m.groups) and len(m.groups) > 0 and float(m.W.sum()) > 0
    bparts = np.array([bone_part(g) for g in m.groups]) if has_w else np.array([])
    bsides = [bone_side(g) for g in m.groups] if has_w else []
    tot = m.area.sum()
    seen = {}
    for k in s.comps_by_size():
        fs = s.comp_faces(k)
        fs = fs[s.labels[fs] != -1]
        if not len(fs):
            continue
        small = m.area[fs].sum() < min_piece * tot
        if has_w:
            w = m.W[m.F[fs]].mean(1)
            dom = w.argmax(1)
            part = np.where(w.max(1) > 0, bparts[dom], 'misc')
            bs = np.array([bsides[d] or '' for d in dom])
            xs = np.where(m.cent[fs, 0] > 0, 'L', 'R')     # character's left = +X (model faces -Y)
            sd = np.where(np.isin(part, list(PAIRED)), np.where(bs != '', bs, xs), '')
            key = np.char.add(np.char.add(part.astype(str), '|'), sd.astype(str))
        else:
            key = np.full(len(fs), 'part|')
        vals, cnt = np.unique(key, return_counts=True)
        if small:                                       # small loose pieces stay whole (e.g. a hair clip)
            vals = vals[[cnt.argmax()]]; key = np.full(len(fs), vals[0])
        ids = []
        for v in vals:
            pt, sd = v.split('|')
            seen[v] = seen.get(v, 0) + 1
            n = seen[v]
            name = pt.capitalize() + (f'_{sd}' if sd else '') + (f'_{n}' if n > 1 else '')
            cn = ('左' if sd == 'L' else '右' if sd == 'R' else '') + PART_CN.get(pt, pt) + (f' {n}' if n > 1 else '')
            ids.append(s.R(name, cn, pt, PART_PRIO.get(pt, 1.0), Z, rotate=bool(small)))
        if len(vals) == 1:
            s.labels[fs] = ids[0]
        else:
            scores = np.stack([(key == v).astype(float) * 2 - 1 for v in vals], 1)
            s.assign(fs, scores, ids, iters=40)
    s.smooth(4)
    s.enforce_connectivity()
    return s.finish()


SEGMENTERS = {'generic': seg_generic,
              # hand-written rules of the example character (five modules), templates for character-specific rules
              'legs': seg_legs, 'body': seg_body, 'head': seg_head, 'front_hair': seg_front_hair, 'back_hair': seg_back_hair}


def segmenter_for(cfg, obj):
    """config `segmenters: {<object>: <SEGMENTERS key>}`; objects not listed use the generic rules."""
    key = (cfg.get('segmenters') or {}).get(obj, 'generic')
    if key not in SEGMENTERS:
        raise KeyError(f'unknown segmenter {key!r} for {obj}; known: {sorted(SEGMENTERS)}')
    return SEGMENTERS[key]
