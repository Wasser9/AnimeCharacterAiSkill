"""Write <output>/report.md from metrics.json + layout comparison (all numbers measured by this
pipeline). Object names are shown with config.display_names when given."""
import json
from pathlib import Path

LAYOUT_CN = {'tripo': '原始 UV', 'previous': '上一版', 'origami': '本版'}

TOUCHUP_CN = {
    'remove_stripe': '竖向灰色条纹（烘焙进贴图的阴影）：识别区域里低饱和、竖向细长的灰色条带及其渐隐的边，用左右两侧的同材质纹理'
                     '克隆、按位置混合填补；靠近边线（例如袜口）时沿边线曲线做亚像素偏移，断开的边线用两侧高度和斜率衔接的三次曲线补齐',
    'seam_hem': '筒状部件（如腿）背后接缝处的边线：两侧边线在接缝前各自上勾/下勾、有亮点、接缝一列偏亮。按模型高度测出两侧边线，'
                '用两边高度、斜率都衔接的三次曲线跨过接缝，接缝两侧 18 像素内按边线对齐克隆重画，接缝旁 5 像素用稍靠里的纹理替换',
    'white_to_skin': '误画成白色的皮肤区域：按线性空间里“相邻主色 → 白色”的投影比例逐像素换成肤色，保留原有明暗与边缘过渡',
    'despeckle': '孤立小亮点/暗点（不超过 16 像素、与周围中值差别很大的点；线条和织纹不动）',
}


def pct(x):
    return f'{x * 100:.1f}%'


def write(out_dir, cfg=None, target_util=0.65):
    cfg = cfg or {}
    out = Path(out_dir)
    names = cfg.get('display_names') or {}
    CN = lambda o: names.get(o, o)
    m = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    cmp_ = json.loads((out / 'validation' / 'layout_comparison.json').read_text(encoding='utf-8'))
    gate = {g['object']: g for g in m['uv_gate']}
    tr = {t['object']: t for t in m['transfer']}
    pk = {p['object']: p for p in m['pack']}
    keys = [k for k in ('tripo', 'previous', 'origami') if all(k in r for r in cmp_)]
    L = []
    L.append('# 按部位（Origami）UV 重排报告\n')
    L.append('每个部位一整块面板、缝线放在背面/内侧/被遮挡处、主要面板方向统一、左右对称件并排、同一部位组在图上连成一片、'
             '小部件集中放在所属部位旁边。模型的顶点、三角面、骨骼、权重、自定义法线和原始 UV（UV_orig）均未改动，新 UV 写在 UV_clean。\n')
    L.append('## 布局对比（同一口径测量）\n')
    L.append('岛数 = UV 上连通的块数；“碎岛”= 面积小于该部件 0.1% 的岛；覆盖率 = 4K 栅格化后被岛屿占用的比例（重叠只算一次）。\n')
    L.append('| 图集 | ' + ' | '.join(f'{LAYOUT_CN[k]} 岛数 / 碎岛 / 覆盖率' for k in keys) + ' |')
    L.append('|---|' + '---:|' * len(keys))
    tot = {k: 0 for k in keys}
    for r in cmp_:
        cells = []
        for k in keys:
            v = r[k]; tot[k] += v['islands']
            cells.append(f"{v['islands']} / {v['tiny_islands']} / {pct(v['coverage'])}")
        L.append(f"| {CN(r['object'])} | " + ' | '.join(cells) + ' |')
    L.append('| **合计岛数** | ' + ' | '.join(f'**{tot[k]}**' for k in keys) + ' |\n')
    if all('outline_turn_deg_per_100px' in r[k] for r in cmp_ for k in keys):
        L.append('**岛屿轮廓锯齿度**（4K 贴图上每 100 像素轮廓的累计转角，越小越平滑；阶梯状边缘每隔几个像素就转一次 90°）：\n')
        L.append('| 图集 | ' + ' | '.join(LAYOUT_CN[k] for k in keys) + ' |')
        L.append('|---|' + '---:|' * len(keys))
        for r in cmp_:
            L.append(f"| {CN(r['object'])} | " + ' | '.join(f"{r[k]['outline_turn_deg_per_100px']:.0f}°" for k in keys) + ' |')
        L.append('')
    L.append('AI 生成模型的原始 UV 覆盖率往往很高，是因为它由大量极小碎片紧挤而成、间距很小，几乎无法人工编辑；'
             '本版用约 16 像素以上的岛间距和大块连续面板换取可读性。\n')
    L.append('## UV 验收\n')
    L.append('| 图集 | 部位数 | 岛数 | 利用率 | 越界 | 翻转 | 岛间重叠(像素) | 岛内自重叠 | 最小岛间距 | 共形畸变 P95（面积加权） | 面积比在 0.8–1.25 的比例 |')
    L.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for o, g in gate.items():
        so = g['self_overlapping_islands']
        L.append(f"| {CN(o)} | {g['regions']} | {g['islands']} | {pct(g['utilisation'])} | {g['out_of_bounds_faces']} | "
                 f"{g['flipped_faces']} | {g['overlap_texels_between_islands']} | {len(so)} | {g['min_gap_px']:.0f} px | "
                 f"{g['distortion']['conf95_area_weighted']:.2f} | {pct(g['distortion']['area_ok_area_weighted'])} |")
    L.append('')
    low = [o for o, g in gate.items() if g['utilisation'] < target_util]
    if low:
        L.append(f"**未达标：** {', '.join(CN(o) for o in low)} 的利用率低于 {int(target_util * 100)}% 的目标。"
                 '常见原因：保持大块连续面板和统一朝向（不为了填缝而旋转、拆碎主面板），并保留 ≥16 像素间距。'
                 '若更看重利用率，可把 `pack.pad_cells` 从 4 降到 3，或给更多部位打开 `rotate`（允许 90° 旋转），代价是可读性下降。\n')
    worst = {}
    for o, g in gate.items():
        for n, v in g['distortion']['worst']:
            if v > 2.0:
                worst[f'{CN(o)}·{n}'] = max(v, worst.get(f'{CN(o)}·{n}', 0))
    if worst:
        L.append('**畸变偏大的部件（共形畸变 P95 > 2）：** ' + '、'.join(f'{k}（{v:.1f}）' for k, v in worst.items()) +
                 '。极小或被遮挡的部件影响不大；可见部件的畸变通常集中在岛屿最外一两排三角形（轮廓平滑的代价），'
                 '画贴图时边缘处纹理会略有拉伸，中心区域正常。接近半球的整块面板（如脸）边缘会有面积伸缩，中心最准。\n')
    L.append('## 打包\n')
    for o, p in pk.items():
        vs = ', '.join(f'v{k}: {pct(v)}' for k, v in sorted(p['variants'].items(), key=lambda x: int(x[0])))
        L.append(f"- {CN(o)}：采用 v{p['chosen_variant']}（{pct(p['utilisation'])}）；候选 {vs}")
    L.append('')
    L.append('## 贴图迁移保真\n')
    L.append('（本节衡量的是任何清理之前的迁移步骤。）原贴图经 UV_orig 直接重采样到新 UV（2 倍超采样、线性空间插值），没有重绘。'
             '在每个三角面内取 4 个点比较新旧颜色（0–1 范围）：\n')
    L.append('| 图集 | 平均绝对差 | P95（面积加权，最大通道） | 差值 > 0.1 的面积占比 |')
    L.append('|---|---:|---:|---:|')
    for o, t in tr.items():
        L.append(f"| {CN(o)} | {t['mean_abs']:.4f} | {t['p95_max_channel']:.4f} | {pct(t['area_over_0p1'])} |")
    L.append('')
    tfx = m.get('texfix')
    if tfx:
        L.append('## 贴图清理（图像模型）\n')
        L.append(f"迁移后的贴图按部位裁成 {sum(1 for t in tfx if not t['name'].endswith('__seam_level'))} 块（只包含该块的岛屿，其余涂成灰色），"
                 '逐块交给图像模型去掉脏色、烘焙阴影和涂抹，再自动贴回原位。贴回时：按轮廓（非灰色区域）与岛屿蒙版做仿射 + 平滑形变对齐，'
                 '取重合度最高的方案；只替换岛屿内部，岛边一圈用内部颜色向外延伸；模型分辨率以下的细纹理（针织纹等）按高频叠回；'
                 '每个岛按中位色把整体色调校正回原贴图；模型删掉的岛保留原样。各块分开清理会让相邻岛出现色差，'
                 '最后做“接缝找平”：在 3D 接缝两侧取样，原贴图本来连续的接缝两边加平滑的颜色修正；露在外面的大块基本保持颜色，'
                 '被遮挡的小块去迁就它们；原图在接缝处本来就不同色的不受影响。最后重新做岛外的平滑外扩。\n')
        L.append('| 裁块 | 对齐方式 | 轮廓重合度（贴回前 → 对齐后） | 保留原样的岛 |')
        L.append('|---|---|---:|---|')
        lv = [t for t in tfx if t['name'].endswith('__seam_level')]
        for t in tfx:
            if t in lv:
                continue
            kept = '、'.join(f'{k} 像素' for k in t.get('islands_kept_original', [])) or '—'
            L.append(f"| {t['name']} | {t.get('method', '')} | {t.get('iou_raw', 0):.3f} → {t.get('iou', 0):.3f} | {kept} |")
        L.append('')
        if lv:
            L.append('接缝找平效果（接缝两侧颜色差，线性空间 0–1，只统计原图连续的接缝）：\n')
            L.append('| 图集 | 参与找平的接缝点对 | 找平前 | 找平后 |')
            L.append('|---|---:|---:|---:|')
            for t in lv:
                L.append(f"| {CN(t['name'].replace('__seam_level', ''))} | {t['constrained']} | {t['seam_diff_before']:.4f} | {t['seam_diff_after']:.4f} |")
            L.append('')
    tu = m.get('touchup') or []
    if tu:
        L.append('## 定点修正\n')
        L.append('config.yaml 的 `touchup`（代码 `pipeline/touchup.py`），每次运行自动重做：\n')
        for t in tu:
            what = TOUCHUP_CN.get(t['op'], t['op'])
            extra = []
            if t.get('texels') is not None:
                extra.append(f"{t['texels']} 像素")
            if t.get('specks') is not None:
                extra.append(f"{t['specks']} 处")
            if t.get('skin_rgb'):
                extra.append(f"肤色 RGB{tuple(t['skin_rgb'])}")
            L.append(f"- {CN(t['object'])} · {'、'.join(t['regions'])}：{what}" + (f"（{'，'.join(extra)}）" if extra else '') + '。')
        L.append('- 修正区域外侧的岛外填充（防止贴图缩小时边缘渗色）按修正后的颜色重新外扩。\n')
    if tfx or tu:
        L.append('清理前的迁移贴图保存在 `output/work/transfer/*_migrated.png`，可随时换回。\n')
    ex = m.get('export', {})
    rb = ex.get('readback', {})
    if rb:
        f = rb['fbx']; g = rb['glb']
        L.append('## 导出与回读\n')
        L.append(f"- 保持性：各网格的顶点、面、原始 UV、权重、自定义法线签名在写入前后完全一致："
                 f"{all(ex['preservation'].values())}；重新打开 blend 后 UV_clean 与计算结果逐值一致："
                 f"{all(v['UV_clean_equal'] for v in ex['blend_reload'].values())}。")
        L.append(f"- FBX（Unreal，厘米）：{f['triangles']} 三角面（应为 {f['expected_triangles']}），UV 通道 {f['uv_layers'][0]}，"
                 f"第 0 通道与 UV_clean 一致：{f['uv0_matches_UV_clean']}，骨骼 {f['bones']}，材质 {f['materials']}，"
                 f"每顶点最多 {f['max_influences']} 个权重，未绑定顶点 {f['unbound_vertices']}，模型高 {f['height']:.4f} m。")
        L.append(f"- GLB：{g['triangles']} 三角面，{g['meshes']} 个网格，第 0 通道与 UV_clean 一致：{g['uv0_matches_UV_clean']}"
                 f"（按 1/4096 容差逐值匹配 {g.get('uv0_value_match', 0) * 100:.2f}%，glTF 存储时翻转 V 轴带来浮点舍入），"
                 f"骨骼 {g['bones']}，未绑定顶点 {g['unbound_vertices']}。\n")
    L.append('## 仍需注意\n')
    L.append('- AI 生成的三角网格没有建模边环，部位边界在 3D 上只能沿三角形走。UV 里的岛屿轮廓已单独平滑'
             '（按弧长高斯平滑、保留真正的拐角，再用 SLIM 软约束把边界拉过去，不翻转、不重叠），代价是靠近边缘的一两排三角形有少量拉伸。'
             '岛外填充是金字塔式平滑外扩（push-pull）。3D 上的分界线位置不变。')
    L.append('- 原网格中的非流形边、退化面、少量悬浮碎面保留不动；碎面（每处 ≤40 面）不单独成岛，而是贴附到相邻主面板上取色。')
    L.append('- 展开后会自我重叠的部位会被自动二分，同名部位可能由 2–4 块组成，编号相同。')
    L.append('- Unreal / 其他引擎中需重新导入网格与贴图（UV 已经变了）；材质槽名沿用输入文件。')
    (out / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    return '\n'.join(L)
