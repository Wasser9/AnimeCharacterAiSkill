# 阶段二A：按部位（Origami）UV 重排

把 AI 生成的模型（如 Tripo、Meshy、Hunyuan3D）上成千上万的碎 UV 岛，重排成按部位划分的大块面板，排布方式参照米哈游角色贴图。整条流水线是确定性的，重跑结果一致。

排布原则：
- 每个部位一整块面板，缝线藏在背面、内侧或被遮挡处。
- 主要面板头朝上、腿竖直、发根在上。
- 左右对称件并排放；同组部位在图上连成一片，小件放在所属部位旁边。

代码在 `scripts/origami_uv/`，入口是 `run.py`。所有参数都在项目里的 `config.yaml`（模板 `config.example.yaml`，完整示例 `examples/sourceparts.config.yaml`）。

## 输入要求

- **一个 .blend 文件**，只读，流水线从不写回。
  - 每个模块是一个网格对象，对象名即图集名，例如 `CHR_Head`、`CHR_Body`。
  - 网格必须**全部是三角面**。AI 模型一般已经满足；否则先在副本上加 Triangulate 并应用。
  - 角色站立，面朝 Blender −Y，Z 向上，单位米，脚底在 z≈0。
  - 通用分区规则默认 +X 是角色左侧；侧面特写的视角也按这个方向。
- **原始 UV 层**：原贴图画在这层上，在 `source_uv` 里填层名。它会在输出里改名为 `UV_orig` 保留下来。
- **原始颜色贴图**，按以下顺序查找：
  1. `<对象名><reference_suffix>` 副本对象材质里的图（未改动的原材质，例如 `CHR_Head_NormalReference`）；
  2. 对象自身材质里接到 Base Color 的那张图，打包在 blend 里或外部文件都可以。
  - 法线、粗糙度等其他贴图不迁移；Blender 里它们继续通过 `UV_orig` 采样，引擎里需要另行处理。
- **骨架（可选）**：网格用 Armature 修改器绑定。
  - 有骨架时，导出带骨骼的 FBX，通用分区规则按骨骼名划分部位。
  - 没有骨架时，导出静态网格，每个连通块作为一个部位。
- **自定义法线、权重、顶点和面**：全部保持不变，导出后会逐项回读核对。

## 环境

- Blender 4.x，在 `config.blender` 里填可执行文件路径。
- Python 3.11 或以上，安装 `scripts/origami_uv/requirements.txt` 里的包：numpy、scipy、libigl、shapely、opencv、pillow、pyyaml。
  - 也可以把这些包装到一个文件夹，在 `python_packages` 里指向它。
- 内存与磁盘：
  - 展开和打包时每个进程峰值约 1 GB。内存或页面文件不够时，调小 `workers` 和 `pack_workers`。
  - 页面文件在 C 盘时，C 盘满了也会报 MemoryError 或“页面文件太小”。
  - 一个 4K × 5 图集的项目，输出约 1 GB。

```bash
python scripts/origami_uv/run.py --config config.yaml                 # 全流程 0–8
python scripts/origami_uv/run.py --config config.yaml --to-stage 3    # 先只排布，确认后再往下跑
python scripts/origami_uv/run.py --config config.yaml --from-stage 3  # 改了排布参数
python scripts/origami_uv/run.py --config config.yaml --only-stage 7  # 只重出对比图和报告
python scripts/origami_uv/run.py --config config.yaml --objects CHR_Legs --from-stage 2   # 只重做一个部件
```

只重做部分部件时，`metrics.json` 里其他部件的记录会保留，定点修正也只作用于重做的部件。每个任务都在新进程里运行（`max_tasks_per_child=1`），所以结果和进程数、任务顺序无关。原生库会在同一进程里残留状态：曾经出现过同一个部件在另一个部件之后展开时，缝线不一样的情况。

## 阶段

| 阶段 | 做什么 | 主要产物（都在 `output/` 下） |
|---|---|---|
| 0 inspect | 用 Blender 读出网格、权重、全部 UV 层、原贴图 | `work/inspect/*.npz`、`src_tex/*.png`、`scene.json` |
| 1 exposure | 从 64 个方向算每个面的可见度，所有部件互相遮挡（头发挡头皮、鞋挡脚） | `work/exposure.npz` |
| 2 unwrap | 分区 → 缝线（最短路径，优先走不可见处）→ LSCM/Tutte + SLIM 展开；岛内自重叠时自动二分；岛轮廓做平滑 | `work/unwrap/*.pkl` |
| 3 pack | 按部位组的亲和度栅格打包，试 6 组参数，取利用率最高的一组 | `work/pack/*.pkl` |
| 4 transfer | 把原贴图经 `UV_orig` 重采样到新 UV（2 倍超采样、线性光插值）；岛外先平滑外扩再过渡到纯色背景；随后做可选的贴图清理和定点修正（见 texture-processing.md） | `final/textures/*_basecolor.png`、`work/transfer/*_migrated.png` |
| 5 gate | UV 硬检查（越界、翻转、岛间重叠、最小间距、密度）、绘制辅助图、写回 Blender 用的数据 | `guides/*`、`metrics.json` |
| 6 export | 在 blend 副本里写入 `UV_clean`（第 0 通道）、`UV_panel` 面属性和新贴图；导出 FBX、GLB、glTF 并回读核对 | `final/*.blend/.fbx/.glb`、`validation/export_readback.json` |
| 7 review | 8 视角棋盘格与贴图前后对比、清理前后特写、图集总览、部位色块表、布局对比 | `review/*.png`、`UV_Atlas_Overview.png`、`report.md` |
| 8 package | 整理交付文件夹（每次重建；不是本流水线建的同名文件夹会拒绝覆盖） | `package_dir/01_Model … 05_Validation`、README、验收报告 |

## 迭代顺序

1. **先跑 0–3，只看排布。** 打开 `UV_Atlas_Overview.png`、`review/Region_ID_Sheet.png` 和 `guides/*_regions.json`，确认：
   - 部位划分合理，没有把一块衣服切成很多块；
   - 左右件成对；
   - 面板方向正确；
   - 利用率和间距可以接受。
2. **调分区或打包，然后重跑。**
   - 改分区规则用 `--from-stage 2`。
   - 改 `pack.groups`、`merge`、`pad_cells`，或改部位的 `rotate`、`prio`，用 `--from-stage 3`。
3. **排布确认后跑完 4–8。** 贴图问题在第 4 阶段处理，见 [texture-processing.md](texture-processing.md)。
4. **验收**：读 `report.md` 和 `metrics.json`，标准见 [quality-checks.md](quality-checks.md) 的“阶段二”。看对比图时以近景渲染为准，不只看数字。

## 分区规则（`pipeline/segment.py`）

每个对象调用一个分区函数，返回“每面的部位编号 + 部位表”。用哪个函数由 `config.segmenters: {对象名: 键}` 决定，没写的对象用 `generic`。

- **`generic`**：适用于任何角色。
  - 每个连通块里，按每个面权重最大的骨骼判断部位。头发、耳、尾、头、裙、配饰、手、手臂、脚、腿、躯干，按骨骼名关键词匹配；UE 小白人、Mixamo、VRM（J_Bip）、Biped 命名都能识别。
  - 成对部位按骨骼名里的 L/R 拆开，名字里没有时按面的 x 正负拆。
  - 在表面上扩散平滑边界，再做连通性、开运算、去锯齿和直线化清理。
  - 没有骨骼权重时，一个连通块就是一个部位。
- **`legs`、`body`、`head`、`front_hair`、`back_hair`**：示例角色手写的规则（五模块：脸、衣服加手、腿袜鞋、刘海、后发猫耳），阈值按那个模型调过。
  - 新角色先用 `generic` 看效果，不满意的部件再仿照这些函数写专用规则。
  - 在示例角色上的对比：
    - `generic`：衣服图集利用率 68%（26 个部位）、腿鞋 57%（10 个部位），验收全部通过；腿部面板和专用规则一样整齐，但毛衣前后片连成一大块（带领口洞），鞋面和鞋内壁混在一起。
    - 专用规则：68% 和 67%，鞋面、鞋内壁、鞋底分开排放。

写专用规则时用 `Seg` 工具类：

```python
def seg_my_legs(m, E):                       # m: 网格（m.cent 面中心, m.normal, m.area, m.W 权重, m.groups 骨骼名）; E: 每面可见度
    s = Seg(m, E)
    s.satellites()                           # 极小碎面不单独成岛，贴附宿主取色
    for k in s.comps_by_size():              # 按连通块从大到小
        fs = s.comp_faces(k)
        if (s.labels[fs] == -1).all():
            continue
        c = m.cent[fs].mean(0); sd = side(c[0]); cs = cn_side(c[0])
        leg = s.R(f'Leg_{sd}', f'{cs}腿', 'legs', 1.0, Z, seam=Y)      # 名称, 中文名, 打包组, 密度系数, 朝上方向, 缝线偏好方向
        sole = s.R(f'Sole_{sd}', f'{cs}鞋底', 'soles', 0.6, -Y, rotate=True)
        s.labels[fs] = np.where(m.normal[fs, 2] < -0.5, sole, leg)
    s.smooth(6); s.enforce_connectivity()
    return s.finish()                        # 开运算 + 去锯齿 + 边界直线化

SEGMENTERS['my_legs'] = seg_my_legs          # config: segmenters: {CHR_Legs: my_legs}
```

部位字段说明：
- `prio` 是纹素密度系数，1 等于图集平均。经验值：
  - 脸 1.4–1.6，眼部细件 2.2，眉、嘴 1.6，刘海外侧 1.2；
  - 被遮挡的面（后脑、头发内侧、鞋里、裙内、袖口封口）0.25–0.6。
- `up` 是面板在 UV 里的朝上方向。
- `seam` 是缝线偏好的 3D 方向；为 None 时按可见度放在最不显眼处。
- `rotate=True` 允许打包时旋转 90°，只给小件用。
- `group` 相同的部位在图上排在一起；`pack.merge` 可以把太小的组并入相邻组。

## 打包参数（`pack`）

- `mode`：
  - `affinity`（默认）：同组岛相互贴靠，组与组之间可以穿插；
  - `blocks`：每组一个严格矩形区域，整齐但利用率低。
- `grid: 2048`：打包栅格，4K 图上每格 2 像素。
- `pad_cells: 4`：4K 图上岛间距至少 16 像素。降到 3 可以多挤出约 3–5% 利用率，可读性会差一些。
- `target_utilisation: 0.70`：密度搜索的目标值，报告按 65% 判断是否达标。
- `groups`：每个图集里组的摆放顺序，越靠前越先放、越靠左上。不写时按分区函数创建部位的顺序。

## 结果怎么用

- 在 Blender 里：
  - `UV_clean` 是渲染和导出用的第 0 通道，`UV_orig` 保留原 UV；
  - 面属性 `UV_panel` 记录部位编号，对象属性 `UV_panel_names_cn` 是部位中文名；
  - 文本编辑器里运行 `UV部位选择工具.py` 后，UV 编辑器 N 面板可以按中文名选中部位。
- 画贴图：
  - 把 `guides/*_UV_Overlay.png` 放在贴图最上层（像素完全对齐）；
  - `*_UV_RegionID.png` 的每种颜色对应一个部位，可以做选区；
  - 岛外已有平滑外扩，画出边界一点也不要紧。

## 常见问题

- **`Blender step failed`**：看 `output/logs/stageN_*.log`。
- **`no source texture found`**：检查 `source_texture.reference_suffix`，以及对象材质里的图是否接在 Base Color 上。
- **`no UV layer ... (config source_uv)`**：`source_uv` 填错了。可用的层名在 `work/inspect/scene.json` 里能查到。
- **报 `UV GATE FAILED`（越界、翻转、重叠）**：
  1. 先看是哪个部位；
  2. 多数是分区边界太碎或岛太窄。可以调高 `enforce_connectivity`、`open_regions` 的阈值，或把该部位并入邻居，然后 `--from-stage 2`。
- **利用率低**：
  - 常见原因是头发这类细长件多，或者组的顺序不合理；
  - 可以调 `groups` 顺序、给小件开 `rotate`、把 `pad_cells` 降到 3；
  - 不要为了利用率把主面板拆碎。
- **贴图边缘锯齿**：已经做了轮廓平滑和 push-pull 外扩。仍有问题时，对比 `layout_comparison.json` 里的 `outline_turn_deg_per_100px`。
- **内存不足**：调小 `workers` 和 `pack_workers`（例如各 2）。第 5 阶段 `apply_data` 也会同时开多个进程。
