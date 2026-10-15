# ${title} · 按部位 UV 交付包

本包是按部位重新排布 UV 后的最终文件。模型的顶点、面、骨骼、权重、法线都没有改动；
新 UV 叫 `UV_clean`（导出时是第 0 通道），原始 UV 保留为 `UV_orig`（第 1 通道）。

## 目录

| 文件夹 | 内容 |
|---|---|
| `01_Model/` | `${blend}`（Blender 源文件，含绑定和两套 UV）<br>`${fbx_ue}`（Unreal 用，厘米，贴图读 `02_Textures`）<br>`${fbx_embedded}`（贴图打包在 FBX 里，单文件可发）<br>`${glb}`（贴图内嵌）<br>`glTF/`（外部贴图版 glTF） |
| `02_Textures/` | ${n} 张 ${size}×${size} 颜色贴图（sRGB），见下表 |
| `03_UV_Guides/` | `*_UV_Overlay.png`：与贴图同尺寸的透明轮廓 + 部位编号，叠在贴图最上层用<br>`*_UV_RegionID.png`：部位色块，每种颜色一个部位，可做魔棒选区<br>`*_UV_Guide.png`：贴图 + 轮廓 + 中文部位名的预览（半尺寸）<br>`*_regions.json`：部位清单（编号、中文名、所在 UV 范围、清晰度系数） |
| `04_Preview/` | 图集总览、棋盘格与贴图前后对比、贴图清理前后特写、部位色块总表 |
| `05_Validation/` | 指标与检查记录（UV 检查、贴图迁移误差、导出回读、工具清单） |
| `验收报告.md` | 数字和未达标项 |

| 贴图 | 部件 |
|---|---|
${texture_rows}

## 改贴图

1. 在绘画软件打开 `02_Textures/*_basecolor.png`。
2. 把 `03_UV_Guides/` 里同名的 `*_UV_Overlay.png` 放到最上层（像素完全对齐），白描边是岛屿边界，数字是部位编号，对应 `*_regions.json`。
3. 只改某个部位：用 `*_UV_RegionID.png` 的颜色做魔棒选区，再扩展 16–24 像素。岛屿外已做平滑外扩，画出一点边也没关系。
4. 保存后：Blender 里在图像编辑器点“重新载入”即可看到（blend 直接引用 `02_Textures` 里的文件）；Unreal 里重新导入贴图即可。

## Blender

* 打开 `01_Model/${blend}`。请保持 `01_Model` 和 `02_Textures` 的相对位置不变，否则贴图会丢失。
* 面属性 `UV_panel` 记录每个面属于哪个部位；对象属性 `UV_panel_names_cn` 是中文部位名。
* 文本编辑器里的 `UV部位选择工具.py` 运行一次后，UV 编辑器右侧 N 面板出现“角色 UV”页：选中角色网格，点“选择部位…”，可按中文名选中某个部位（可同时选左右对称件）。

## Unreal

导入 `01_Model/${fbx_ue}`（单位厘米），贴图从 `02_Textures` 一并导入或手动指定。UV 已经变了，替换旧资产时网格和贴图都要重新导入。
