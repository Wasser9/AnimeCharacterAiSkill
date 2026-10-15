# 阶段二B：贴图迁移、清理与定点修正

UV 重排后，第 4 阶段依次做三件事：

1. **迁移**：自动完成。
2. **图像模型清理**：可选。需要人工把裁块交给 ChatGPT 之类的图像模型，结果放回后由流水线自动贴回。
3. **定点修正**：可选，在 config 的 `touchup` 里配置。

三步都能重跑：每次运行都从迁移后的原贴图重新做起，所以改了参数直接 `--from-stage 4` 即可。
清理前的贴图一直保存在 `output/work/transfer/<对象>_migrated.png`。

## 1. 迁移（自动）

- 原贴图经 `UV_orig` 重采样到新 UV：三角形 ID 光栅化，2 倍超采样，在线性光空间做双线性插值。
- 岛外先做 16 像素的 push-pull 平滑外扩，再过渡到纯色背景。这样贴图缩小时边缘不会渗进背景色。
- 保真度记在 `metrics.json → transfer` 里：平均绝对差、P95、差值大于 0.1 的面积占比。迁移本身不重绘，数值应该很小；数值大通常说明 `source_uv` 或原贴图找错了。

## 2. 图像模型清理

AI 模型贴图的常见问题：脏色斑、烘焙进去的阴影、涂抹拉丝、投影色带。思路是：
- 把图集按部位裁成块，逐块交给图像模型清理；
- 流水线按轮廓把结果对齐贴回，再找平各块之间的色差。

**脸和眼睛默认不送清理**：容易改变角色的长相和眼神。

### 步骤

1. 先跑完 0–4，然后切裁块：
   ```bash
   python scripts/origami_uv/run.py --config config.yaml --texfix-prepare
   ```
   产物在 `output/work/texfix/`：
   - `crops/<对象>__NN.png`：要发送的图，尺寸 1024×1024、1536×1024 或 1024×1536，按图像模型支持的比例补成；
   - `crops/<名>_mask.png`：岛屿蒙版；
   - `manifest.json`：每块在图集里的窗口位置和缩放比例。

   一个岛只会完整落在一块里。需要单独处理某个岛时，可以手动在 manifest 里加窗口（坐标用整数），并改对应的蒙版。
2. **决定送哪些块**：看 crops。脸、眼睛、本来就干净的块、细线条图案（格纹、领口线）可以不送；图案块送了的话，在 `texfix.detail_override` 里设为 0（见下文）。
3. **写提示词**：每块一条，填进模板。示例见 `scripts/origami_uv/examples/gpt_prompts.sourceparts.json`。
   ```text
   This image is one piece of a UV texture atlas for a stylized anime 3D character. The flat grey area is empty background.
   Content: {这块里有什么：材质、颜色、可见结构}. Clean up the texture only: {要去掉什么：脏斑、烘焙阴影、某条色带；要保留什么：针织纹、罗纹、缝线}
   Keep EXACTLY the same image size, composition and layout: every shape, outline, position and size must stay identical;
   do not move, rotate, resize, add or remove anything; keep the grey background flat grey. No text, no new objects.
   ```
   `{fix}` 要写具体，例如“去掉胸前那条围巾色的竖向投影”，不要只写“清理一下”。
4. **在浏览器里用 ChatGPT 生成**：用户自己的账号，通过 Claude in Chrome 或手动操作，一次一块。
   - 发送前先确认三样东西都在：图片附件、“创建图像”模式、完整提示词。发送是对外操作，需要用户同意。
   - 结果按原名存为 `output/work/texfix/gpt_results/<对象>__NN.png`；不满意的移到 `gpt_rejected/` 重做。
5. **自动贴回**：
   ```bash
   python scripts/origami_uv/run.py --config config.yaml --from-stage 4
   ```
   有结果的块会贴回，没有结果的块保持原样。

### 浏览器操作经验（ChatGPT 网页版）

- 用真实点击打开输入框旁的“+”菜单，选“创建图像”。图片通过隐藏的文件输入“附加照片或视频”上传。
- 上传偶尔失败（提示“上传未成功”）：刷新页面后重新附加。
- 页面里的 JS 调用约 45 秒超时：发送、等待、下载分成几次调用，轮询结果是否出现。
- 下载：把结果图画到 canvas 再导出为指定文件名的 PNG。
- 返回尺寸可能和发送的不同（例如 1254²、1448×1086）。贴回时会按轮廓重新对齐，不需要手动缩放。

### 贴回做了什么（`pipeline/gpt_texfix.py`）

- **对齐**：按轮廓对齐，不按颜色边缘对齐。把非灰色区域当作剪影，与岛屿蒙版比较：
  - 先试拉伸、等比适配、填满三种缩放，再做仿射 ECC 和基于 SDF 的稠密光流微调；
  - 取剪影重合度（IoU）最高的方案。
- **合成**：只替换岛屿内部；岛边一圈用内部颜色向外延伸，不带模型画的轮廓线，也不带原来的脏边。
- **细节回叠**：图像模型分辨率以下的原贴图细纹理（针织纹等）按高频叠回，强度 `detail=0.7`。格纹、线稿类块在 `detail_override: {块名: 0.0}` 里关掉，否则原来的脏线会被叠回来。
- **偏色校正**：每个岛只算一个增益，按中位色把整体色调拉回原贴图（限制在 0.75–1.33 之间）。分开清理的块会各自偏色，这一步把它们统一回来。
- **边缘贴合**：用导向滤波把边缘贴合到岛轮廓。
- **接缝找平**：在 3D 接缝两侧取样，只处理原贴图里本来就连续的接缝，求一个平滑的颜色修正。
  - 露在外面的大块被锚定（可见度权重高），藏起来的小块去迁就它们。
  - 如果全局一起平移颜色，会把袜子这种大块整体拉暗，所以不这样做。
- **重做外扩**：最后重新做岛外的平滑外扩。
- 模型删掉了的岛（剪影里缺失）保留原样，记录在日志的 `islands_kept_original` 里。

## 3. 定点修正（`config.touchup`，`pipeline/touchup.py`）

用户能指出具体问题时用，比如“这条纹其实是阴影”或“这块白色应该是皮肤”。每条配置写清对象、部位名和操作；部位名取自 `guides/<对象>_regions.json`。

每个操作只在这些部位的纹素内修改，改完后把这些部位外侧的填充按新颜色重新外扩。不重新外扩的话，旧颜色会在贴图缩小时从边缘透出来；例如皮肤边上会出现一圈白边。

| op | 用途 | 要点 |
|---|---|---|
| `remove_stripe` | 衣物上烘焙进去的灰色竖条、阴影带、小污块 | 识别低饱和、竖向细长的灰带及两侧渐隐的边。用左右两侧的同材质纹理克隆填补，按位置混合两侧，避免出现切换线。靠近袜口这类边线时，按边线曲线做亚像素上下偏移；边线在条纹处断开或错位时，用两侧高度和斜率衔接的三次曲线补上 |
| `seam_hem` | 筒状部件（腿、袖）在 UV 接缝处，边线两侧对不上，或有勾角、亮点 | 按模型高度（不是像素行）测两侧边线，用三次曲线跨过接缝。只在接缝两侧 18 像素内整段替换（宽范围半透明混合会出“獠牙”状楔形），接缝旁 5 像素用稍靠里的纹理替换。部位必须是完整的筒 |
| `white_to_skin` | 误画成白色的露肤区域（如长袜上方的大腿） | 每个纹素按“相邻主色 → 白色”的投影比例换成肤色，明暗和抗锯齿边保持不变。肤色用 `params.skin_from` 取另一部位的中位色（例如脖子），保证和脸、手一致 |
| `despeckle` | 孤立的小亮点、小暗点 | 只处理不超过 16 像素、与 7×7 中值差别很大的点，线条和织纹不受影响。放在其他修正之后做 |

示例：

```yaml
touchup:
  - {object: CHR_Legs, regions: [Leg_L, Leg_R], op: remove_stripe}
  - {object: CHR_Legs, regions: [Leg_L, Leg_R], op: seam_hem}
  - object: CHR_Legs
    regions: [Leg_L, Leg_R, LegTop_L, LegTop_R]
    op: white_to_skin
    params: {skin_from: {object: CHR_Head, regions: [Neck]}}
  - {object: CHR_Legs, regions: [Leg_L, Leg_R], op: despeckle}
```

需要新类型的修正时，在 `touchup.py` 里加一个函数 `op(tex_bgr, region, **params) -> (new_bgr, info)`，注册到 `OPS`。需要网格信息（接缝、高度）的操作同时加进 `NEEDS_CTX`。

修正时踩过的坑：
- 横向平移克隆遇到斜的边线，会克隆出台阶；必须沿边线曲线偏移。
- 整数行偏移会出现锯齿，要用亚像素双线性采样。
- 一条二次曲线拟合不了在条纹处错位的边线，要用两侧衔接的三次曲线。
- 只取一侧克隆或在两侧之间硬切换，会留下切换线，要按位置混合两侧。

## 4. 验证

- 第 7 阶段的 `review/Closeup_texture_cleanup.png`：7 个视角（上身正面、背面、大腿正面、腿侧面、腿背面、鞋、侧面头发），左列清理前，右列清理后。镜头按模型高度自动缩放，可以在 `review.closeup_views` 里改。
- 看到可疑痕迹时：
  1. 用 `pipeline/bl_closeup.py` 配一个只含 1–3 个近景视角的 job，范围约 0.2 m，渲染局部特写。
  2. 定位痕迹来自哪些纹素：复制一份贴图，把候选区域染色（例如岛边一圈染绿、修补区域染红），重新渲染同一视角。痕迹落在哪种颜色上，就是哪里的问题。
  3. 在贴图上按最近邻放大 2–4 倍，并排对比修改前后。
- 只看数字不够：接缝、背面、袜口这类问题只有近景渲染才能看出来。每次改完都要看背面和接缝。
