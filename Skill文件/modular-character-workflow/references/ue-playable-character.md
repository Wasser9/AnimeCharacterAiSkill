# 阶段三：导入 UE5，做成可直接操控的第三人称角色

把第二阶段 `finaloutput` 里带骨骼的 FBX 和颜色贴图做成一个能直接 Play 的角色：WASD 移动、鼠标转视角、空格跳；动作从 UE 第三人称模板的 Manny 重定向过来。

脚本在 `scripts/ue_character/`，已在 UE 5.7 上验证：

| 文件 | 用途 |
|---|---|
| `ue_build_playable.py` | 构建全部资产 |
| `ue_verify_play.py` | PIE 实机测试 + 截图 |
| `ue_diag_pose.py` | 不渲染，排查姿势问题 |
| `make_preview_sheet.py` | 把截图拼成一张预览图 |
| `ue_common.py` | 读取配置 |

配置写在 JSON 里：脚本旁边的 `ue_config.json`，或者用环境变量 `UE_CHAR_CONFIG` 指向别处。示例见 `ue_config.example.json`，全部选项见 `ue_common.py` 的 `DEFAULTS`。

## 前提

- **模型**：
  - 骨骼命名最好接近 UE 小白人（root、pelvis、spine_01、neck_01、head、clavicle、upperarm、lowerarm、hand、thigh、calf、foot、ball），这样 IK Rig 能自动识别链条。
  - 其他命名也能用，但要检查 build_report 里自动生成的链条是否完整。
  - 没有手指骨骼可以，手指不会动。
- **FBX**：单位厘米，角色在 UE 里面朝 +Y（Blender 里面朝 −Y），对应 `mesh_yaw: -90`。
- **项目里需要第三人称模板内容**：从引擎目录 `Templates/TemplateResources/High/{Characters,Input,LevelPrototyping}/Content` 和 `Templates/TP_ThirdPersonBP/Content/{ThirdPerson,__ExternalActors__,__ExternalObjects__}` 复制到项目 `Content/` 下，路径对应 `/Game/Characters`、`/Game/Input`、`/Game/ThirdPerson`、`/Game/LevelPrototyping`。复制前确认这些文件夹不存在，不覆盖用户已有内容。
- 项目的默认地图、默认 GameMode 等全局设置都不改。生成的关卡在 World Settings 里自带 GameMode 覆盖。

## 构建

先关闭编辑器，删除上一次生成的 `Content/<root>`，以及 `Content/__ExternalActors__/<root>`、`Content/__ExternalObjects__/<root>`。这几个文件夹完全由脚本生成；删除前先确认里面没有手工改过的资产。然后运行：

```bash
UnrealEditor.exe <项目>.uproject -ExecutePythonScript=<skill>/scripts/ue_character/ue_build_playable.py -nullrhi -unattended -nosplash
```

构建步骤（每一步写入 `build_report.json`）：
1. **导入 FBX**：用传统 FBX 导入，导入法线，不导入材质，生成物理资产；导入后立即保存骨骼和物理资产。
2. **材质**：建一个母材质（Lit，粗糙度 0.85，自发光 0.12 以保留暗部原色），每个部件一个材质实例。`two_sided` 里的部件开双面；材质槽名包含哪个部件名就用哪个部件的贴图。
3. **IK Rig**：Manny 和角色各建一个，都用 `apply_auto_generated_retarget_definition` 加 `apply_auto_fbik` 自动生成。
4. **重定向器**：默认 op（Pelvis Motion、FK Chains、IK Goals、Run IK Rig、Root Motion、Remap Curves）+ 模糊自动映射链条。**只对 `align_bones`（默认两侧手臂）做 Chain-to-Chain 对齐**，用来把 T-pose 的手臂转成 A-pose。
5. **批量重定向**：`duplicate_and_retarget(ABP_Unarmed)`，连同 BlendSpace 和 20 段待机/走/跑/跳动作一起复制，再统一移到 `<root>/Animations`。
6. **脚部 IK 直通**：把复制出来的 ABP 里 `CR_Mannequin_FootIK` 节点换成一个空的直通 Control Rig。
7. **角色蓝图**：复制模板角色蓝图，换网格和动画蓝图，设置胶囊（半高约为身高的一半）和网格偏移（z = −半高，yaw = `mesh_yaw`）。
8. **GameMode 和关卡**：复制模板 GameMode 并设为我们的 Pawn；用模板关卡生成新关卡，设置 GameMode 覆盖，确认 PlayerStart 存在。

## 测试

```bash
UnrealEditor.exe <项目>.uproject -ExecCmds="py <skill>/scripts/ue_character/ue_verify_play.py" -unattended -nosplash
python <skill>/scripts/ue_character/make_preview_sheet.py
```

测试流程：打开关卡 → 放一个不保存的 SceneCapture2D → 开 PIE → 待机采样并截正、侧、背、脸 → 向前跑 4 秒采样并截图 → 停下 → 起跳采样并截图 → 落地 → 结束 PIE → 删除截图用的 actor → 退出编辑器。

`verify_report.json` 里的 `checks` 会自动判断姿势：
- 待机时手低于上臂；
- 脚不是竖直的（脚尖向前偏下）；
- 头高合理；
- 跑速大于 100；
- 两次采样之间腿在动；
- 起跳时处于下落状态。

任何一项不过，状态就是 `CHECKS_FAILED`。最后一定要看拼好的 `UE_Playtest_Sheet.png`。

姿势不对时，运行 `ue_diag_pose.py`（`-nullrhi` 即可）。它比较源动画和重定向后动画的骨骼位置：
- 重定向后的序列本身就错，是重定向器的问题，例如对齐了不该对齐的骨骼；
- 序列是对的、PIE 里才错，是动画蓝图里的节点问题，例如 Control Rig。

## UE 5.7 Python 的坑（都踩过）

- `-run=pythonscript` 命令行模式下，`duplicate_and_retarget` 结束时要同步内容浏览器（需要 Slate），会直接崩溃。构建必须用完整编辑器 `UnrealEditor.exe -ExecutePythonScript=… -nullrhi`。
- `-ExecutePythonScript` 在脚本返回后的下一帧就退出编辑器。基于 tick 回调的测试要用 `-ExecCmds="py <file>"`。
- 编辑器启动后头几帧的 delta time 可能很大。等待计时按帧数算，不按秒算。
- `load_level` 会驱动 Slate tick，从而重入 tick 回调。调用前先切换状态，防止重复加载。
- `duplicate_and_retarget` 把副本放在 `/Game` 根目录，返回列表里也不包含复制出的 ABP，要按后缀到根目录里找。
- `auto_align_all_bones` 会把这类模型的脚转成脚尖朝下约 56°。只对齐手臂。
- 模板 `ABP_Unarmed` 里的 `CR_Mannequin_FootIK` 按 Manny 的骨骼轴向写，会把别的骨架的脚掰成竖直。
  - 这个节点的 Alpha 是引脚，改结构体里的值不生效。
  - 把 Control Rig 类清空，节点会输出参考姿势（T-pose）。
  - 正确做法：指向一个空的 Control Rig（`ControlRigBlueprintFactory.create_new_control_rig_asset`），等于直通。代价是斜坡上脚不贴地。
- 在 `-nullrhi` 下生成 SceneCapture2D 会报整数除零崩溃。截图相关的事只在带 RHI 的测试里做。
- PIE 里用 `HighResShot` 截图是纯白；SceneCapture 用 `FINAL_COLOR_LDR` 也是纯白：单次拍摄没有自动曝光历史，物理光照下直接过曝。改为拍两次：
  - `SCS_BASE_COLOR`：看贴图和姿势；
  - `SCS_SCENE_COLOR_HDR`：导出 .hdr，离线按平均亮度曝光并色调映射。
- Python 里没有 `add_component_by_class`，也没有 `begin_deferred_actor_spawn_from_class`，不能直接往 PIE 世界里生成 actor。做法是开 PIE 前在编辑器世界放一个不保存的 actor，PIE 会复制它。
- 半成品资产的骨骼没保存时，`delete_directory` 会报 RuntimeError，但删除其实已经完成；脚本里捕获后检查目录是否已空。
- 内存：编辑器加着色器编译要好几 GB 提交内存。页面文件在 C 盘时 C 盘满了会报“页面文件太小”；同时开着大型游戏时，D3D12 也会因显存或提交不足崩溃。测试前确认资源充足。

## 后续可选

- **头发、裙摆、耳朵、围巾的次级动画**：在动画蓝图里加 AnimDynamics 或 RigidBody 节点，或者配好物理资产后开局部模拟。这些骨骼目前跟随身体，保持静止。
- **卡通着色**：项目里如果有 toon 材质（例如 Gakumas 的 `M_GkmsActor`），通常还需要 Shade/Ramp 等额外贴图和顶点色，要单独准备。
- **斜坡贴地**：给本骨架另写一个脚部 IK（Control Rig 或 Leg IK 节点），替换直通节点。
