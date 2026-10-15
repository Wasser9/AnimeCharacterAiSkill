"""Build a playable third-person character from a delivered, rigged character FBX inside a UE 5.x project.

Everything goes to config.root (default /Game/<name>_Playable); the template content it relies on (/Game/Characters,
/Game/Input, /Game/ThirdPerson, /Game/LevelPrototyping) is the stock Third Person template copied into the project.
Project-wide settings (default map / game mode) are not touched: the level carries its own GameMode override.
Settings: ue_config.json next to this file or $UE_CHAR_CONFIG (see ue_common.py, ue_config.example.json).

Run with the editor closed, after deleting <Content>/<root> and its __ExternalActors__/__ExternalObjects__ folders
(needs the editor's Slate application: the batch retarget syncs the Content Browser, so a -run=pythonscript
commandlet crashes; -nullrhi keeps it off the GPU):
  UnrealEditor.exe <Project>.uproject -ExecutePythonScript=<this file> -nullrhi -unattended -nosplash
The editor quits when the script is done. Writes <out_dir>/build_report.json (updated after every step).
"""
import json, math, os, sys, traceback
import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ue_common import load_config

C = load_config()
NAME = C['name']
FBX = os.path.join(C['package_dir'], C['fbx'])
TEX = os.path.join(C['package_dir'], C['textures_dir'])
PARTS = list(C['parts'])
TWO_SIDED = set(C['two_sided'])
ROOT = C['root']
TPL = C['template']
MANNY, ABP_SRC, CHAR_SRC, GM_SRC, LVL_SRC = TPL['manny'], TPL['abp'], TPL['character'], TPL['game_mode'], TPL['level']
SUFFIX = C['retarget_suffix']
CAPSULE_HALF = float(C['capsule_half_height'])
CAPSULE_RADIUS = float(C['capsule_radius'])

EAL = unreal.EditorAssetLibrary
AT = unreal.AssetToolsHelpers.get_asset_tools()
report = dict(root=ROOT, config=C['config_path'], steps=[])
REPORT = os.path.join(C['out_dir'], 'build_report.json')


def write_report():
    with open(REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


def step(name, **kw):
    report['steps'].append(dict(step=name, **kw))
    unreal.log(f'[UECHAR] {name} {kw}')
    write_report()


def import_task(filename, dest_path, dest_name, options=None, factory=None):
    t = unreal.AssetImportTask()
    t.filename = filename; t.destination_path = dest_path; t.destination_name = dest_name
    t.automated = True; t.replace_existing = True; t.save = True
    if options is not None:
        t.options = options
    if factory is not None:
        t.factory = factory
    AT.import_asset_tasks([t])
    return [unreal.load_asset(p) for p in t.imported_object_paths]


def clean():
    if EAL.does_directory_exist(ROOT):
        try:            # loading half-built assets for deletion can log errors; the deletion itself still goes through
            EAL.delete_directory(ROOT)
        except RuntimeError as e:
            unreal.log_warning(f'[UECHAR] delete_directory reported: {str(e)[:300]}')
        left = EAL.list_assets(ROOT, recursive=True, include_folder=False)
        if left:
            raise RuntimeError(f'could not clear {ROOT}: {left[:5]}')
    # leftovers of an interrupted batch retarget (in 5.7 it writes its copies to the /Game root)
    for folder, rec in (('/Game', False), (TPL['anims_folder'], True)):
        for p in EAL.list_assets(folder, recursive=rec, include_folder=False):
            if p.split('.')[0].endswith(SUFFIX):
                try:
                    EAL.delete_asset(p.split('.')[0])
                except RuntimeError as e:
                    unreal.log_warning(f'[UECHAR] delete_asset reported: {str(e)[:300]}')
    for sub in ('Mesh', 'Textures', 'Materials', 'Rig', 'Animations', 'Blueprints', 'Maps'):
        EAL.make_directory(f'{ROOT}/{sub}')


def import_mesh():
    ui = unreal.FbxImportUI()
    ui.automated_import_should_detect_type = False
    ui.mesh_type_to_import = unreal.FBXImportType.FBXIT_SKELETAL_MESH
    ui.import_as_skeletal = True
    ui.import_mesh = True; ui.import_animations = False
    ui.import_materials = False; ui.import_textures = False
    ui.create_physics_asset = True
    sk = ui.skeletal_mesh_import_data
    for n, v in [('import_uniform_scale', 1.0), ('convert_scene', True), ('force_front_x_axis', False),
                 ('convert_scene_unit', False), ('normal_import_method', unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS),
                 ('import_morph_targets', False), ('use_t0_as_ref_pose', False), ('preserve_smoothing_groups', True)]:
        sk.set_editor_property(n, v)
    assets = import_task(FBX, f'{ROOT}/Mesh', f'SK_{NAME}', ui, unreal.FbxFactory())
    mesh = next(a for a in assets if isinstance(a, unreal.SkeletalMesh))
    EAL.save_directory(f'{ROOT}/Mesh', only_if_is_dirty=False, recursive=True)     # skeleton + physics asset too
    b = mesh.get_bounds()
    skel = mesh.get_editor_property('skeleton')
    step('import_mesh', mesh=mesh.get_path_name(), skeleton=skel.get_path_name(),
         bounds_extent=[round(b.box_extent.x, 1), round(b.box_extent.y, 1), round(b.box_extent.z, 1)],
         bounds_origin=[round(b.origin.x, 1), round(b.origin.y, 1), round(b.origin.z, 1)],
         slots=[str(s.material_slot_name) for s in mesh.get_editor_property('materials')],
         physics_asset=str(mesh.get_editor_property('physics_asset').get_name() if mesh.get_editor_property('physics_asset') else None))
    return mesh


def make_materials(mesh):
    # master: lit base colour with a little self-illumination so the shadow side keeps the painted colour
    master = AT.create_asset(f'M_{NAME}_Character', f'{ROOT}/Materials', unreal.Material, unreal.MaterialFactoryNew())
    mp = C['material']
    MEL = unreal.MaterialEditingLibrary
    tex = MEL.create_material_expression(master, unreal.MaterialExpressionTextureSampleParameter2D, -600, 0)
    tex.set_editor_property('parameter_name', 'BaseColor')
    tex.set_editor_property('sampler_type', unreal.MaterialSamplerType.SAMPLERTYPE_COLOR)
    MEL.connect_material_property(tex, 'RGB', unreal.MaterialProperty.MP_BASE_COLOR)
    rough = MEL.create_material_expression(master, unreal.MaterialExpressionScalarParameter, -600, 250)
    rough.set_editor_property('parameter_name', 'Roughness'); rough.set_editor_property('default_value', float(mp['roughness']))
    MEL.connect_material_property(rough, '', unreal.MaterialProperty.MP_ROUGHNESS)
    spec = MEL.create_material_expression(master, unreal.MaterialExpressionScalarParameter, -600, 350)
    spec.set_editor_property('parameter_name', 'Specular'); spec.set_editor_property('default_value', float(mp['specular']))
    MEL.connect_material_property(spec, '', unreal.MaterialProperty.MP_SPECULAR)
    glow = MEL.create_material_expression(master, unreal.MaterialExpressionScalarParameter, -600, 450)
    glow.set_editor_property('parameter_name', 'SelfIllum'); glow.set_editor_property('default_value', float(mp['self_illum']))
    mul = MEL.create_material_expression(master, unreal.MaterialExpressionMultiply, -300, 450)
    MEL.connect_material_expressions(tex, 'RGB', mul, 'A'); MEL.connect_material_expressions(glow, '', mul, 'B')
    MEL.connect_material_property(mul, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    master.set_editor_property('two_sided', True)
    master.set_editor_property('used_with_skeletal_mesh', True)
    MEL.recompile_material(master); EAL.save_loaded_asset(master)

    mis = {}
    for part in PARTS:
        t = next(a for a in import_task(os.path.join(TEX, C['texture_pattern'].format(part=part)), f'{ROOT}/Textures', f'T_{part}_BaseColor',
                                        factory=unreal.TextureFactory()) if isinstance(a, unreal.Texture2D))
        t.set_editor_property('srgb', True)
        t.set_editor_property('lod_group', unreal.TextureGroup.TEXTUREGROUP_CHARACTER)
        EAL.save_loaded_asset(t)
        mi = AT.create_asset(f'MI_{part}', f'{ROOT}/Materials', unreal.MaterialInstanceConstant,
                             unreal.MaterialInstanceConstantFactoryNew())
        MEL.set_material_instance_parent(mi, master)
        MEL.set_material_instance_texture_parameter_value(mi, 'BaseColor', t)
        if part not in TWO_SIDED:
            ov = mi.get_editor_property('base_property_overrides')
            ov.set_editor_property('override_two_sided', True); ov.set_editor_property('two_sided', False)
            mi.set_editor_property('base_property_overrides', ov)
        MEL.update_material_instance(mi); EAL.save_loaded_asset(mi)
        mis[part] = mi
    slots = []
    for s in mesh.get_editor_property('materials'):
        key = str(s.material_slot_name)
        part = next((p for p in sorted(PARTS, key=len, reverse=True) if p in key), None)
        if part is None:
            raise RuntimeError(f'material slot {key} does not name a part')
        v = unreal.SkeletalMaterial()
        v.set_editor_property('material_slot_name', s.material_slot_name)
        v.set_editor_property('material_interface', mis[part])
        slots.append(v)
    mesh.set_editor_property('materials', slots)
    EAL.save_loaded_asset(mesh)
    step('materials', slots={str(s.material_slot_name): s.material_interface.get_name() for s in slots})


def make_ik_rig(name, mesh):
    rig = AT.create_asset(name, f'{ROOT}/Rig', unreal.IKRigDefinition, unreal.IKRigDefinitionFactory())
    c = unreal.IKRigController.get_controller(rig)
    c.set_skeletal_mesh(mesh)
    c.apply_auto_generated_retarget_definition()
    c.apply_auto_fbik()
    EAL.save_loaded_asset(rig)
    chains = {}
    for ch in c.get_retarget_chains():
        n = str(ch.get_editor_property('chain_name'))
        chains[n] = [str(c.get_retarget_chain_start_bone(n)), str(c.get_retarget_chain_end_bone(n))]
    step('ik_rig', rig=rig.get_path_name(), retarget_root=str(c.get_retarget_root()), chains=chains)
    return rig


def make_retargeter(src_rig, tgt_rig, src_mesh, tgt_mesh):
    rtg = AT.create_asset(f'RTG_Manny_to_{NAME}', f'{ROOT}/Rig', unreal.IKRetargeter, unreal.IKRetargetFactory())
    c = unreal.IKRetargeterController.get_controller(rtg)
    S, T = unreal.RetargetSourceOrTarget.SOURCE, unreal.RetargetSourceOrTarget.TARGET
    c.set_ik_rig(S, src_rig); c.set_ik_rig(T, tgt_rig)
    c.set_preview_mesh(S, src_mesh); c.set_preview_mesh(T, tgt_mesh)
    if c.get_num_retarget_ops() == 0:
        c.add_default_ops()
    c.assign_ik_rig_to_all_ops(S, src_rig); c.assign_ik_rig_to_all_ops(T, tgt_rig)
    c.auto_map_chains(unreal.AutoMapChainType.FUZZY, True)
    # AI models are usually bound in a T-pose, the mannequin in an A-pose: bring only the configured bones (the arms) to
    # the source's chain directions. Aligning every bone also turned the feet ~56 degrees toe-down on the example model.
    arms = list(C['align_bones'])
    if arms:
        c.auto_align_bones(arms, unreal.RetargetAutoAlignMethod.CHAIN_TO_CHAIN, T)
    EAL.save_loaded_asset(rtg)
    ops = [str(c.get_op_name(i)) for i in range(c.get_num_retarget_ops())]
    offs = {}
    for b in arms[:4] + ['thigh_l', 'calf_l', 'foot_l', 'ball_l']:
        try:
            q = c.get_rotation_offset_for_retarget_pose_bone(b, T)
            offs[b] = round(2 * math.degrees(math.acos(min(1.0, abs(q.w)))), 1)
        except Exception:
            pass
    step('retargeter', asset=rtg.get_path_name(), ops=ops, target_pose_offset_deg=offs)
    return rtg


def retarget_anims(src_mesh, tgt_mesh, rtg):
    ad = [EAL.find_asset_data(ABP_SRC)]
    out = unreal.IKRetargetBatchOperation.duplicate_and_retarget(ad, src_mesh, tgt_mesh, rtg, '', '', '', SUFFIX, True, True)
    made = [str(a.package_name) for a in out]
    # the duplicated anim blueprint is not part of the returned list (5.7); it lands in the /Game root like the rest
    for p in EAL.list_assets('/Game', recursive=False, include_folder=False):
        p = p.split('.')[0]
        if p.endswith(SUFFIX) and p not in made:
            made.append(p)
    # gather the copies (wherever the batch operation put them) in one folder
    moved = {}
    for p in made:
        dst = f'{ROOT}/Animations/{p.split("/")[-1]}'
        if p != dst and EAL.rename_asset(p, dst):
            moved[p] = dst
    reds = []
    for folder, rec in (('/Game', False), (TPL['anims_folder'], True)):
        for p in EAL.list_assets(folder, recursive=rec, include_folder=False):
            if str(EAL.find_asset_data(p).asset_class_path.asset_name) == 'ObjectRedirector':
                reds.append(unreal.load_asset(p))
    if reds:
        AT.fixup_referencers(reds)
    EAL.save_directory(f'{ROOT}/Animations', only_if_is_dirty=False, recursive=True)
    step('retarget_anims', count=len(made), source_paths=sorted(made)[:3], moved=sorted(moved.values()), redirectors_fixed=len(reds))
    abp = unreal.load_asset(next(v for v in moved.values() if v.split('/')[-1].startswith('ABP_')))
    disable_foot_ik(abp)
    return abp


def disable_foot_ik(abp):
    """ABP_Unarmed runs CR_Mannequin_FootIK, which is written for the mannequin's bone axes and turned the example model's
    feet vertical. The node is pointed at an empty Control Rig instead: with no hierarchy it copies the input pose through.
    (Its Alpha is a pin, so setting the struct value is ignored; a node without any rig class outputs the ref pose.)"""
    path = f'{ROOT}/Rig/CR_{NAME}_PassThrough'
    cr = unreal.ControlRigBlueprintFactory.create_new_control_rig_asset(path)
    unreal.BlueprintEditorLibrary.compile_blueprint(cr)
    EAL.save_loaded_asset(cr)
    pkg = abp.get_outermost()
    done = []
    for node in unreal.ObjectIterator(unreal.EdGraphNode):
        if node.get_outermost() != pkg or 'ControlRig' not in node.get_class().get_name():
            continue
        st = node.get_editor_property('node')
        before = str(st.get_editor_property('control_rig_class'))
        st.set_editor_property('control_rig_class', cr.generated_class())
        node.set_editor_property('node', st)
        done.append(dict(node=node.get_name(), rig_before=before,
                         rig_after=str(node.get_editor_property('node').get_editor_property('control_rig_class'))))
    unreal.BlueprintEditorLibrary.compile_blueprint(abp)
    EAL.save_loaded_asset(abp)
    step('disable_foot_ik', abp=abp.get_path_name(), pass_through_rig=path, nodes=done)


def subobject_templates(bp):
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    lib = unreal.SubobjectDataBlueprintFunctionLibrary
    out = {}
    for h in sds.k2_gather_subobject_data_for_blueprint(bp):
        d = sds.k2_find_subobject_data_from_handle(h)
        o = lib.get_object(d)
        if o is not None:
            out[o.get_name()] = o
    return out


def make_character(mesh, abp):
    bp = EAL.duplicate_asset(CHAR_SRC, f'{ROOT}/Blueprints/BP_{NAME}Character')
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    cdo = unreal.get_default_object(bp.generated_class())
    comp = cdo.get_editor_property('mesh')
    cap = cdo.get_editor_property('capsule_component')
    before = dict(mesh=str(comp.get_editor_property('skeletal_mesh_asset').get_name() if comp.get_editor_property('skeletal_mesh_asset') else None),
                  loc=str(comp.get_editor_property('relative_location')), rot=str(comp.get_editor_property('relative_rotation')),
                  half=cap.get_editor_property('capsule_half_height'), radius=cap.get_editor_property('capsule_radius'))
    comp.set_editor_property('skeletal_mesh_asset', mesh)
    comp.set_editor_property('animation_mode', unreal.AnimationMode.ANIMATION_BLUEPRINT)
    comp.set_editor_property('anim_class', abp.generated_class())
    comp.set_editor_property('relative_location', unreal.Vector(0, 0, -CAPSULE_HALF))
    comp.set_editor_property('relative_rotation', unreal.Rotator(roll=0, pitch=0, yaw=float(C['mesh_yaw'])))
    comp.set_editor_property('visibility_based_anim_tick_option', unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES)
    cap.set_editor_property('capsule_half_height', CAPSULE_HALF)
    cap.set_editor_property('capsule_radius', CAPSULE_RADIUS)
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    EAL.save_loaded_asset(bp)
    cdo = unreal.get_default_object(bp.generated_class())
    comp = cdo.get_editor_property('mesh')
    after = dict(mesh=comp.get_editor_property('skeletal_mesh_asset').get_name(),
                 anim=comp.get_editor_property('anim_class').get_name(),
                 loc=str(comp.get_editor_property('relative_location')),
                 half=cdo.get_editor_property('capsule_component').get_editor_property('capsule_half_height'))
    step('character', bp=bp.get_path_name(), template_before=before, after=after)
    return bp


def make_game_mode(char_bp):
    gm = EAL.duplicate_asset(GM_SRC, f'{ROOT}/Blueprints/BP_{NAME}GameMode')
    unreal.BlueprintEditorLibrary.compile_blueprint(gm)
    cdo = unreal.get_default_object(gm.generated_class())
    cdo.set_editor_property('default_pawn_class', char_bp.generated_class())
    unreal.BlueprintEditorLibrary.compile_blueprint(gm)
    EAL.save_loaded_asset(gm)
    cdo = unreal.get_default_object(gm.generated_class())
    step('game_mode', gm=gm.get_path_name(), pawn=cdo.get_editor_property('default_pawn_class').get_name(),
         controller=cdo.get_editor_property('player_controller_class').get_name())
    return gm


def make_level(gm):
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    path = C['level']
    ok = les.new_level_from_template(path, LVL_SRC)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    ws = world.get_world_settings()
    ws.set_editor_property('default_game_mode', gm.generated_class())
    # remove any pawn already placed in the template level so the GameMode spawns ours at the PlayerStart
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    removed = [a.get_actor_label() for a in eas.get_all_level_actors() if isinstance(a, unreal.Character)]
    for a in [a for a in eas.get_all_level_actors() if isinstance(a, unreal.Character)]:
        eas.destroy_actor(a)
    starts = [a.get_actor_label() for a in eas.get_all_level_actors() if isinstance(a, unreal.PlayerStart)]
    saved = les.save_current_level()
    step('level', path=path, created=bool(ok), saved=bool(saved), world=world.get_path_name(), player_starts=starts,
         removed_pawns=removed, game_mode=ws.get_editor_property('default_game_mode').get_name())


try:
    unreal.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    clean()
    manny = unreal.load_asset(MANNY)
    mesh = import_mesh()
    make_materials(mesh)
    src_rig = make_ik_rig(f'IK_Manny_{NAME}', manny)
    tgt_rig = make_ik_rig(f'IK_{NAME}', mesh)
    rtg = make_retargeter(src_rig, tgt_rig, manny, mesh)
    abp = retarget_anims(manny, mesh, rtg)
    char_bp = make_character(mesh, abp)
    gm = make_game_mode(char_bp)
    make_level(gm)
    EAL.save_directory(ROOT, only_if_is_dirty=True, recursive=True)
    report['status'] = 'PASSED'
except Exception:
    report['status'] = 'FAILED'
    report['error'] = traceback.format_exc()
    unreal.log_error(report['error'])
write_report()
unreal.log('UECHAR_BUILD_' + report['status'])
if os.environ.get('UECHAR_KEEP_EDITOR') != '1':
    unreal.SystemLibrary.quit_editor()
