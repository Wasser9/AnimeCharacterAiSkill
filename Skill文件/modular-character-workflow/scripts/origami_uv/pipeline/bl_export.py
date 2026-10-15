"""Shared Blender export + read-back helpers (used by bl_apply.py and bl_package.py)."""
import bpy
import numpy as np
from pathlib import Path
from mathutils import Matrix, Vector


def find_rig(objs, rig_name=None):
    """The armature named in the config, else the one the meshes are skinned to (None for static meshes)."""
    if rig_name and bpy.data.objects.get(rig_name) is not None:
        return bpy.data.objects[rig_name]
    for o in objs:
        for mod in o.modifiers:
            if mod.type == 'ARMATURE' and mod.object is not None:
                return mod.object
        if o.parent is not None and o.parent.type == 'ARMATURE':
            return o.parent
    return None


def export_all(objs, out_dir, names, rig_name=None, gltf_dir=None):
    """GLB (embedded), glTF (separate), FBX for Unreal (cm, external textures, relative paths) and FBX with embedded
    textures. UV_clean is channel 0, UV_orig channel 1; only deforming bones, single root, no leaf bones.
    names: dict with keys glb, gltf, fbx_ue, fbx_embedded and optional rig_name, mesh_name."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gltf_dir = Path(gltf_dir) if gltf_dir else out_dir
    gltf_dir.mkdir(parents=True, exist_ok=True)
    source_rig = find_rig(objs, rig_name or names.get('rig_name'))
    temp = bpy.data.collections.new('ORIGAMI_EXPORT_TEMP'); bpy.context.scene.collection.children.link(temp)
    rig = None
    if source_rig is not None:
        rig = source_rig.copy(); rig.data = source_rig.data.copy(); rig.name = 'Armature'; temp.objects.link(rig)
        rig.parent = None; rig.matrix_world = Matrix.Identity(4)
        if rig.animation_data:
            rig.animation_data.action = None
        for pb in rig.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
            for c in pb.constraints:
                c.mute = True
    copies = []
    for o in objs:
        q = o.copy(); q.data = o.data.copy(); temp.objects.link(q); q.parent = None; q.matrix_world = o.matrix_world.copy()
        for mod in list(q.modifiers):
            q.modifiers.remove(mod)
        if rig is not None:
            mod = q.modifiers.new('Skin', 'ARMATURE'); mod.object = rig
        nl = len(q.data.loops)
        orig = np.empty(nl * 2, np.float32); clean = orig.copy()
        q.data.uv_layers['UV_orig'].data.foreach_get('uv', orig); q.data.uv_layers['UV_clean'].data.foreach_get('uv', clean)
        while len(q.data.uv_layers):
            q.data.uv_layers.remove(q.data.uv_layers[-1])
        q.data.uv_layers.new(name='UV_clean').data.foreach_set('uv', clean)
        q.data.uv_layers.new(name='UV_orig').data.foreach_set('uv', orig)
        q.data.uv_layers.active = q.data.uv_layers['UV_clean']; q.data.uv_layers['UV_clean'].active_render = True
        copies.append(q)
    bpy.ops.object.select_all(action='DESELECT')
    for q in copies:
        q.select_set(True)
    if rig is not None:
        rig.select_set(True); bpy.context.view_layer.objects.active = rig
    else:
        bpy.context.view_layer.objects.active = copies[0]
    gp = bpy.ops.export_scene.gltf.get_rna_type().properties
    gopts = {'use_selection': True, 'export_animations': False, 'export_skins': True, 'export_texcoords': True,
             'export_normals': True, 'export_materials': 'EXPORT'}
    if 'export_def_bones' in gp:
        gopts['export_def_bones'] = True
    bpy.ops.export_scene.gltf(filepath=str(out_dir / names['glb']), export_format='GLB', **gopts)
    bpy.ops.export_scene.gltf(filepath=str(gltf_dir / names['gltf']), export_format='GLTF_SEPARATE',
                              export_texture_dir='textures_gltf', **gopts)
    unit0 = bpy.context.scene.unit_settings.scale_length
    if rig is not None:
        rig.data.transform(Matrix.Scale(100, 4))
    for q in copies:
        q.data.transform(q.matrix_world); q.matrix_world = Matrix.Identity(4)
        q.data.transform(Matrix.Scale(100, 4))
    bpy.context.scene.unit_settings.scale_length = .01
    bpy.ops.object.select_all(action='DESELECT')
    for q in copies:
        q.select_set(True)
    bpy.context.view_layer.objects.active = copies[0]; bpy.ops.object.join(); joined = bpy.context.object
    joined.name = names.get('mesh_name') or 'SKM_Character_OrigamiUV'
    if rig is not None:
        rig.select_set(True); bpy.context.view_layer.objects.active = rig
    fopts = dict(use_selection=True, object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False, use_armature_deform_only=True,
                 apply_unit_scale=True, apply_scale_options='FBX_SCALE_UNITS', axis_forward='-Y', axis_up='Z',
                 mesh_smooth_type='FACE', use_mesh_modifiers=True, bake_anim=False)
    bpy.ops.export_scene.fbx(filepath=str(out_dir / names['fbx_ue']), path_mode='RELATIVE', **fopts)
    bpy.ops.export_scene.fbx(filepath=str(out_dir / names['fbx_embedded']), path_mode='COPY', embed_textures=True, **fopts)
    for ob in list(temp.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.collections.remove(temp); bpy.context.scene.unit_settings.scale_length = unit0


def readback(path, kind, uv_expect, expected_tris):
    """Import an exported file into an empty scene and measure it. Leaves the empty scene loaded."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if str(path).lower().endswith('.fbx'):
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))
    # the glTF importer adds an 'Icosphere' bone-display mesh that is not part of the file
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and not (o.name.startswith('Icosphere') and o.parent is None)]
    arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    tris = sum(len(o.data.polygons) for o in meshes)
    uvl = [[l.name for l in o.data.uv_layers] for o in meshes]
    uv0 = np.concatenate([np.array([d.uv[:] for d in o.data.uv_layers[0].data], np.float32) for o in meshes])

    # UV channel 0 compared as a set of values at 1/4096 tolerance (exporters reorder loops; glTF flips V in float)
    def q(a):
        r = np.unique(np.round(a.astype(np.float64) * 4096).astype(np.int64), axis=0)
        return r[:, 0] * 10 ** 6 + r[:, 1]
    qa, qb = q(uv0), q(uv_expect)
    match = float(len(np.intersect1d(qa, qb)) / max(len(qb), 1))
    maxinf = 0; unbound = 0
    for o in meshes:
        for v in o.data.vertices:
            n = sum(1 for g in v.groups if g.weight > 0)
            maxinf = max(maxinf, n); unbound += n == 0
    images = sorted({n.image.filepath for o in meshes for s in o.material_slots if s.material and s.material.use_nodes
                     for n in s.material.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image})
    found = [bool(Path(bpy.path.abspath(p)).exists()) for p in images]
    return dict(triangles=tris, expected_triangles=expected_tris, meshes=len(meshes), uv_layers=uvl,
                uv0_matches_UV_clean=bool(len(uv0) == len(uv_expect) and match > 0.999), uv0_value_match=round(match, 5),
                bones=sum(len(a.data.bones) for a in arms),
                materials=len({s.material.name for o in meshes for s in o.material_slots if s.material}),
                max_influences=maxinf, unbound_vertices=unbound, textures_found=f'{sum(found)}/{len(found)}',
                height=float(max((o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box)))
