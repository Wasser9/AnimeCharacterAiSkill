"""Stage 0 (Blender side): dump every character mesh to npz for offline analysis.

Run: blender -b <input.blend> --python bl_inspect.py -- <job.json>
job: {out_dir, objects, source_uv, reference_suffix}
Writes <out_dir>/<object>.npz, <out_dir>/src_tex/<object>.png and <out_dir>/scene.json. Never saves the blend.
The source UV layer (job.source_uv) is stored as uv_UV_orig as well, so every later stage reads one name.
Source texture per object: the packed image of '<object><reference_suffix>' if that object exists (an untouched copy
of the original material), else the image of the object's own material (packed, or copied from its file).
"""
import bpy, bmesh, sys, json, shutil
import numpy as np
from pathlib import Path

arg = Path(sys.argv[sys.argv.index('--') + 1])
job = json.loads(arg.read_text(encoding='utf-8')) if arg.suffix == '.json' else dict(out_dir=str(arg))
out = Path(job['out_dir'])
out.mkdir(parents=True, exist_ok=True)
SRC_UV = job.get('source_uv') or 'UV_orig'
SUFFIX = job.get('reference_suffix') or ''
WANT = set(job.get('objects') or [])
scene = {'objects': {}, 'armatures': {}, 'images': {}}

for img in bpy.data.images:
    scene['images'][img.name] = {
        'size': list(img.size), 'filepath': img.filepath, 'packed': img.packed_file is not None,
        'colorspace': img.colorspace_settings.name}

for ob in bpy.data.objects:
    if ob.type == 'ARMATURE':
        scene['armatures'][ob.name] = {
            b.name: {'head': list(ob.matrix_world @ b.head_local), 'tail': list(ob.matrix_world @ b.tail_local),
                     'parent': b.parent.name if b.parent else None, 'deform': b.use_deform}
            for b in ob.data.bones}
        continue
    if ob.type != 'MESH':
        continue
    me = ob.data
    mw = np.array(ob.matrix_world)
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    co = co @ mw[:3, :3].T + mw[:3, 3]
    nl = len(me.loops)
    lv = np.empty(nl, np.int32); me.loops.foreach_get('vertex_index', lv)
    npoly = len(me.polygons)
    ps = np.empty(npoly, np.int32); me.polygons.foreach_get('loop_start', ps)
    pt = np.empty(npoly, np.int32); me.polygons.foreach_get('loop_total', pt)
    pm = np.empty(npoly, np.int32); me.polygons.foreach_get('material_index', pm)
    uvs = {}
    for layer in me.uv_layers:
        a = np.empty(nl * 2, np.float32); layer.data.foreach_get('uv', a); uvs['uv_' + layer.name] = a.reshape(-1, 2)
    if 'uv_UV_orig' not in uvs and 'uv_' + SRC_UV in uvs:
        uvs['uv_UV_orig'] = uvs['uv_' + SRC_UV]
    # vertex groups -> dense weight matrix (only groups that are used)
    gnames = [g.name for g in ob.vertex_groups]
    W = np.zeros((nv, max(1, len(gnames))), np.float32)
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    attrs = {}
    for at in me.attributes:
        if at.name.startswith('.') or at.data_type not in ('INT', 'FLOAT', 'INT8', 'BOOLEAN'):
            continue
        if at.domain not in ('FACE', 'POINT'):
            continue
        n = len(at.data)
        a = np.empty(n, np.float32 if at.data_type == 'FLOAT' else np.int32 if at.data_type != 'BOOLEAN' else bool)
        at.data.foreach_get('value', a)
        attrs['attr_' + at.domain + '_' + at.name] = a
    mats = []
    for slot in ob.material_slots:
        m = slot.material
        imgs = []
        if m and m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image:
                    uvn = None
                    for lk in n.inputs['Vector'].links:
                        if lk.from_node.type == 'UVMAP':
                            uvn = lk.from_node.uv_map
                    imgs.append({'node': n.name, 'image': n.image.name, 'uv': uvn,
                                 'to': [l.to_socket.name for o_ in n.outputs for l in o_.links]})
        mats.append({'name': m.name if m else None, 'images': imgs})
    bm = bmesh.new(); bm.from_mesh(me)
    nonman = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()
    np.savez_compressed(out / f'{ob.name}.npz', co=co, loop_vert=lv, poly_start=ps, poly_total=pt, poly_mat=pm,
                        weights=W, groups=np.array(gnames), **uvs, **attrs)
    scene['objects'][ob.name] = {
        'verts': nv, 'polys': npoly, 'loops': nl, 'tris_only': bool((pt == 3).all()),
        'uv_layers': [l.name for l in me.uv_layers], 'active_uv': me.uv_layers.active.name if me.uv_layers.active else None,
        'render_uv': [l.name for l in me.uv_layers if l.active_render],
        'materials': mats, 'groups': len(gnames), 'nonmanifold_edges': nonman, 'boundary_edges': boundary,
        'shape_keys': [k.name for k in me.shape_keys.key_blocks] if me.shape_keys else [],
        'modifiers': [(m.name, m.type) for m in ob.modifiers], 'parent': ob.parent.name if ob.parent else None,
        'attributes': [(a.name, a.domain, a.data_type) for a in me.attributes]}
# source textures: the image of '<object><suffix>' (untouched copy of the original material) if present, else the
# object's own material image
tex_dir = out / 'src_tex'; tex_dir.mkdir(exist_ok=True)
scene['source_textures'] = {}


def material_image(ob):
    """First image texture that feeds the Base Color (or, failing that, any image texture) of the object's materials."""
    found = []
    for slot in ob.material_slots:
        m = slot.material
        if not (m and m.use_nodes):
            continue
        for n in m.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image is not None:
                to_base = any(l.to_socket.name in ('Base Color', 'Color') for o_ in n.outputs for l in o_.links)
                found.append((0 if to_base else 1, n.image))
    return sorted(found, key=lambda t: t[0])[0][1] if found else None


def write_image(img, path):
    if img.packed_file is not None:
        path.write_bytes(bytes(img.packed_file.data)); return True
    src = Path(bpy.path.abspath(img.filepath))
    if src.exists():
        shutil.copyfile(src, path); return True        # stage 4 decodes by content, so a .jpg copied as .png is fine
    return False


for ob in bpy.data.objects:
    if ob.type != 'MESH' or (SUFFIX and ob.name.endswith(SUFFIX)):
        continue
    if WANT and ob.name not in WANT:
        continue
    ref = bpy.data.objects.get(ob.name + SUFFIX) if SUFFIX else None
    img = material_image(ref) if ref is not None else None
    if img is None:
        img = material_image(ob)
    path = tex_dir / f'{ob.name}.png'
    if img is not None and write_image(img, path):
        scene['source_textures'][ob.name] = {'image': img.name, 'file': str(path), 'size': list(img.size),
                                             'from': ref.name if ref is not None else ob.name}
(out / 'scene.json').write_text(json.dumps(scene, indent=1, ensure_ascii=False), encoding='utf-8')
print('INSPECT_DONE', list(scene['objects']))
