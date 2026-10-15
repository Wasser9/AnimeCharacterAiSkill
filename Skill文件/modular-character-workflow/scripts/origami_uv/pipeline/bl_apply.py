"""Stage 7 (Blender): write UV_clean / UV_panel / textures into a copy of the rigged blend, export, read back.

blender --factory-startup -b <input.blend> --python bl_apply.py -- <job.json>
job.json: {objects, apply_dir, tex_dir, out_dir, export:{...}, render:bool}
The input blend is never saved over; the result goes to <out_dir>/final/<blend_name>.
"""
import bpy, sys, json, hashlib
import numpy as np
from pathlib import Path
from mathutils import Matrix, Vector

job = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups in the outputs
OUT = Path(job['out_dir']); FINAL = OUT / 'final'; FINAL.mkdir(parents=True, exist_ok=True)
VAL = OUT / 'validation'; VAL.mkdir(parents=True, exist_ok=True)
objs = [bpy.data.objects[n] for n in job['objects']]
report = {}
# the source UV layer is called UV_orig from here on (renamed in this copy only; the input blend is never saved)
SRC_UV = job.get('source_uv') or 'UV_orig'
for o in objs:
    uvl = o.data.uv_layers
    if 'UV_orig' not in uvl:
        if SRC_UV not in uvl:
            raise RuntimeError(f'{o.name}: no UV layer {SRC_UV!r} (config source_uv)')
        uvl[SRC_UV].name = 'UV_orig'


def feeds_base_color(node):
    return any(l.to_socket.name in ('Base Color', 'Color') for out in node.outputs for l in out.links)


def signature(o):
    me = o.data
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co)
    pv = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', pv)
    uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers['UV_orig'].data.foreach_get('uv', uv)
    W = []
    for v in me.vertices:
        for g in v.groups:
            W.append((v.index, g.group, round(g.weight, 7)))
    h = hashlib.sha256()
    for a in (co, pv, uv, np.array(W, np.float64)):
        h.update(a.tobytes())
    cn = me.attributes.get('custom_normal')
    if cn is not None:
        c = np.empty(len(cn.data) * 2, np.int16); cn.data.foreach_get('value', c); h.update(c.tobytes())
    return h.hexdigest()


before = {o.name: signature(o) for o in objs}

# ------------------------------------------------------------------ write UVs, panels, textures
for o in objs:
    d = np.load(Path(job['apply_dir']) / f'{o.name}.npz')
    me = o.data
    assert len(me.loops) == len(d['loop_uv']), (o.name, len(me.loops), len(d['loop_uv']))
    if 'UV_clean' not in me.uv_layers:
        me.uv_layers.new(name='UV_clean')
    me.uv_layers['UV_clean'].data.foreach_set('uv', d['loop_uv'].ravel())
    me.uv_layers.active = me.uv_layers['UV_clean']
    me.uv_layers['UV_clean'].active_render = True
    at = me.attributes.get('UV_panel')
    if at is None or at.domain != 'FACE':
        if at is not None:
            me.attributes.remove(at)
        at = me.attributes.new('UV_panel', 'INT', 'FACE')
    at.data.foreach_set('value', d['panel'].astype(np.int32))
    o['UV_panel_names'] = json.dumps([str(x) for x in d['names']], ensure_ascii=False)
    o['UV_panel_names_cn'] = json.dumps([str(x) for x in d['cn']], ensure_ascii=False)
    o['UV_layout'] = 'origami semantic layout (UV_clean = channel 0, UV_orig kept)'
    # texture
    tex_path = Path(job['tex_dir']) / f'{o.name}_basecolor.png'
    for slot in o.material_slots:
        m = slot.material
        imgs = [n for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image is not None]
        base = [n for n in imgs if feeds_base_color(n)] or imgs[:1]
        for n in imgs:
            if n not in base and not n.inputs['Vector'].is_linked:
                # other maps (normal, roughness ...) were painted for the source UV: keep them on UV_orig
                uvo = m.node_tree.nodes.new('ShaderNodeUVMap'); uvo.uv_map = 'UV_orig'
                m.node_tree.links.new(uvo.outputs[0], n.inputs['Vector'])
        for n in list(m.node_tree.nodes):
            if n.type == 'TEX_IMAGE' and n.image is not None and n in base:
                old = n.image
                oldname = old.name
                img = bpy.data.images.load(str(tex_path), check_existing=False)
                img.colorspace_settings.name = 'sRGB'
                img.pack()
                n.image = img
                if old.users == 0:
                    bpy.data.images.remove(old)
                img.name = oldname if oldname not in bpy.data.images else f'{o.name}_basecolor.png'
            if n.type == 'UVMAP':
                feeds = [l.to_node for out in n.outputs for l in out.links]
                n.uv_map = 'UV_clean' if not feeds or any(t in base for t in feeds) else 'UV_orig'

# replace the region selection helper with one reading the object's own region names
tool = bpy.data.texts.get('UV部位选择工具.py') or bpy.data.texts.new('UV部位选择工具.py')
tool.from_string(Path(job['tool_script']).read_text(encoding='utf-8'))

after = {o.name: signature(o) for o in objs}
report['preservation'] = {n: before[n] == after[n] for n in before}
assert all(report['preservation'].values()), report['preservation']
blend_path = FINAL / job['export']['blend_name']
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True, copy=False)

# ------------------------------------------------------------------ exports + read back
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bl_export import export_all, readback
export_all(objs, FINAL, job['export'])
expected_tris = sum(len(o.data.polygons) for o in objs)
uv_expect = np.concatenate([np.load(Path(job['apply_dir']) / f'{o.name}.npz')['loop_uv'] for o in objs])
report['readback'] = {kind: readback(FINAL / job['export'][key], kind, uv_expect, expected_tris)
                      for kind, key in (('fbx', 'fbx_ue'), ('glb', 'glb'))}

# reload the saved blend and confirm UV_clean / UV_orig / geometry
bpy.ops.wm.open_mainfile(filepath=str(blend_path))
chk = {}
for name in job['objects']:
    o = bpy.data.objects[name]
    d = np.load(Path(job['apply_dir']) / f'{name}.npz')
    uv = np.empty(len(o.data.loops) * 2, np.float32); o.data.uv_layers['UV_clean'].data.foreach_get('uv', uv)
    chk[name] = dict(UV_clean_equal=bool(np.array_equal(uv.reshape(-1, 2), d['loop_uv'])), signature_equal=signature(o) == before[name],
                     uv_layers=[l.name for l in o.data.uv_layers], render_uv=[l.name for l in o.data.uv_layers if l.active_render])
report['blend_reload'] = chk
(VAL / 'export_readback.json').write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
print('APPLY_DONE', json.dumps(report)[:2000])
