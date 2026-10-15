"""8-view review renders (checker / textured) for a UV layer. Emission shading, orthographic camera.

blender --factory-startup -b <blend> --python bl_render.py -- <job.json>
job: {objects, out_dir, res, passes:[{name, uv, mode: checker|texture, texture_from: self|reference}]}
"""
import bpy, sys, json
import numpy as np
from pathlib import Path
from mathutils import Vector

job = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
out = Path(job['out_dir']); out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE'
res = job.get('res', 768)
scene.render.resolution_x = res; scene.render.resolution_y = res; scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.view_settings.view_transform = 'Standard'; scene.view_settings.look = 'None'
scene.render.image_settings.file_format = 'PNG'; scene.render.image_settings.color_mode = 'RGB'
if scene.world is None:
    scene.world = bpy.data.worlds.new('W')
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get('Background')
bg.inputs[0].default_value = (0.86, 0.87, 0.89, 1); bg.inputs[1].default_value = 1.0
keep = set(job['objects'])
for o in bpy.data.objects:
    if o.type in ('MESH', 'CURVE', 'EMPTY', 'LIGHT') and o.name not in keep:
        o.hide_render = True
for name in keep:
    bpy.data.objects[name].hide_render = False
# frame the character's bounding box (the 8 views orbit its vertical axis)
_pts = [o.matrix_world @ Vector(c) for o in (bpy.data.objects[n] for n in keep) for c in o.bound_box]
_lo = Vector((min(p.x for p in _pts), min(p.y for p in _pts), min(p.z for p in _pts)))
_hi = Vector((max(p.x for p in _pts), max(p.y for p in _pts), max(p.z for p in _pts)))
CTR = (_lo + _hi) / 2
SIZE = max(_hi.z - _lo.z, _hi.x - _lo.x, _hi.y - _lo.y)
cam_data = bpy.data.cameras.new('OrigamiCam'); cam_data.type = 'ORTHO'; cam_data.ortho_scale = 1.17 * SIZE
cam = bpy.data.objects.new('OrigamiCam', cam_data); scene.collection.objects.link(cam); scene.camera = cam
orig_mats = {n: list(bpy.data.objects[n].data.materials) for n in keep}


def ref_image(name):
    """The source texture: the '<name><reference_suffix>' object's image, else the copy written by stage 0."""
    sfx = job.get('reference_suffix')
    r = bpy.data.objects.get(name + sfx) if sfx else None
    if r is not None:
        for s in r.material_slots:
            for n in s.material.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image:
                    return n.image
    p = Path(job.get('src_tex_dir', '')) / f'{name}.png'
    if p.exists():
        return bpy.data.images.load(str(p), check_existing=True)
    raise RuntimeError(f'no source texture for {name}')


def own_image(name):
    for m in orig_mats[name]:
        for n in m.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image:
                return n.image


for ps in job['passes']:
    temps = []
    for name in keep:
        o = bpy.data.objects[name]
        m = bpy.data.materials.new(f"TMP_{ps['name']}_{name}"); m.use_nodes = True; temps.append(m)
        nt = m.node_tree; nt.nodes.clear(); L = nt.links
        outn = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission')
        L.new(em.outputs[0], outn.inputs['Surface'])
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = ps['uv']
        if ps['mode'] == 'checker':
            ch = nt.nodes.new('ShaderNodeTexChecker')
            ch.inputs['Color1'].default_value = (.07, .09, .11, 1); ch.inputs['Color2'].default_value = (.72, .83, .94, 1)
            ch.inputs['Scale'].default_value = ps.get('scale', 32)
            L.new(uvn.outputs[0], ch.inputs['Vector']); L.new(ch.outputs['Color'], em.inputs[0])
        else:
            tx = nt.nodes.new('ShaderNodeTexImage')
            tx.image = ref_image(name) if ps.get('texture_from') == 'reference' else own_image(name)
            tx.interpolation = 'Linear'
            L.new(uvn.outputs[0], tx.inputs['Vector']); L.new(tx.outputs['Color'], em.inputs[0])
        o.data.materials.clear(); o.data.materials.append(m)
    for i in range(8):
        a = i * 2 * np.pi / 8
        cam.location = CTR + Vector((2.0 * SIZE * np.sin(a), -2.0 * SIZE * np.cos(a), 0.035 * SIZE))
        cam.rotation_euler = (CTR - cam.location).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = str(out / f"{ps['name']}_{i + 1}.png")
        bpy.ops.render.render(write_still=True)
    for name in keep:
        o = bpy.data.objects[name]; o.data.materials.clear()
        for m in orig_mats[name]:
            o.data.materials.append(m)
    for m in temps:
        bpy.data.materials.remove(m)
print('RENDER_DONE')
