"""Close-up renders of the textured model (flat emission shading so the texture itself is judged, not the lighting).
job: objects, out_dir, tex_dir (optional: swap every basecolor image for <tex_dir>/<object>_basecolor.png), views."""
import bpy, json, sys
from mathutils import Vector
from pathlib import Path

job = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
out = Path(job['out_dir']); out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x, scene.render.resolution_y = job.get('res', (1400, 1000))
scene.render.film_transparent = False
scene.view_settings.view_transform = 'Standard'
if scene.world is None:
    scene.world = bpy.data.worlds.new('W')
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.86, .87, .89, 1)
keep = set(job['objects'])
for o in bpy.data.objects:
    if o.type in ('MESH', 'LIGHT') and o.name not in keep:
        o.hide_render = True
for n in keep:
    o = bpy.data.objects[n]
    for s in o.material_slots:
        nt = s.material.node_tree
        tex = next(x for x in nt.nodes if x.type == 'TEX_IMAGE')
        if job.get('tex_dir'):
            tex.image = bpy.data.images.load(str(Path(job['tex_dir']) / f'{n}_basecolor.png'), check_existing=False)
        em = nt.nodes.new('ShaderNodeEmission')
        nt.links.new(tex.outputs['Color'], em.inputs[0])
        outn = next(x for x in nt.nodes if x.type == 'OUTPUT_MATERIAL')
        nt.links.new(em.outputs[0], outn.inputs['Surface'])
cd = bpy.data.cameras.new('C'); cd.type = 'ORTHO'
cam = bpy.data.objects.new('C', cd); scene.collection.objects.link(cam); scene.camera = cam
for name, (loc, tgt, sc) in job['views'].items():
    cd.ortho_scale = sc
    cam.location = Vector(loc)
    cam.rotation_euler = (Vector(tgt) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(out / f'{name}.png')
    bpy.ops.render.render(write_still=True)
print('CLOSEUP_DONE')
