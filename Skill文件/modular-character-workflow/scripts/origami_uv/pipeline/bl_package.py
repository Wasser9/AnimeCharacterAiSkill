"""Stage 8 (Blender): build the delivery package.

blender --factory-startup -b <output/final/*.blend> --python bl_package.py -- <job.json>
job: {objects, pkg_dir, model_dir, tex_dir, export:{...}, apply_dir}
The blend in the package references the PNGs in <tex_dir> (relative paths, not packed) so edited textures show up
after "Reload" in Blender; the Unreal FBX references the same files relatively.
"""
import bpy, sys, json
import numpy as np
from pathlib import Path

job = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups in the outputs
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bl_export import export_all, readback

model_dir = Path(job['model_dir']); tex_dir = Path(job['tex_dir'])
model_dir.mkdir(parents=True, exist_ok=True)
blend_path = model_dir / job['export']['blend_name']
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)

objs = [bpy.data.objects[n] for n in job['objects']]
retarget = {}
for o in objs:
    png = tex_dir / f'{o.name}_basecolor.png'
    assert png.exists(), png
    for slot in o.material_slots:
        for n in slot.material.node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image is not None:
                img = n.image
                if img.packed_file is not None:
                    img.unpack(method='REMOVE')
                img.filepath = bpy.path.relpath(str(png))
                img.source = 'FILE'
                img.reload()
                retarget[o.name] = dict(image=img.name, filepath=img.filepath, size=list(img.size))
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)

export_all(objs, model_dir, job['export'], gltf_dir=model_dir / 'glTF')

expected_tris = sum(len(o.data.polygons) for o in objs)
uv_expect = np.concatenate([np.load(Path(job['apply_dir']) / f'{o.name}.npz')['loop_uv'] for o in objs])
rb = {kind: readback(model_dir / job['export'][key], kind, uv_expect, expected_tris)
      for kind, key in (('fbx_ue', 'fbx_ue'), ('fbx_embedded', 'fbx_embedded'), ('glb', 'glb'))}

# reopen the packaged blend: textures must resolve from the package
bpy.ops.wm.open_mainfile(filepath=str(blend_path))
chk = {}
for name in job['objects']:
    o = bpy.data.objects[name]
    imgs = [n.image for s in o.material_slots for n in s.material.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
    chk[name] = dict(texture=imgs[0].filepath, exists=Path(bpy.path.abspath(imgs[0].filepath)).exists(),
                     loaded_size=list(imgs[0].size), packed=imgs[0].packed_file is not None,
                     uv_layers=[l.name for l in o.data.uv_layers])
report = dict(retarget=retarget, readback=rb, blend_reopen=chk)
Path(job['report']).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
print('PACKAGE_DONE', json.dumps(report, ensure_ascii=False)[:1500])
