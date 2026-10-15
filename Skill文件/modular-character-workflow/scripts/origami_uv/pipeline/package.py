"""Stage 8: assemble the tidy delivery folder (finaloutput)."""
import json, shutil
from pathlib import Path

MARK = '.origami_package'


def readme(cfg, here):
    """Delivery README from pipeline/package_readme.md ($-placeholders filled from the config)."""
    from string import Template
    names = cfg.get('display_names') or {}
    ex = cfg['export']
    rows = '\n'.join(f"| `{o}_basecolor.png` | {names.get(o, o)} |" for o in cfg['objects'])
    return Template((here / 'pipeline' / 'package_readme.md').read_text(encoding='utf-8')).safe_substitute(
        title=cfg.get('title', '角色'), size=cfg.get('texture_size', 4096), n=len(cfg['objects']), texture_rows=rows,
        blend=ex['blend_name'], fbx_ue=ex['fbx_ue'], fbx_embedded=ex['fbx_embedded'], glb=ex['glb'])


def build(cfg, out, here, run_blender):
    out = Path(out)
    pkg = Path(cfg['package_dir'])
    if pkg.exists():
        if not (pkg / MARK).exists():
            raise RuntimeError(f'{pkg} exists and was not created by this pipeline; refusing to overwrite it')
        shutil.rmtree(pkg)
    d = {k: pkg / k for k in ('01_Model', '02_Textures', '03_UV_Guides', '04_Preview', '05_Validation')}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    (pkg / MARK).write_text('created by origami_uv/run.py stage 8; the folder is rebuilt on every run\n', encoding='utf-8')
    objs = cfg['objects']
    for o in objs:
        shutil.copy2(out / 'final' / 'textures' / f'{o}_basecolor.png', d['02_Textures'])
        for suf in ('_UV_Overlay.png', '_UV_RegionID.png', '_UV_Guide.png', '_regions.json'):
            shutil.copy2(out / 'guides' / f'{o}{suf}', d['03_UV_Guides'])
    previews = [out / 'UV_Atlas_Overview.png'] + [out / 'review' / n for n in
                ('Compare_checker_3way.png', 'Compare_texture.png', 'Closeup_texture_cleanup.png', 'Compare_checker.png', 'Region_ID_Sheet.png')]
    for p in previews:
        if p.exists():
            shutil.copy2(p, d['04_Preview'])
    for p in [out / 'metrics.json', out / 'validation' / 'export_readback.json', out / 'validation' / 'layout_comparison.json']:
        if p.exists():
            shutil.copy2(p, d['05_Validation'])
    shutil.copy2(here / 'uvfix.manifest.yaml', d['05_Validation'])
    (pkg / 'README_使用说明.md').write_text(readme(cfg, here), encoding='utf-8')
    if (out / 'report.md').exists():
        shutil.copy2(out / 'report.md', pkg / '验收报告.md')
    job = dict(objects=objs, model_dir=str(d['01_Model']), tex_dir=str(d['02_Textures']), export=cfg['export'],
               apply_dir=str(out / 'work' / 'apply'), report=str(d['05_Validation'] / 'package_readback.json'))
    run_blender(out / 'final' / cfg['export']['blend_name'], 'bl_package.py', job, out / 'logs' / 'stage8_package.log', 'PACKAGE_DONE')
    rep = json.loads((d['05_Validation'] / 'package_readback.json').read_text(encoding='utf-8'))
    files = sorted(str(p.relative_to(pkg)).replace('\\', '/') for p in pkg.rglob('*') if p.is_file() and p.name != MARK)
    size = sum(p.stat().st_size for p in pkg.rglob('*') if p.is_file())
    return dict(package=str(pkg), files=len(files), megabytes=round(size / 2 ** 20, 1), readback=rep['readback'],
                blend_reopen=rep['blend_reopen'])
