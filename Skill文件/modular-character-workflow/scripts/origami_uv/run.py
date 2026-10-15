"""Origami UV pipeline entry point: per-body-part ("origami") UV re-layout + texture migration / clean-up for a
character made of several meshes in one Blender file (typically an AI-generated model, already rigged).

  python run.py --config config.yaml                    # full run
  python run.py --config config.yaml --from-stage 3     # re-pack, re-transfer, re-export
  python run.py --config config.yaml --only-stage 7     # renders only
  python run.py --config config.yaml --to-stage 3       # stop after packing (check the layout first)
  python run.py --config config.yaml --texfix-prepare   # cut the migrated atlases into crops for an image model

Stages
  0 inspect   dump meshes / weights / source UVs / source textures from the input blend (Blender)
  1 exposure  per-face visibility from 64 directions (seam placement, texel priority)
  2 unwrap    semantic regions -> seams -> LSCM/Tutte + SLIM flattening
  3 pack      semantic affinity packing, global texel density search
  4 transfer  source basecolor (UV_orig) resampled onto UV_clean; then optional image-model clean-up (texfix) and
              targeted touch-ups (config.touchup)
  5 gate      UV checks, artist guides, data for Blender
  6 export    write UV_clean / UV_panel / textures into a copy of the blend, FBX + GLB + glTF, read back (Blender)
  7 review    8-view renders (checker before/after, textured before/after), close-ups, sheets, metrics.json, report.md
  8 package   tidy delivery folder (config.package_dir): model / textures / UV guides / previews / validation
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'pipeline'))

DEFAULTS = dict(source_uv='UV_orig', source_texture=dict(reference_suffix='_NormalReference'), display_names={},
                texture_size=4096, supersample=2, workers=5, pack_workers=6, transfer_workers=2,
                exposure=dict(directions=64, resolution=2048), compare_previous_uv=None)


def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    base = Path(path).resolve().parent
    for k in ('input_blend', 'output', 'package_dir'):
        p = Path(cfg[k]); cfg[k] = str(p if p.is_absolute() else (base / p).resolve())
    if cfg.get('python_packages') and cfg['python_packages'] not in sys.path:
        sys.path.insert(1, cfg['python_packages'])
    return cfg


def _worker_init(pkgs):
    sys.path.insert(0, str(HERE / 'pipeline'))
    if pkgs:
        sys.path.insert(1, pkgs)


def _call(fn_name, *args):
    import build
    return getattr(build, fn_name)(*args)


def parallel(cfg, fn_name, args_list, workers=None):
    # one fresh process per task: native libraries keep state between tasks (an object unwrapped after another one in
    # the same process came out differently), so results must not depend on the worker count or the task order
    with ProcessPoolExecutor(max_workers=workers or cfg.get('workers', 5), initializer=_worker_init,
                             initargs=(cfg.get('python_packages'),), max_tasks_per_child=1) as ex:
        futs = [ex.submit(_call, fn_name, *a) for a in args_list]
        return [f.result() for f in futs]


def blender(cfg, blend, script, job, log, marker):
    jp = Path(log).with_suffix('.job.json'); jp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding='utf-8')
    cmd = [cfg['blender'], '--factory-startup', '-b', str(blend), '--python', str(HERE / 'pipeline' / script), '--', str(jp)]
    with open(log, 'w', encoding='utf-8', errors='replace') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    txt = Path(log).read_text(encoding='utf-8', errors='replace')
    if marker not in txt:
        raise RuntimeError(f'Blender step failed, see {log}')
    return txt


def owner(entry):
    """Object a metrics entry belongs to ('object' field, or the '<object>__NN' crop name of texfix entries)."""
    return entry.get('object') or str(entry.get('name', '')).split('__')[0]


def merged(old, new, objs):
    """Per-object metrics lists: a run with --objects replaces only the entries of those objects."""
    return [e for e in (old or []) if owner(e) not in objs] + list(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'config.yaml'))
    ap.add_argument('--from-stage', type=int, default=0)
    ap.add_argument('--only-stage', type=int, default=None)
    ap.add_argument('--to-stage', type=int, default=8, help='last stage to run')
    ap.add_argument('--objects', nargs='*')
    ap.add_argument('--texfix-prepare', action='store_true', help='write image-model crops + manifest, then stop')
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    objs = a.objects or cfg['objects']
    out = Path(cfg['output']); work = out / 'work'; work.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    if a.texfix_prepare:
        import gpt_texfix
        src = work / 'transfer' if all((work / 'transfer' / f'{o}_migrated.png').exists() for o in objs) else None
        man = gpt_texfix.prepare(str(out), str(work / 'texfix'), objects=objs, src_dir=str(src) if src else None)
        print(f'{len(man)} crops in {work / "texfix" / "crops"}; put the edited images in {work / "texfix" / "gpt_results"} '
              f'(same file names) and rerun with --from-stage 4')
        return
    stages = [a.only_stage] if a.only_stage is not None else list(range(a.from_stage, a.to_stage + 1))
    metrics_path = out / 'metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}
    (out / 'run_config.resolved.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')
    T = {}
    for st in stages:
        t0 = time.time()
        print(f'== stage {st}', flush=True)
        if st == 0:
            job = dict(out_dir=str(work / 'inspect'), objects=cfg['objects'], source_uv=cfg['source_uv'],
                       reference_suffix=cfg['source_texture'].get('reference_suffix'))
            blender(cfg, cfg['input_blend'], 'bl_inspect.py', job, out / 'logs' / 'stage0_inspect.log', 'INSPECT_DONE')
            assert (work / 'inspect' / 'scene.json').exists(), 'inspect failed'
            scene = json.loads((work / 'inspect' / 'scene.json').read_text(encoding='utf-8'))
            missing = [o for o in cfg['objects'] if o not in scene.get('source_textures', {})]
            if missing:
                raise RuntimeError(f'no source texture found for {missing}; see stage0_inspect.log')
        elif st == 1:
            from exposure import main as expo
            e = cfg['exposure']
            # all character meshes take part (hair occludes the scalp, shoes the feet ...), not only --objects
            expo(str(work / 'inspect'), str(work / 'exposure.npz'), e['directions'], e['resolution'], objects=cfg['objects'])
        elif st == 2:
            metrics['unwrap'] = merged(metrics.get('unwrap'), parallel(cfg, 'stage_unwrap', [(cfg, str(work), o) for o in objs]), objs)
        elif st == 3:
            import build
            nv = len(build.PACK_VARIANTS) if cfg['pack']['mode'] == 'affinity' else 1
            res = parallel(cfg, 'stage_pack', [(cfg, str(work), o, v) for o in objs for v in range(nv)],
                           workers=cfg.get('pack_workers', 6))
            metrics['pack'] = merged(metrics.get('pack'), [build.choose_pack(str(work), o, res) for o in objs], objs)
        elif st == 4:
            metrics['transfer'] = merged(metrics.get('transfer'), parallel(cfg, 'stage_transfer', [(cfg, str(work), str(out), o) for o in objs],
                                                                           workers=cfg.get('transfer_workers', 2)), objs)
            # optional external clean-up (image-model edits per crop, see pipeline/gpt_texfix.py):
            # the migrated textures are kept as work/transfer/*_migrated.png, the cleaned ones replace final/textures
            import shutil
            for o in objs:
                shutil.copy2(out / 'final' / 'textures' / f'{o}_basecolor.png', work / 'transfer' / f'{o}_migrated.png')
            tf = cfg.get('texfix') or {}
            rdir = work / 'texfix' / 'gpt_results'
            if tf.get('enabled', True) and rdir.exists() and any(rdir.glob('*.png')):
                import gpt_texfix
                log = gpt_texfix.apply(str(out), str(work / 'texfix'), str(rdir), str(work / 'texfix' / 'textures_new'),
                                       detail_override=tf.get('detail_override', {}), objects=objs)
                for o in objs:
                    shutil.copy2(work / 'texfix' / 'textures_new' / f'{o}_basecolor.png', out / 'final' / 'textures' / f'{o}_basecolor.png')
                metrics['texfix'] = merged(metrics.get('texfix'), log, objs)
            items = [it for it in (cfg.get('touchup') or []) if it['object'] in objs]   # only re-migrated textures
            if items:
                import touchup
                metrics['touchup'] = merged(metrics.get('touchup'), touchup.run(items, str(work), str(out / 'final' / 'textures')), objs)
        elif st == 5:
            metrics['uv_gate'] = merged(metrics.get('uv_gate'), parallel(cfg, 'uv_gate', [(cfg, str(work), o) for o in objs], workers=2), objs)
            metrics['guides'] = merged(metrics.get('guides'), parallel(cfg, 'stage_guides', [(cfg, str(work), str(out), o) for o in objs], workers=2), objs)
            metrics['apply_data'] = merged(metrics.get('apply_data'), parallel(cfg, 'stage_apply_data', [(cfg, str(work), o) for o in objs]), objs)
            bad = [g for g in metrics['uv_gate'] if g['out_of_bounds_faces'] or g['flipped_faces'] or g['overlap_texels_between_islands']]
            if bad:
                print('UV GATE FAILED', json.dumps(bad, ensure_ascii=False)[:2000])
        elif st == 6:
            job = dict(objects=cfg['objects'], apply_dir=str(work / 'apply'), tex_dir=str(out / 'final' / 'textures'),
                       out_dir=str(out), export=cfg['export'], source_uv=cfg['source_uv'],
                       tool_script=str(HERE / 'pipeline' / 'uv_region_tool.py'))
            blender(cfg, cfg['input_blend'], 'bl_apply.py', job, out / 'logs' / 'stage6_export.log', 'APPLY_DONE')
            metrics['export'] = json.loads((out / 'validation' / 'export_readback.json').read_text(encoding='utf-8'))
        elif st == 7:
            job = dict(objects=cfg['objects'], out_dir=str(out / 'review' / 'renders'), res=768,
                       src_tex_dir=str(work / 'inspect' / 'src_tex'),
                       reference_suffix=cfg['source_texture'].get('reference_suffix'),
                       passes=[dict(name='checker_before', uv='UV_orig', mode='checker'),
                               dict(name='checker_after', uv='UV_clean', mode='checker'),
                               dict(name='texture_before', uv='UV_orig', mode='texture', texture_from='reference'),
                               dict(name='texture_after', uv='UV_clean', mode='texture', texture_from='self')])
            blender(cfg, out / 'final' / cfg['export']['blend_name'], 'bl_render.py', job, out / 'logs' / 'stage7_render.log', 'RENDER_DONE')
            if cfg.get('compare_previous_uv'):
                # an earlier layout stored in the input blend (e.g. a previous delivery) for a three-way comparison
                job2 = dict(job, passes=[dict(name='checker_previous', uv=cfg['compare_previous_uv'], mode='checker')])
                blender(cfg, cfg['input_blend'], 'bl_render.py', job2, out / 'logs' / 'stage7_render_prev.log', 'RENDER_DONE')
            import review, compare_layouts, report
            metrics['review'] = review.make_sheets(cfg, str(work), str(out))
            # close-ups of the texture clean-up: migrated texture vs the final one (image-model clean-up + touch-ups)
            changed = metrics.get('texfix') or metrics.get('touchup')
            if changed and all((work / 'transfer' / f'{o}_migrated.png').exists() for o in objs):
                import shutil
                rd = out / 'review' / 'renders'
                for tag, tdir in (('before', work / 'transfer_closeup'), ('after', out / 'final' / 'textures')):
                    if tag == 'before':
                        tdir.mkdir(exist_ok=True)
                        for o in objs:
                            shutil.copy2(work / 'transfer' / f'{o}_migrated.png', tdir / f'{o}_basecolor.png')
                    blender(cfg, out / 'final' / cfg['export']['blend_name'], 'bl_closeup.py',
                            review.closeup_job(cfg, str(work), rd / f'closeup_{tag}', tdir), out / 'logs' / f'stage7_closeup_{tag}.log',
                            'CLOSEUP_DONE')
                notes = ' + '.join(x for x, on in (('图像模型清理 + 接缝找平', metrics.get('texfix')), ('定点修正', metrics.get('touchup'))) if on)
                metrics['review']['closeup'] = review.make_closeup_sheet(cfg, rd, out / 'review' / 'Closeup_texture_cleanup.png', notes)
            metrics['layout_comparison'] = compare_layouts.run(str(work), str(out / 'validation' / 'layout_comparison.json'),
                                                               cfg['objects'], cfg.get('compare_previous_uv'))
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding='utf-8')
            report.write(str(out), cfg)
        elif st == 8:
            import package
            metrics['package'] = package.build(cfg, out, HERE, lambda *x: blender(cfg, *x))
        T[st] = round(time.time() - t0, 1)
        metrics.setdefault('stage_seconds', {}).update({str(k): v for k, v in T.items()})
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'   stage {st} done in {T[st]}s', flush=True)


if __name__ == '__main__':
    main()
