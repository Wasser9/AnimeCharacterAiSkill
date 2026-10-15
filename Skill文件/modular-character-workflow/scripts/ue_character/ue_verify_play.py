"""Play-test the character built by ue_build_playable.py: open its level, start PIE, drive the pawn (idle, run, jump),
sample bone positions, check the pose and capture screenshots.

  UnrealEditor.exe <Project>.uproject -ExecCmds="py <this file>" -unattended -nosplash   (screenshots need the RHI;
  -ExecutePythonScript would quit the editor right after the script returns, before the test has run)
Settings: ue_config.json / $UE_CHAR_CONFIG (ue_common.py). Writes <out_dir>/verify_report.json and <out_dir>/screenshots
(<view>_base.png = base colour, <view>.hdr = linear scene colour; make_preview_sheet.py tone-maps them into one sheet).
"""
import json, math, os, sys, traceback
import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ue_common import load_config

C = load_config()
REPORT = os.path.join(C['out_dir'], 'verify_report.json')
LEVEL = C['level']
BONES = list(C['verify_bones'])
HALF = float(C['capsule_half_height'])
K = HALF / 86.0                   # camera distances below were tuned for a 172 cm capsule
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
rep = dict(level=LEVEL, samples=[], shots=[])
state = dict(t=0.0, t0=None, phase='start', handle=None, ended_at=None, frames=0, wait_frames=0)


def v3(v):
    return [round(v.x, 1), round(v.y, 1), round(v.z, 1)]


def sample(tag, world, pawn):
    mesh = pawn.get_editor_property('mesh')
    cmc = pawn.get_editor_property('character_movement')
    ai = mesh.get_anim_instance()
    s = dict(tag=tag, t=round(state['t'] - state['t0'], 2), location=v3(pawn.get_actor_location()),
             yaw=round(pawn.get_actor_rotation().yaw, 1), speed=round(pawn.get_velocity().length(), 1),
             falling=bool(cmc.is_falling()), anim_instance=ai.get_class().get_name() if ai else None,
             bones={b: v3(mesh.get_socket_location(b)) for b in BONES})
    rep['samples'].append(s)


SHOT_DIR = os.path.join(C['out_dir'], 'screenshots')


def shot(world, pawn, name, yaw=0.0, dist=380.0, height=10.0, target_z=0.0, fov=40.0):
    """Two captures per view with the SP_ReviewCapture actor: base colour (texture + pose, no exposure involved) and
    linear scene colour HDR, tone-mapped afterwards (a one-off capture has no eye adaptation and saturates to white)."""
    if 'cap' not in state:
        actor = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.SceneCapture2D)[0]
        state['cap_actor'] = actor
        state['cap'] = actor.get_editor_property('capture_component2d')
        state['rt_ldr'] = unreal.RenderingLibrary.create_render_target2d(world, 1280, 960, unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
        state['rt_hdr'] = unreal.RenderingLibrary.create_render_target2d(world, 1280, 960, unreal.TextureRenderTargetFormat.RTF_RGBA16F)
    cap = state['cap']
    cap.set_editor_property('fov_angle', fov)
    dist, height, target_z = dist * K, height * K, target_z * K
    base = pawn.get_actor_location()
    a = math.radians(pawn.get_actor_rotation().yaw + yaw)
    loc = unreal.Vector(base.x + dist * math.cos(a), base.y + dist * math.sin(a), base.z + height)
    tgt = unreal.Vector(base.x, base.y, base.z + target_z)
    state['cap_actor'].set_actor_location_and_rotation(loc, unreal.MathLibrary.find_look_at_rotation(loc, tgt), False, False)
    os.makedirs(SHOT_DIR, exist_ok=True)
    for src, rt, fname in ((unreal.SceneCaptureSource.SCS_BASE_COLOR, state['rt_ldr'], f'{name}_base.png'),
                           (unreal.SceneCaptureSource.SCS_SCENE_COLOR_HDR, state['rt_hdr'], f'{name}.hdr')):
        cap.set_editor_property('capture_source', src)
        cap.set_editor_property('texture_target', rt)
        cap.capture_scene()
        unreal.RenderingLibrary.export_render_target(world, rt, os.path.abspath(SHOT_DIR), fname)
    rep['shots'].append(name)


def face(pawn, yaw_offset):
    pc = pawn.get_controller()
    if pc:
        pc.set_control_rotation(unreal.Rotator(roll=0, pitch=-12, yaw=pawn.get_actor_rotation().yaw + yaw_offset))


def checks():
    """Pose sanity checks on the samples (catch retarget problems a PASSED build does not show)."""
    S = {s['tag']: s for s in rep['samples']}
    out = {}
    if 'idle' in S:
        b = S['idle']['bones']; g = S['idle']['location'][2] - HALF
        if 'hand_l' in b and 'upperarm_l' in b:
            out['idle_arms_down'] = (b['upperarm_l'][2] - b['hand_l'][2]) > 0.10 * 2 * HALF
        for sd in 'lr':
            f, ba = b.get(f'foot_{sd}'), b.get(f'ball_{sd}')
            if f and ba:
                dz = ba[2] - f[2]; dh = math.hypot(ba[0] - f[0], ba[1] - f[1])
                out[f'idle_foot_{sd}_not_vertical'] = dh > abs(dz) * 0.6
        if 'head' in b:
            out['idle_head_height_ok'] = 0.5 * 2 * HALF < b['head'][2] - g < 1.1 * 2 * HALF
    if 'run_a' in S:
        out['run_speed'] = S['run_a']['speed'] > 100
    if 'run_a' in S and 'run_b' in S and 'foot_l' in S['run_a']['bones']:
        a, b = S['run_a'], S['run_b']
        rel = lambda s, n: [s['bones'][n][i] - s['location'][i] for i in range(3)]
        out['run_legs_move'] = math.dist(rel(a, 'foot_l'), rel(b, 'foot_l')) > 5 * K
    if 'jump' in S:
        out['jump_airborne'] = S['jump']['falling']
    return out


def finish():
    unreal.unregister_slate_post_tick_callback(state['handle'])
    if rep.get('status') == 'PASSED':
        rep['checks'] = checks()
        if not all(rep['checks'].values()):
            rep['status'] = 'CHECKS_FAILED'

    with open(REPORT, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    unreal.log('UECHAR_VERIFY_' + rep.get('status', 'DONE'))
    if os.environ.get('UECHAR_KEEP_EDITOR') != '1':
        unreal.SystemLibrary.quit_editor()


def tick(dt):
    # the first editor frames can report very long deltas, so start-up waits count frames, the test itself uses time
    state['frames'] += 1
    state['t'] += min(dt, 0.1)
    try:
        if state['phase'] == 'start':
            if state['frames'] > 30:
                state['phase'] = 'loading'          # load_level pumps Slate, which re-enters this callback
                ok = les.load_level(LEVEL)
                unreal.log(f'[UECHAR] level loaded: {ok}')
                # a scene capture for the screenshots (PIE viewport shots come out blank in an unattended editor and
                # Python cannot spawn into the PIE world): placed in the editor world, copied into PIE, never saved
                eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
                cap = eas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 300), unreal.Rotator())
                cap.set_actor_label('SP_ReviewCapture')
                cc = cap.get_editor_property('capture_component2d')
                cc.set_editor_property('capture_every_frame', False)
                cc.set_editor_property('capture_on_movement', False)
                state['editor_cap'] = cap
                les.editor_request_begin_play()
                state['phase'] = 'wait_pawn'
            return
        if state['phase'] == 'ended':
            if state['t'] - state['ended_at'] > 2.0:
                state['phase'] = 'done'
                if state.get('editor_cap'):
                    unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(state['editor_cap'])
                finish()
            return
        if state['phase'] in ('done', 'loading'):
            return
        world = ues.get_game_world()
        pawn = unreal.GameplayStatics.get_player_pawn(world, 0) if world else None
        if state['phase'] == 'wait_pawn':
            state['wait_frames'] += 1
            if pawn is not None:
                state['t0'] = state['t']; state['phase'] = 'run'
                rep['pawn'] = pawn.get_class().get_name()
                rep['game_mode'] = unreal.GameplayStatics.get_game_mode(world).get_class().get_name()
                unreal.log(f"[UECHAR] PIE pawn {rep['pawn']} game mode {rep['game_mode']}")
            elif state['wait_frames'] > 3000:
                rep['status'] = 'NO_PAWN'; state['phase'] = 'done'; finish()
            return
        e = state['t'] - state['t0']
        k = rep.setdefault('_k', 0)
        if 3.5 <= e < 7.5:                       # walk / run forward
            pawn.add_movement_input(unreal.Vector(1, 0, 0), 1.0, False)
        script = [(1.0, lambda: face(pawn, 180)),
                  (2.5, lambda: sample('idle', world, pawn)),
                  (2.6, lambda: shot(world, pawn, 'SP_idle_front', yaw=0)),
                  (2.8, lambda: shot(world, pawn, 'SP_idle_side', yaw=90)),
                  (3.0, lambda: shot(world, pawn, 'SP_idle_back', yaw=180)),
                  (3.2, lambda: shot(world, pawn, 'SP_idle_face', yaw=0, dist=110, height=40, target_z=46, fov=30)),
                  (3.4, lambda: face(pawn, 0)),
                  (5.0, lambda: sample('run_a', world, pawn)),
                  (5.25, lambda: sample('run_b', world, pawn)),
                  (5.35, lambda: shot(world, pawn, 'SP_run_side', yaw=90)),
                  (5.6, lambda: shot(world, pawn, 'SP_run_front', yaw=0)),
                  (7.6, lambda: face(pawn, 160)),
                  (9.5, lambda: sample('stop', world, pawn)),
                  (9.6, lambda: pawn.jump()),
                  (9.9, lambda: sample('jump', world, pawn)),
                  (9.95, lambda: shot(world, pawn, 'SP_jump_side', yaw=90)),
                  (12.0, lambda: sample('landed', world, pawn))]
        while k < len(script) and e >= script[k][0]:
            script[k][1](); k += 1
        rep['_k'] = k
        if k >= len(script) and e > 12.5:
            rep.pop('_k', None)
            rep['status'] = 'PASSED'
            les.editor_request_end_play()
            state['phase'] = 'ended'; state['ended_at'] = state['t']
    except Exception:
        rep['status'] = 'ERROR'; rep['error'] = traceback.format_exc()
        unreal.log_error(rep['error'])
        state['phase'] = 'done'
        try:
            les.editor_request_end_play()
        except Exception:
            pass
        finish()


state['handle'] = unreal.register_slate_post_tick_callback(tick)
unreal.log('[UECHAR] play test armed')
