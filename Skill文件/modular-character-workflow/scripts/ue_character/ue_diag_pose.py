"""Diagnostics without rendering: component-space bone positions of source (Manny) vs retargeted sequences, both
reference poses and the target retarget-pose rotation offsets. Use it when the play test reports a bad pose (e.g. feet
vertical): if the retargeted sequences are already wrong the retargeter is at fault, if they are right but PIE is wrong
the anim blueprint (Control Rig / IK nodes) is.

  UnrealEditor.exe <Project>.uproject -ExecutePythonScript=<this file> -nullrhi -unattended -nosplash
Settings: ue_config.json / $UE_CHAR_CONFIG. Writes <out_dir>/diag_pose.json.
Compare e.g. ball_l - foot_l: forward and slightly down in a standing pose, in source and target alike.
"""
import json, os, sys, traceback
import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ue_common import load_config

C = load_config()
OUT = os.path.join(C['out_dir'], 'diag_pose.json')
SRC = C['template']['anims_folder']
SFX = C['retarget_suffix']
PAIRS = [(f'{SRC}/Unarmed/MM_Idle', f"{C['root']}/Animations/MM_Idle{SFX}"),
         (f'{SRC}/Unarmed/Walk/MF_Unarmed_Walk_Fwd', f"{C['root']}/Animations/MF_Unarmed_Walk_Fwd{SFX}")]
BONES = ['pelvis', 'thigh_l', 'calf_l', 'foot_l', 'ball_l', 'thigh_r', 'calf_r', 'foot_r', 'ball_r', 'upperarm_l', 'hand_l', 'head']
APE = unreal.AnimPoseExtensions
WS = unreal.AnimPoseSpaces.WORLD
rep = {}


def v(t):
    l = t.translation
    return [round(l.x, 1), round(l.y, 1), round(l.z, 1)]


def pose_dict(pose):
    out = {}
    for b in BONES:
        try:
            out[b] = v(APE.get_bone_pose(pose, b, WS))
        except Exception:
            pass
    return out


try:
    opts = unreal.AnimPoseEvaluationOptions()
    for src, dst in PAIRS:
        a = unreal.load_asset(src); b = unreal.load_asset(dst)
        if a is None or b is None:
            rep[dst.split('/')[-1]] = 'missing'
            continue
        r = {}
        for t in (0.0, 0.3, 0.6):
            r[str(t)] = dict(src=pose_dict(APE.get_anim_pose_at_time(a, t, opts)), dst=pose_dict(APE.get_anim_pose_at_time(b, t, opts)))
        rep[dst.split('/')[-1]] = r
    for name, path in (('manny_ref', C['template']['manny']), ('target_ref', f"{C['root']}/Mesh/SK_{C['name']}")):
        m = unreal.load_asset(path)
        rep[name] = pose_dict(APE.get_reference_pose(m.get_editor_property('skeleton')))
    rtg = unreal.load_asset(f"{C['root']}/Rig/RTG_Manny_to_{C['name']}")
    c = unreal.IKRetargeterController.get_controller(rtg)
    T = unreal.RetargetSourceOrTarget.TARGET
    rep['target_retarget_pose'] = str(c.get_current_retarget_pose_name(T))
    rep['target_pose_offsets'] = {b: str(c.get_rotation_offset_for_retarget_pose_bone(b, T)) for b in BONES}
    rep['status'] = 'OK'
except Exception:
    rep['status'] = 'ERROR'; rep['error'] = traceback.format_exc()
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(rep, f, indent=1)
unreal.log('UECHAR_DIAG_' + rep['status'])
