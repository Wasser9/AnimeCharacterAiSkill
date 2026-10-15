"""Shared configuration for the UE character scripts (ue_build_playable.py, ue_verify_play.py, ue_diag_pose.py).

The config is a JSON file: $UE_CHAR_CONFIG if set, else ue_config.json next to these scripts. Relative paths in it are
relative to the config file. Missing keys take the defaults below (UE 5.x Third Person template, UE-mannequin bone names).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    'name': 'Character',                       # asset names: SK_<name>, BP_<name>Character, Lvl_<name>_Playable ...
    'root': None,                              # content folder, default /Game/<name>_Playable
    'package_dir': 'finaloutput',              # delivery folder of the origami pipeline
    'fbx': '01_Model/SK_Character_OrigamiUV_UE.fbx',
    'textures_dir': '02_Textures',
    'texture_pattern': '{part}_basecolor.png',
    'parts': [],                               # mesh parts = texture names; a material slot whose name contains a part gets it
    'two_sided': [],                           # parts rendered two-sided (open cloth, hair cards)
    'capsule_half_height': 88.0,               # ~ half the model height in cm
    'capsule_radius': 30.0,
    'mesh_yaw': -90.0,                         # model facing +Y in UE (Blender -Y) -> -90 so it faces the capsule's +X
    'align_bones': [f'{b}_{s}' for s in 'lr' for b in ('clavicle', 'upperarm', 'lowerarm', 'hand')],
    'retarget_suffix': '_RT',                  # suffix of the batch-retargeted copies
    'material': {'roughness': 0.85, 'specular': 0.25, 'self_illum': 0.12},
    'template': {
        'manny': '/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple',
        'abp': '/Game/Characters/Mannequins/Anims/Unarmed/ABP_Unarmed',
        'character': '/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter',
        'game_mode': '/Game/ThirdPerson/Blueprints/BP_ThirdPersonGameMode',
        'level': '/Game/ThirdPerson/Lvl_ThirdPerson',
        'anims_folder': '/Game/Characters/Mannequins/Anims',
    },
    'out_dir': '.',                            # build_report.json, verify_report.json, diag_pose.json, screenshots/
    'verify_bones': ['pelvis', 'head', 'hand_l', 'hand_r', 'upperarm_l', 'upperarm_r', 'foot_l', 'foot_r', 'ball_l', 'ball_r'],
}


def load_config():
    path = os.environ.get('UE_CHAR_CONFIG') or os.path.join(HERE, 'ue_config.json')
    with open(path, encoding='utf-8') as f:
        user = json.load(f)
    cfg = json.loads(json.dumps(DEFAULTS))
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    base = os.path.dirname(os.path.abspath(path))
    for k in ('package_dir', 'out_dir'):
        if not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(base, cfg[k]))
    cfg['root'] = cfg['root'] or f"/Game/{cfg['name']}_Playable"
    cfg['level'] = f"{cfg['root']}/Maps/Lvl_{cfg['name']}_Playable"
    cfg['config_path'] = path
    os.makedirs(cfg['out_dir'], exist_ok=True)
    return cfg
