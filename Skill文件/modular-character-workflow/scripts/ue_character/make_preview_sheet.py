"""Turn the play-test captures (<out_dir>/screenshots from ue_verify_play.py) into one labelled preview sheet.
The HDR captures are linear scene colour; they are exposed to a mid-grey average and tone-mapped here.
  python make_preview_sheet.py [out.png]          (plain Python: numpy, opencv, pillow; same ue_config.json)"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ue_common import load_config

C = load_config()
SHOTS = Path(C['out_dir']) / 'screenshots'
VIEWS = [('SP_idle_front', '待机 · 正面'), ('SP_idle_side', '待机 · 侧面'), ('SP_idle_back', '待机 · 背面'),
         ('SP_idle_face', '待机 · 脸部'), ('SP_run_side', '跑动 · 侧面'),
         ('SP_run_front', '跑动 · 正面'), ('SP_jump_side', '起跳 · 侧面')]


def tonemap(h):
    lum = 0.2126 * h[..., 2] + 0.7152 * h[..., 1] + 0.0722 * h[..., 0]
    x = h * (0.18 / max(float(np.exp(np.mean(np.log(lum + 1e-4)))), 1e-4))
    x = x / (1 + x) * 1.6
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def font(size):
    for f in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main(out=Path(C['out_dir']) / 'UE_Playtest_Sheet.png', cols=4, w=480, h=360):
    rows = (len(VIEWS) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * w + (cols + 1) * 10, 60 + rows * (h + 36) + 10), (238, 239, 242))
    d = ImageDraw.Draw(sheet)
    d.text((12, 14), f"{C['name']} 实机测试（PIE，模板第三人称关卡，重定向后的 Manny 动作）", font=font(26), fill=(30, 30, 35))
    for i, (name, label) in enumerate(VIEWS):
        p = SHOTS / f'{name}.hdr'
        if not p.exists():
            continue
        im = cv2.cvtColor(tonemap(cv2.imread(str(p), cv2.IMREAD_UNCHANGED)), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im).resize((w, h), Image.LANCZOS)
        r, c = divmod(i, cols)
        x, y = 10 + c * (w + 10), 60 + r * (h + 36)
        sheet.paste(im, (x, y))
        d.text((x, y + h + 6), label, font=font(20), fill=(40, 40, 45))
    sheet.save(out)
    print(out)


if __name__ == '__main__':
    main(*sys.argv[1:2])
