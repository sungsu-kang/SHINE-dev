"""Find which saved checkpoint produced an existing set of denoised frames.

main.py --test=1 predicts with the best-validation checkpoint, which is not necessarily the
newest file in model/. Each checkpoint re-denoises a few frames (fp16, like production) and is
compared with the saved outputs; the producer gives a median absolute difference of ~0.

Usage: python find_checkpoint.py RAW_DIR DENOISED_DIR MODEL_DIR [--model 3x3_blind] [--frames 0 50]
"""
import argparse
import glob
import os
import numpy as np
from common import list_tifs, read_stack, read_tif, load_shine, denoise_stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir')
    ap.add_argument('den_dir')
    ap.add_argument('model_dir')
    ap.add_argument('--model', default='3x3_blind')
    ap.add_argument('--frame_num', type=int, default=5)
    ap.add_argument('--frames', type=int, nargs='+', default=None)
    a = ap.parse_args()
    names = list_tifs(a.raw_dir)
    raw, _ = read_stack(a.raw_dir, names)
    frames = a.frames or [0, len(names) // 2]
    refs = [read_tif(os.path.join(a.den_dir, names[i])) for i in frames]
    best = None
    for ck in sorted(glob.glob(os.path.join(a.model_dir, '*.ckpt'))):
        net = load_shine(ck, a.model, frame_num=a.frame_num)
        out = denoise_stack(net, raw, frame_num=a.frame_num, idx=frames)
        med = float(np.median([np.median(np.abs(o - r)) for o, r in zip(out, refs)]))
        print(f'{os.path.basename(ck)}: median |diff| = {med:.3f}')
        if best is None or med < best[1]:
            best = (ck, med)
    print(f'producer: {best[0]}' + ('' if best[1] < 1 else '  (WARNING: no exact match)'))


if __name__ == '__main__':
    main()
