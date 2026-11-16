"""Image strips for the moving-atom test: raw input, clean reference, and each method's output.

Clean reference = mean of all raw frames (flat background, noise averaged out) + the inserted atom
at its true position in that frame. Outputs are on the raw intensity scale (per-frame affine map,
as in temporal_test.Model). Methods: Gaussian blur (3 px, the even/odd baseline) and the SHINE models.

Usage: python moving_atom_images.py RAW DEN OUT.png --null_summary J --cond drift_x|hop_20
           --model_spec "LABEL=CKPT[:FRAME_NUM]" ... [--t0 T] [--site 0]
"""
import argparse
import json
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from temporal_test import Stack, Model, background_sites, frames_t0, gauss
from moving_atom import edits_for, trajectory, HALF_MEASURE

INK2 = '#52514e'
NAMES = {'production': '5-frame SHINE,\nproduction', 'p 0.5': '5-frame SHINE,\np = 0.5',
         'p 0.75': '5-frame SHINE,\np = 0.75', '1-frame': '1-frame SHINE'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir'); ap.add_argument('den_dir'); ap.add_argument('out')
    ap.add_argument('--null_summary', required=True)
    ap.add_argument('--model_spec', action='append', required=True)
    ap.add_argument('--cond', default='drift_x')
    ap.add_argument('--speed', type=float, default=8)
    ap.add_argument('--sigma', type=float, default=5)
    ap.add_argument('--blob_mult', type=float, default=4)
    ap.add_argument('--t0', type=int, default=None)
    ap.add_argument('--site', type=int, default=0)
    ap.add_argument('--half', type=int, default=48)
    a = ap.parse_args()
    floor = json.load(open(a.null_summary))['null_floor_score']
    A = 2 * floor * a.blob_mult
    st = Stack(a.raw_dir, a.den_dir)
    sites = background_sites(st)
    t0s = [int(t) for t in frames_t0(st, 10)]
    t0 = a.t0 if a.t0 is not None else t0s[len(t0s) // 2]
    y0, x0 = sites[a.site]
    shape = (st.H, st.W)
    ed = edits_for(a.cond, t0, sites, A, a.sigma, shape, a.speed)
    ms = list(range(-HALF_MEASURE, HALF_MEASURE + 1))
    h = a.half
    cr = (slice(y0 - h, y0 + h + 1), slice(x0 - h, x0 + h + 1))
    bg = st.raw.mean(0)

    rows = [('raw input\n(atom added)', [(st.raw[t0 + m] + ed[t0 + m])[cr] for m in ms]),
            ('clean reference\n(mean bg + atom)', [(bg + ed[t0 + m])[cr] for m in ms]),
            ('Gaussian blur\n3 px', [cv.GaussianBlur((st.raw[t0 + m] + ed[t0 + m]).astype(np.float32),
                                                       (0, 0), 3)[cr] for m in ms])]
    for spec in a.model_spec:
        lab, ck = spec.split('=', 1)
        ck, fn = (ck.rsplit(':', 1) if ck.rsplit(':', 1)[-1].isdigit() else (ck, '5'))
        model = Model(ck, st, frame_num=int(fn))
        rows.append((NAMES.get(lab, lab), [model.edited(t0 + m, ed)[cr] for m in ms]))
        print(lab, 'done', flush=True)
        del model

    clean = np.concatenate([r.ravel() for r in rows[1][1]])
    lo, hi = np.percentile(clean, 0.5), clean.max()
    pad = 0.25 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    rlo, rhi = np.percentile(np.concatenate([r.ravel() for r in rows[0][1]]), [1, 99])
    n = len(rows)
    fig, axs = plt.subplots(n, len(ms), figsize=(2.2 * len(ms), 2.3 * n), gridspec_kw=dict(hspace=0.08, wspace=0.04))
    for i, (name, imgs) in enumerate(rows):
        for j, m in enumerate(ms):
            ax = axs[i, j]
            ax.imshow(imgs[j], cmap='gray', vmin=rlo if i == 0 else lo, vmax=rhi if i == 0 else hi)
            cy, cx = trajectory(a.cond, t0 + m, t0, y0, x0, a.speed)
            ax.add_patch(Circle((cx - x0 + h, cy - y0 + h), 2 * a.sigma, fill=False, ec='#1baf7a', lw=1.2, ls='--'))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f't0{m:+d}' if m else 't0', fontsize=12, color=INK2)
            if j == 0:
                ax.set_ylabel(name, fontsize=11)
    what = (f'faint atom drifting {a.speed:g} px/frame' if a.cond.startswith('drift') else
            f'faint atom hopping {a.cond.split("_")[1]} px at t0 and staying')
    fig.text(0.12, 0.9, f'Insitu(8), frame t0 = {t0}: {what} (sigma {a.sigma:g} px, {a.blob_mult:g}x detection '
             f'threshold), flat background.\nGreen circle = true position. All rows except raw share one grey scale.',
             ha='left', va='bottom', fontsize=12, color=INK2)
    fig.savefig(a.out, dpi=90, facecolor='white', bbox_inches='tight')


if __name__ == '__main__':
    main()
