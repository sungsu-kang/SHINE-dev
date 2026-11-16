"""Example frames for the Tier 3 one-frame edits: raw input, SHINE output, and output change,
for five consecutive frames t0-2 ... t0+2, with the edit applied to raw frame t0 only.

Usage: python temporal_examples.py RAW_DIR DEN_DIR CKPT OUT.png --null_summary J
           [--t0 104] [--part 1] [--frame_num 5]
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from temporal_test import (Stack, Model, OFFSETS, background_sites, gauss, particle_sites,
                           particle_profile, place)

INK2 = '#52514e'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir'); ap.add_argument('den_dir'); ap.add_argument('ckpt'); ap.add_argument('out')
    ap.add_argument('--null_summary', required=True)
    ap.add_argument('--t0', type=int, default=104)
    ap.add_argument('--part', type=int, default=1, help='index into the particles found in frame t0')
    ap.add_argument('--jump', type=int, default=10)
    ap.add_argument('--blob_mult', type=float, default=4, help='blob amplitude, x detection threshold')
    ap.add_argument('--frame_num', type=int, default=5)
    ap.add_argument('--label', default='production 5-frame model')
    a = ap.parse_args()
    floor = json.load(open(a.null_summary))['null_floor_score']
    st = Stack(a.raw_dir, a.den_dir)
    model = Model(a.ckpt, st, frame_num=a.frame_num)
    cache = {}
    t0 = a.t0
    y, x, r = particle_sites(st, t0, cache)[a.part]
    p, sl = particle_profile(st, t0, y, x, r, cache)
    shape = (st.H, st.W)

    by, bx = background_sites(st)[0]
    s = 5
    A = 2 * floor * a.blob_mult          # score ~ amplitude / 2
    events = [
        (f'Particle removed from raw frame t0', -place(shape, p, sl), (y, x)),
        (f'Particle moved {a.jump} px to the right in raw frame t0 only',
         place(shape, p, sl, 0, a.jump) - place(shape, p, sl), (y, x + a.jump // 2)),
        (f'Faint blob ({a.blob_mult:g}x detection threshold) added to raw frame t0',
         (A * gauss(shape, by, bx, s)).astype(np.float32), (by, bx)),
    ]
    h = int(max(sl[0].stop - sl[0].start, 80) // 2 + 6)

    fig = plt.figure(figsize=(16, 28))
    subs = fig.subfigures(3, 1, hspace=0.04)
    for e, (title, edit, (cy, cx)) in enumerate(events):
        sf = subs[e]
        axs = sf.subplots(3, 6, gridspec_kw=dict(hspace=0.08, wspace=0.05))
        sf.suptitle(f'{"ABC"[e]}. {title}', x=0.02, ha='left', fontsize=16, fontweight='bold')
        c = (slice(cy - h, cy + h), slice(cx - h, cx + h))
        raw = [st.raw[t0 + k] + (edit if k == 0 else 0) for k in OFFSETS]
        out_u = [model.unedited(t0 + k) for k in OFFSETS]
        out_e = [model.edited(t0 + k, {t0: edit}) for k in OFFSETS]
        ref = np.concatenate([o[c].ravel() for o in out_u])
        lo, hi = np.percentile(ref, [0.5, 99.8])
        rref = np.concatenate([r_[c].ravel() for r_ in raw])
        rlo, rhi = np.percentile(rref, [1, 99])
        lim = np.abs(edit[c]).max()
        rows = [('raw input', raw, 'gray', rlo, rhi), ('SHINE output', out_e, 'gray', lo, hi),
                ('output change\n(edited - unedited)', [oe - ou for oe, ou in zip(out_e, out_u)],
                 'RdBu_r', -lim, lim)]
        for i, (name, imgs, cmap, vmin, vmax) in enumerate(rows):
            for j, k in enumerate(OFFSETS):
                ax = axs[i, j]
                ax.imshow(imgs[j][c], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(k == 0); sp.set_color('#eb6834'); sp.set_linewidth(3)
                if i == 0:
                    ax.set_title(f't0{k:+d}' if k else 't0 (edited)', fontsize=12,
                                 color='#eb6834' if k == 0 else INK2)
                if j == 0:
                    ax.set_ylabel(name, fontsize=12)
        # last column: the true change, and the ideal output change (the edit itself)
        ax = axs[0, 5]
        ax.imshow(edit[c], cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title('edit added to\nraw t0', fontsize=11, color=INK2)
        ax = axs[2, 5]
        ax.imshow(edit[c], cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title('ideal change\nat t0', fontsize=11, color=INK2)
        for i in range(3):
            ax = axs[i, 5]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axs[1, 5].axis('off')
    fig.text(0.02, 1.0, f'Edits made in one raw frame only (t0 = frame {t0}, {a.label}); '
             'neighbouring raw frames untouched.\nOutput change: red = brighter, blue = darker than without the edit, '
             'on the same colour scale as the edit itself.', fontsize=13, color=INK2, va='bottom')
    fig.savefig(a.out, dpi=80, facecolor='white', bbox_inches='tight')


if __name__ == '__main__':
    main()
