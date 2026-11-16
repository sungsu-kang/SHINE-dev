"""5-frame (production) vs 1-frame SHINE on one dataset: images, Tier 2 accuracy, Tier 3 timing.

Reads the outputs of the Tier 2/3 scripts for both models; needs no GPU.

Usage: python single_frame_compare.py OUT.png --raw RAW_DIR \
           --prod_den DIR --prod OUT_DIR       # residual/, null/, evenodd/analysis/, temporal/
           --sf_den DIR --sf SF_DIR            # same layout (single_frame_tests.sh)
           [--frame T] [--crop y0,y1,x0,x1]
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from common import list_tifs, read_tif, affine_fit

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C5, C1, CREF = '#2a78d6', '#eb6834', '#9c9b96'          # 5-frame, 1-frame, references
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})


def jload(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


def headline(ax, title, sub=None):
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold', color=INK, pad=24 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color=INK2, va='bottom')


def bar_labels(ax, bars, fmt='{:.2f}'):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                ha='center', va='bottom', fontsize=10, color=INK)


def grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def images(axs, a, names):
    t = a.frame if a.frame is not None else len(names) // 2
    raw = read_tif(os.path.join(a.raw, names[t]))
    outs = [read_tif(os.path.join(d, names[t])) for d in (a.prod_den, a.sf_den)]
    m = 40
    outs = [np.polyval(affine_fit(o[m:-m, m:-m], raw[m:-m, m:-m]), o) for o in outs]
    y0, y1, x0, x1 = a.crop
    lo, hi = np.percentile(outs[0][y0:y1, x0:x1], [0.5, 99.5])
    for ax, img, title in zip(axs, [raw] + outs, [f'raw frame {t}', '5-frame SHINE (production)',
                                                  '1-frame SHINE (no temporal input)']):
        ax.imshow(img[y0:y1, x0:x1], cmap='gray', vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=13, color=INK)
        ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    for k in ('raw', 'prod_den', 'prod', 'sf_den', 'sf'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--crop', default='300,556,300,556')
    a = ap.parse_args()
    a.crop = [int(v) for v in a.crop.split(',')]
    names = list_tifs(a.raw)

    ev = {k: jload(d, 'evenodd', 'analysis', 'evenodd_summary.json') for k, d in (('5', a.prod), ('1', a.sf))}
    eo = {k: np.load(os.path.join(d, 'evenodd', 'analysis', 'evenodd_data.npz')) for k, d in (('5', a.prod), ('1', a.sf))}
    nu = {k: jload(d, 'null', 'null_summary.json') for k, d in (('5', a.prod), ('1', a.sf))}
    tp = {k: jload(d, 'temporal', 'temporal_summary.json') for k, d in (('5', a.prod), ('1', a.sf))}

    fig = plt.figure(figsize=(19, 15.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1, 1], hspace=0.45, wspace=0.3)
    images([fig.add_subplot(gs[0, j]) for j in range(3)], a, names)

    # held-out error (lower = closer to the independent raw frame)
    ax = fig.add_subplot(gs[1, 0])
    h5 = ev['5']['held_out_error_over_noise_var']
    lab = ['best Gaussian\nblur (3 px)', '1-frame\nSHINE', '5-frame\nSHINE']
    val = [h5['Gaussian blur 3 px'], ev['1']['held_out_error_over_noise_var']['SHINE (E model)'],
           h5['SHINE (E model)']]
    bars = ax.bar(lab, val, 0.6, color=[CREF, C1, C5])
    bar_labels(ax, bars, '{:.4f}')
    ax.set_ylabel('held-out error / noise variance\n(lower is better)')
    grid(ax)
    headline(ax, 'Accuracy vs an independent frame', 'even/odd test; upper bound (includes real change)')

    # relative spectral error E(k)/S(k)
    ax = fig.add_subplot(gs[1, 1])
    for k, c, name in (('5', C5, '5-frame'), ('1', C1, '1-frame')):
        d = eo[k]
        ok = d['S_k'] > 0.02 * d['N_k']
        r = np.where(ok, d['E_k_SHINE'] / np.where(ok, d['S_k'], 1), np.nan)
        ax.plot(d['fk'], r, color=c, lw=2, label=name)
    ax.axhline(1, color=INK, ls='--', lw=1.2)
    ax.text(0.005, 1.03, 'error = signal', fontsize=10, color=INK, va='bottom')
    ax.set_xlabel('spatial frequency (cycles/px)')
    ax.set_ylabel('error power / signal power  E(k)/S(k)')
    ax.set_xlim(0, ev['5']['signal_measurable_below'] * 1.6)
    ax.set_ylim(0, 2)
    ax.legend(loc='upper left')
    grid(ax)
    headline(ax, 'Error at each length scale', 'below 1 = more signal than error')

    # blob reproducibility vs score (own null floor)
    ax = fig.add_subplot(gs[1, 2])
    for k, c, name in (('5', C5, '5-frame'), ('1', C1, '1-frame')):
        rc = np.array(ev[k]['reproduced_fraction_by_score'], float)
        ax.plot(rc[:, 0] / ev[k]['floor'], rc[:, 1], 'o-', color=c, lw=2, ms=5,
                label=f'{name} ({ev[k]["detections_above_floor"]} blobs, {100 * ev[k]["reproduced_fraction"]:.0f}%)')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel('blob score / own null floor')
    ax.set_ylabel('fraction found by the other half-model')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    grid(ax)
    headline(ax, 'Blobs reproduced by an independent model')

    # one-frame events at t0
    ax = fig.add_subplot(gs[2, 0:2])
    cats = ['blob added, 4x thr.\n(ideal 1)', 'particle removed\n(ideal 0)', 'jump 2-20 px: old\nposition (ideal 0)',
            'jump 2-20 px: new\nposition (ideal 1)', 'blob leak into\nneighbours (ideal 0)']
    w = 0.36
    for i, (k, c, name) in enumerate((('5', C5, '5-frame SHINE'), ('1', C1, '1-frame SHINE'))):
        s = tp[k]
        key = max((q for q in s['insert'] if q.startswith('sigma5')), key=lambda q: float(q.split('_x')[1]))
        jm = lambda f: float(np.mean([v[f] for v in s['jump_t0'].values()]))
        leak = float(np.mean([s['insert'][key][str(o)] for o in (-2, -1, 1, 2)]))
        y = [s['insert'][key]['0'], s['remove']['0'], jm('old'), jm('new'), leak]
        bars = ax.bar(np.arange(5) + (i - 0.5) * w, y, w * 0.92, color=c, label=name)
        bar_labels(ax, bars)
    ax.set_xticks(range(5), cats)
    ax.set_ylabel('shown in output (fraction of the change)')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper center', ncol=2)
    grid(ax)
    headline(ax, 'Change made in raw frame t0 only', 'bars 1-4: output of frame t0; bar 5: mean over outputs t0±1, t0±2')

    # one-frame blob: response in its own output frame vs inserted amplitude
    ax = fig.add_subplot(gs[2, 2])
    for k, c, name in (('5', C5, '5-frame'), ('1', C1, '1-frame')):
        for sig, ls, mk in (('5', '-', 'o'), ('2', '--', 's')):
            keys = sorted((q for q in tp[k]['insert'] if q.startswith(f'sigma{sig}_')), key=lambda q: float(q.split('_x')[1]))
            m = [float(q.split('_x')[1]) for q in keys]     # A = mult x (2 x score floor) = mult x threshold amplitude
            ax.plot(m, [tp[k]['insert'][q]['0'] for q in keys], ls=ls, marker=mk, color=c, lw=2, ms=6,
                    label=f'{name}, blob sigma {sig} px')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xlabel('inserted amplitude (x 5-frame detection threshold)')
    ax.set_ylabel('kept in output frame t0 (ideal 1)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', fontsize=9.5)
    grid(ax)
    headline(ax, 'One-frame blob vs its strength', '1-frame keeps strong blobs, suppresses faint ones')

    fig.savefig(a.out, dpi=100, facecolor='white', bbox_inches='tight')


if __name__ == '__main__':
    main()
