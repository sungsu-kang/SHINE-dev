"""Compare SHINE variants on one dataset: accuracy (even/odd) vs timing (one-frame edits).

Each variant directory has evenodd/analysis/, null/ and temporal/ as written by the Tier 2/3
scripts (the production run's OUT dir, or a train_variant.sh + variant_tests.sh dir).

Usage: python variant_compare.py OUT.png --variant "LABEL=DIR" [--variant ...] [--title TEXT]
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

INK, INK2, GRID, CREF = '#0b0b0b', '#52514e', '#e4e3df', '#9c9b96'
COLORS = ['#2a78d6', '#1baf7a', '#eb6834', '#6250d6', '#eda100', '#d6336c']
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


def bar_labels(ax, bars, fmt='{:.2f}', size=9.5):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                ha='center', va='bottom', fontsize=size, color=INK)


def grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--variant', action='append', required=True, help='LABEL=DIR')
    ap.add_argument('--title', default=None)
    a = ap.parse_args()
    V = []
    for i, spec in enumerate(a.variant):
        lab, d = spec.rsplit('=', 1)
        V.append(dict(label=lab, color=COLORS[i], ev=jload(d, 'evenodd', 'analysis', 'evenodd_summary.json'),
                      eo=np.load(os.path.join(d, 'evenodd', 'analysis', 'evenodd_data.npz')),
                      tp=jload(d, 'temporal', 'temporal_summary.json')))
    n = len(V)

    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.3)

    # 1. held-out error
    ax = fig.add_subplot(gs[0, 0])
    labs = ['Gaussian\nblur 3 px'] + [v['label'].replace(', ', ',\n').replace(' p=', '\np=') for v in V]
    vals = [V[0]['ev']['held_out_error_over_noise_var']['Gaussian blur 3 px']] + \
           [v['ev']['held_out_error_over_noise_var']['SHINE (E model)'] for v in V]
    bars = ax.bar(range(n + 1), vals, 0.62, color=[CREF] + [v['color'] for v in V])
    bar_labels(ax, bars, '{:.4f}')
    ax.set_xticks(range(n + 1), labs, fontsize=9.5)
    ax.set_ylabel('held-out error / noise variance\n(lower is better)')
    grid(ax)
    headline(ax, 'Accuracy vs an independent frame', 'even/odd test')

    # 2. relative spectral error
    ax = fig.add_subplot(gs[0, 1])
    for v in V:
        d = v['eo']
        ok = d['S_k'] > 0.02 * d['N_k']
        ax.plot(d['fk'], np.where(ok, d['E_k_SHINE'] / np.where(ok, d['S_k'], 1), np.nan), color=v['color'],
                lw=2, label=v['label'])
    ax.axhline(1, color=INK, ls='--', lw=1.2)
    ax.text(0.003, 1.03, 'error = signal', fontsize=10, color=INK, va='bottom')
    ax.set_xlabel('spatial frequency (cycles/px)')
    ax.set_ylabel('E(k) / S(k)')
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0, 1.6)
    ax.legend(loc='lower right', fontsize=10)
    grid(ax)
    headline(ax, 'Error at each length scale', 'below 1 = more signal than error')

    # 3. robustness to wrong neighbours
    ax = fig.add_subplot(gs[0, 2])
    names = ['real neighbours', 'far frames as neighbours', 'noise-only neighbours']
    w = 0.8 / n
    for i, v in enumerate(V):
        nb = v['tp'].get('neighbours')
        if not nb:
            continue
        bars = ax.bar(np.arange(3) + (i - (n - 1) / 2) * w, [nb[k]['held_out'] for k in names], w * 0.92,
                      color=v['color'], label=v['label'])
        bar_labels(ax, bars[1:], '{:.3f}', 9)       # real-neighbour bars are all ~equal: label once below
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.text(0, ax.get_ylim()[1] * 0.18, f"{V[0]['tp']['neighbours']['real neighbours']['held_out']:.3f}", ha='center', fontsize=9, color=INK)
    ax.set_xticks(range(3), ['real\nneighbours', 'neighbours from\nfar-away frames', 'noise-only\nneighbours'])
    ax.set_ylabel('held-out error / noise variance')
    ax.legend(loc='upper left', fontsize=10)
    grid(ax)
    headline(ax, 'When the neighbours are wrong', 'output of frame t with replaced neighbours')

    # 4. one-frame events at t0
    ax = fig.add_subplot(gs[1, 0:2])
    cats = ['particle removed\n(ideal 0)', 'jump 2-20 px: old\nposition (ideal 0)', 'jump 2-20 px: new\nposition (ideal 1)',
            'blob added, 4x thr.\n(ideal 1)', 'removal seen in\noutputs t0±1, ±2 (ideal 1)']
    for i, v in enumerate(V):
        s = v['tp']
        jm = lambda f: float(np.mean([q[f] for q in s['jump_t0'].values()]))
        key = max((q for q in s['insert'] if q.startswith('sigma5')), key=lambda q: float(q.split('_x')[1]))
        nbr = float(np.mean([s['remove'][str(o)] for o in (-2, -1, 1, 2)]))
        y = [s['remove']['0'], jm('old'), jm('new'), s['insert'][key]['0'], nbr]
        bars = ax.bar(np.arange(5) + (i - (n - 1) / 2) * w, y, w * 0.92, color=v['color'], label=v['label'])
        bar_labels(ax, bars)
    ax.set_xticks(range(5), cats)
    ax.set_ylabel('shown in output (fraction)')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper center', ncol=n, fontsize=10.5)
    grid(ax)
    headline(ax, 'Change made in raw frame t0 only', 'events the models were not trained on; bars 1-4: output of frame t0')

    # 5. jump vs distance
    ax = fig.add_subplot(gs[1, 2])
    for v in V:
        jt = v['tp']['jump_t0']
        ds = sorted({int(k[1:]) for k in jt})
        new = [np.mean([jt[f'x{d}']['new'], jt[f'y{d}']['new']]) for d in ds]
        ax.plot(ds, new, 'o-', color=v['color'], lw=2, ms=6, label=v['label'])
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.set_xticks(ds)
    ax.set_xlabel('jump distance in raw frame t0 (px)')
    ax.set_ylabel('shown at new position in output t0')
    ax.set_ylim(0, 1.5)
    ax.axhline(1, color=INK, ls='--', lw=1)
    ax.legend(loc='upper left', fontsize=10)
    grid(ax)
    headline(ax, 'Jump vs distance', 'ideal 1 at every distance')

    if a.title:
        fig.suptitle(a.title, x=0.12, y=0.99, ha='left', fontsize=13, color=INK2)
    fig.savefig(a.out, dpi=95, facecolor='white', bbox_inches='tight')


if __name__ == '__main__':
    main()
