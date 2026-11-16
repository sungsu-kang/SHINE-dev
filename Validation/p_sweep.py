"""Neighbour-corruption probability p: leakage vs accuracy trade-off on one dataset.

Each p has a variant dir laid out as in variant_compare.py; the 1-frame model is drawn as a
reference line (the no-leakage limit).

Usage: python p_sweep.py OUT.png --point "P=DIR" [--point ...] --single_frame DIR [--title TEXT]
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, GRID, C1 = '#0b0b0b', '#52514e', '#e4e3df', '#eb6834'
CA, CB, CC = '#2a78d6', '#1baf7a', '#6250d6'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})


def jload(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


def metrics(d):
    ev = jload(d, 'evenodd', 'analysis', 'evenodd_summary.json')
    tp = jload(d, 'temporal', 'temporal_summary.json')
    nu = jload(d, 'null', 'null_summary.json')
    jm = lambda f: float(np.mean([q[f] for q in tp['jump_t0'].values()]))
    key = max((q for q in tp['insert'] if q.startswith('sigma5')), key=lambda q: float(q.split('_x')[1]))
    return dict(held=ev['held_out_error_over_noise_var']['SHINE (E model)'], remove=tp['remove']['0'],
                old=jm('old'), new=jm('new'), blob=tp['insert'][key]['0'], floor=nu['null_floor_score'])


def headline(ax, title, sub=None):
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold', color=INK, pad=24 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color=INK2, va='bottom')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--point', action='append', required=True, help='P=DIR')
    ap.add_argument('--single_frame', required=True)
    ap.add_argument('--title', default=None)
    a = ap.parse_args()
    pts = sorted((float(s.split('=', 1)[0]), metrics(s.split('=', 1)[1])) for s in a.point)
    P = [p for p, _ in pts]
    M = {k: [m[k] for _, m in pts] for k in pts[0][1]}
    sf = metrics(a.single_frame)

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(wspace=0.3))

    def ref(ax, v, txt, va='bottom'):
        ax.axhline(v, color=C1, ls=':', lw=1.6)
        ax.text(P[-1], v, txt, color=C1, fontsize=10, ha='right', va=va)

    def setx(ax):
        ax.set_xticks(P, [f'{p:g}' for p in P])
        ax.set_xlabel('neighbour-corruption probability p  (0 = production)')
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # 1. leakage: what the output at t0 shows after a one-frame edit
    ax = axs[0]
    for k, c, lab in (('remove', CA, 'particle removed: still shown (ideal 0)'),
                      ('old', CB, 'jump: old position (ideal 0)'),
                      ('new', CC, 'jump: new position (ideal 1)')):
        ax.plot(P, M[k], 'o-', color=c, lw=2, ms=7, label=lab)
        ax.plot(P[-1] + 0.06, sf[k], 'D', color=c, ms=7, mfc='white', mew=2)
    ax.text(P[-1] + 0.06, 1.05, '1-frame', color=C1, fontsize=10, ha='center')
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.05, P[-1] + 0.12)
    ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(0.88, 1.0))
    setx(ax)
    headline(ax, 'Temporal leakage', 'one-frame edits in raw t0; output of frame t0')

    # 2. accuracy
    ax = axs[1]
    ax.plot(P, M['held'], 'o-', color=CA, lw=2, ms=7)
    for p, v in zip(P, M['held']):
        ax.text(p, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9.5, color=INK, transform=ax.transData)
    ref(ax, sf['held'], f"1-frame {sf['held']:.4f}", 'top')
    ax.set_ylim(0.0075, 0.011)
    ax.set_ylabel('held-out error / noise variance\n(lower is better)')
    setx(ax)
    headline(ax, 'Accuracy vs an independent frame', 'even/odd test')

    # 3. false-detection threshold and faint blobs
    ax = axs[2]
    ax.plot(P, M['floor'], 'o-', color=CA, lw=2, ms=7)
    for p, v in zip(P, M['floor']):
        ax.text(p, v + 4, f'{v:.0f}', ha='center', va='bottom', fontsize=9.5, color=INK)
    ref(ax, sf['floor'], f"1-frame {sf['floor']:.0f}", 'top')
    ax.set_ylim(150, 345)
    ax.set_ylabel('null floor (blob score)\n(lower = finer features detectable)')
    setx(ax)
    headline(ax, 'False-detection threshold', 'score of the strongest blob made from pure noise')

    if a.title:
        fig.suptitle(a.title, x=0.12, y=1.06, ha='left', fontsize=13, color=INK2)
    fig.savefig(a.out, dpi=95, facecolor='white', bbox_inches='tight')


if __name__ == '__main__':
    main()
