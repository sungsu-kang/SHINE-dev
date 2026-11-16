"""Figures for moving_atom.py: summary (drift split, size, hop timing) and example frames.

Usage: python moving_atom_figure.py MOVING_ATOM_DIR SUMMARY.png EXAMPLES.png [--title TEXT]
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'production': '#2a78d6', 'p 0.5': '#eb6834', 'p 0.75': '#6250d6', '1-frame': '#d6336c'}
NAMES = {'production': '5-frame, production', 'p 0.5': '5-frame, p = 0.5', 'p 0.75': '5-frame, p = 0.75',
         '1-frame': '1-frame'}
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})


def headline(ax, title, sub=None):
    ax.set_title(title, loc='left', fontsize=14, fontweight='bold', color=INK, pad=24 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color=INK2, va='bottom')


def grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def summary_fig(S, out, title):
    meta, M = S['meta'], S['models']
    labs = list(M)
    n = len(labs)
    v = meta['speed']
    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.3)

    # 1. where the moving atom is shown
    ax = fig.add_subplot(gs[0, 0:2])
    K = (-2, -1, 0, 1, 2)
    w = 0.8 / n
    for i, l in enumerate(labs):
        y = [M[l]['drift'][f'w{k:+d}'] for k in K]
        bars = ax.bar(np.arange(5) + (i - (n - 1) / 2) * w, y, w * 0.92, color=COLORS[l], label=NAMES[l])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0), f'{b.get_height():.2f}',
                    ha='center', va='bottom', fontsize=9, color=INK)
    ax.set_xticks(range(5), [f'{k * v:+g} px\n(position at t{k:+d})' if k else 'true position\nnow (ideal 1)'
                             for k in K])
    ax.set_ylabel('fraction of the atom shown there')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', ncol=2, fontsize=10.5)
    grid(ax)
    headline(ax, f'Atom drifting {v:g} px/frame: where the output frame shows it',
             'least-squares split of the output change onto the atom\'s positions in nearby frames; '
             'other positions ideal 0')

    # 2. size / shape
    ax = fig.add_subplot(gs[0, 2])
    cats = ['stationary atom:\namount kept', f'moving atom:\ntotal shown', 'moving atom:\nlength / width']
    for i, l in enumerate(labs):
        y = [M[l]['static']['amp'], M[l]['drift']['total'], M[l]['drift']['elong']]
        bars = ax.bar(np.arange(3) + (i - (n - 1) / 2) * w, y, w * 0.92, color=COLORS[l])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{b.get_height():.2f}',
                    ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.axhline(1, color=INK, ls='--', lw=1)
    ax.set_xticks(range(3), cats)
    ax.set_ylabel('ideal 1 for all three')
    grid(ax)
    headline(ax, 'Amount and shape', 'stationary baseline vs moving')

    # 3-4. hop and stay
    for j, D in enumerate(meta['hops']):
        ax = fig.add_subplot(gs[1, j])
        ms = sorted(int(m) for m in M[labs[0]]['hop'][str(D)])
        ax.step(ms, [0 if m < 0 else 1 for m in ms], where='mid', color=INK, ls='--', lw=1.2, label='ideal')
        for l in labs:
            h = M[l]['hop'][str(D)]
            ax.plot(ms, [h[str(m)]['B'] for m in ms], 'o-', color=COLORS[l], lw=2, ms=6, label=NAMES[l])
        ax.set_xticks(ms, [f't0{m:+d}' if m else 't0' for m in ms])
        ax.set_xlabel('output frame (atom at new site from raw frame t0 on)')
        ax.set_ylabel('fraction shown at the new site')
        ax.set_ylim(-0.05, 1.15)
        if j == 0:
            ax.legend(loc='lower right', fontsize=10)
        grid(ax)
        headline(ax, f'Atom hops {D} px and stays', 'ideal: a step at t0')

    # 5. hop: old site
    ax = fig.add_subplot(gs[1, 2])
    D = meta['hops'][0]
    for l in labs:
        h = M[l]['hop'][str(D)]
        ax.plot(ms, [h[str(m)]['A'] for m in ms], 'o-', color=COLORS[l], lw=2, ms=6, label=NAMES[l])
    ax.step(ms, [1 if m < 0 else 0 for m in ms], where='mid', color=INK, ls='--', lw=1.2)
    ax.set_xticks(ms, [f't0{m:+d}' if m else 't0' for m in ms])
    ax.set_xlabel('output frame')
    ax.set_ylabel('fraction still shown at the old site')
    ax.set_ylim(-0.05, 1.15)
    grid(ax)
    headline(ax, f'Hop {D} px: ghost at the old site', 'ideal: drops to 0 at t0')

    fig.suptitle(title or '', x=0.12, y=0.99, ha='left', fontsize=13, color=INK2)
    fig.savefig(out, dpi=95, facecolor='white', bbox_inches='tight')


def examples_fig(S, ex, out, cond, title, half=48, scale=0.5):
    meta = S['meta']
    labs = list(ex)
    ms = sorted(ex[labs[0]][cond])
    s, v = meta['sigma'], meta['speed']
    lim = scale * meta['A']
    c0 = ex[labs[0]][cond][ms[0]]['raw'].shape[0] // 2
    cr = (slice(c0 - half, c0 + half + 1),) * 2
    c = half
    rows = len(labs) + 2
    fig, axs = plt.subplots(rows, len(ms), figsize=(2.3 * len(ms), 2.4 * rows),
                            gridspec_kw=dict(hspace=0.1, wspace=0.04))
    raw = [ex[labs[0]][cond][m]['raw'][cr] for m in ms]
    rlo, rhi = np.percentile(np.concatenate([r.ravel() for r in raw]), [1, 99])

    def truth(m):
        if cond.startswith('drift'):
            return (c, c + v * m) if cond == 'drift_x' else (c + v * m, c)
        D = int(cond.split('_')[1])
        return (c, c - D / 2) if m < 0 else (c, c + D / 2)

    yy, xx = np.indices(raw[0].shape)
    for j, m in enumerate(ms):
        y, x = truth(m)
        axs[0, j].imshow(raw[j], cmap='gray', vmin=rlo, vmax=rhi)
        axs[0, j].set_title(f't0{m:+d}' if m else 't0', fontsize=12, color=INK2)
        ideal = meta['A'] * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * s * s))
        axs[1, j].imshow(ideal, cmap='RdBu_r', vmin=-lim, vmax=lim)
        for i, l in enumerate(labs):
            axs[i + 2, j].imshow(ex[l][cond][m]['d'][cr], cmap='RdBu_r', vmin=-lim, vmax=lim)
        for i in range(rows):
            ax = axs[i, j]
            ax.add_patch(Circle((x, y), 2 * s, fill=False, ec='#1baf7a', lw=1.5, ls='--'))
            ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].set_ylabel('raw input\n(atom added)', fontsize=11)
    axs[1, 0].set_ylabel('ideal output\nchange', fontsize=11)
    for i, l in enumerate(labs):
        axs[i + 2, 0].set_ylabel(f'{NAMES[l]}\noutput change', fontsize=11)
    what = (f'atom drifting {v:g} px/frame' if cond.startswith('drift') else
            f'atom hopping {cond.split("_")[1]} px at t0 and staying')
    fig.text(0.12, 0.905, f'{title}: {what}. Green circle = true position in that frame (radius 2 sigma).\n'
             f'Output change = with atom - without; colour scale +-{scale:g} x atom peak (red = brighter).',
             ha='left', va='bottom', fontsize=12, color=INK2)
    fig.savefig(out, dpi=90, facecolor='white', bbox_inches='tight')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('summary_png'); ap.add_argument('examples_png')
    ap.add_argument('--title', default='Insitu(8)')
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.dir, 'moving_atom_summary.json')))
    ex = np.load(os.path.join(a.dir, 'examples.npy'), allow_pickle=True).item()
    summary_fig(S, a.summary_png, f'{a.title}: faint atom (sigma {S["meta"]["sigma"]:g} px, '
                f'4x detection threshold) present in every frame and moving')
    examples_fig(S, ex, a.examples_png, 'drift_x', a.title)
    D = S['meta']['hops'][-1]
    examples_fig(S, ex, a.examples_png.replace('.png', f'_hop{D}.png'), f'hop_{D}', a.title)


if __name__ == '__main__':
    main()
