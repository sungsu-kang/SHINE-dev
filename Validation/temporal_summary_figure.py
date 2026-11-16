"""One-page summary of the Tier 3 (temporal) results: one panel per finding.

Reads only the summary JSONs written by temporal_test.py, so it needs no GPU.

Usage: python temporal_summary_figure.py OUT.png \
           --sweep LABEL=DIR/temporal [LABEL=DIR/temporal ...] \
           [--control PROD_DIR RETRAINED_DIR]     # new_events_prod / new_events_retrained of one dataset
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834']             # dataset 1, dataset 2
PROD, RETR = '#7a7974', '#1baf7a'           # production / retrained model
K = [-2, -1, 0, 1, 2]

plt.rcParams.update({'font.size': 12, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})


def load(d):
    with open(os.path.join(d, 'temporal_summary.json')) as fh:
        return json.load(fh)


def jump_mean(s, key):
    return float(np.mean([v[key] for v in s['jump_t0'].values()]))


def headline(ax, n, title, sub):
    ax.set_title(f'{n}. {title}', loc='left', fontsize=15, fontweight='bold', color=INK, pad=30)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=11, color=INK2, va='bottom')


def bar_labels(ax, bars, fmt='{:.2f}'):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h), ha='center', va='bottom',
                fontsize=10, color=INK)


def panel_ghosts(ax, sweeps):
    # strongest insertion (largest mult), largest blob; amplitude in units of the detection floor
    w = 0.38
    for i, (lab, s) in enumerate(sweeps):
        key = max((k for k in s['insert'] if k.startswith('sigma5')), key=lambda k: float(k.split('_x')[1]))
        mult = float(key.split('_x')[1])
        y = [s['insert'][key][str(k)] * mult for k in K]
        x = np.arange(5) + (i - 0.5) * w
        bars = ax.bar(x, y, w * 0.92, color=SERIES[i], label=lab)
        bar_labels(ax, bars)
    ax.bar(2, mult, 0.92, fill=False, edgecolor=INK2, ls='--', lw=1.5, zorder=0)
    ax.text(2, mult + 0.05, f'inserted in raw frame t0\n({mult:g}x threshold)', ha='center', va='bottom',
            fontsize=10, color=INK2)
    ax.axhline(1, color=INK, ls='--', lw=1.5)
    ax.text(4.55, 1.05, 'detection threshold (null floor)', ha='right', va='bottom', fontsize=10, color=INK)
    ax.set_xticks(range(5), ['t0-2', 't0-1', 't0', 't0+1', 't0+2'])
    ax.set_xlabel('output frame')
    ax.set_ylabel('blob amplitude in output\n(x detection threshold)')
    ax.set_ylim(0, 4.8)
    ax.legend(loc='upper left')
    headline(ax, 1, 'No ghost features in neighbouring frames',
             'A blob added to one raw frame never reaches the detection threshold in any output frame')


def panel_noise(ax, sweeps):
    w = 0.38
    for i, (lab, s) in enumerate(sweeps):
        y = [100 * s['kernel'][str(k)]['passthrough'] for k in K]
        bars = ax.bar(np.arange(5) + (i - 0.5) * w, y, w * 0.92, color=SERIES[i], label=lab)
        bar_labels(ax, bars, '{:.2f}')
    ax.set_xticks(range(5), ['t-2', 't-1', 't\n(own frame)', 't+1', 't+2'])
    ax.set_xlabel('raw input frame')
    ax.set_ylabel('noise passed to output of frame t\n(% per pixel)')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left')
    headline(ax, 2, "A frame's own noise is not copied to its output",
             'Own frame: <0.1%; each neighbour: ~0.6%  (blind spot works; total noise power ~0.2%)')


def panel_smoothing(ax, sweeps):
    cats = ['faint blob added\n(ideal 1)', 'particle removed\n(ideal 0)', 'jump: old position\n(ideal 0)',
            'jump: new position\n(ideal 1)']
    w = 0.38
    for i, (lab, s) in enumerate(sweeps):
        key = max((k for k in s['insert'] if k.startswith('sigma5')), key=lambda k: float(k.split('_x')[1]))
        y = [s['insert'][key]['0'], s['remove']['0'], jump_mean(s, 'old'), jump_mean(s, 'new')]
        bars = ax.bar(np.arange(4) + (i - 0.5) * w, y, w * 0.92, color=SERIES[i], label=lab)
        bar_labels(ax, bars)
    ax.set_xticks(range(4), cats)
    ax.set_ylabel('shown in output frame t0\n(fraction of the change)')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper center', ncol=2)
    headline(ax, 3, 'One-frame events are heavily smoothed',
             'Edit in raw frame t0 only (t0±1, t0±2 untouched); bars = output of frame t0')


def panel_control(ax, prod, retr, lab):
    cats = ['particle removed\n(ideal 0)', 'jump 10 px: old position\n(ideal 0)', 'jump 10 px: new position\n(ideal 1)']
    j = lambda s, k: float(np.mean([s['jump_t0']['x10'][k], s['jump_t0']['y10'][k]]))
    w = 0.38
    for i, (name, s, c) in enumerate([('production model', prod, PROD),
                                      ('retrained on a stack with disappearances', retr, RETR)]):
        y = [s['remove']['0'], j(s, 'old'), j(s, 'new')]
        bars = ax.bar(np.arange(3) + (i - 0.5) * w, y, w * 0.92, color=c, label=name)
        bar_labels(ax, bars)
    ax.set_xticks(range(3), cats)
    ax.set_ylabel('shown in output frame t0\n(fraction)')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    headline(ax, 4, 'The smoothing is learned from the data',
             f'{lab}, tested on new events that neither model was trained on')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--sweep', nargs='+', required=True, help='LABEL=DIR')
    ap.add_argument('--control', nargs=2, default=None, metavar=('PROD_DIR', 'RETRAINED_DIR'))
    ap.add_argument('--control_label', default='')
    a = ap.parse_args()
    sweeps = [(p.split('=', 1)[0], load(p.split('=', 1)[1])) for p in a.sweep]

    fig, axs = plt.subplots(2, 2, figsize=(17, 12.5))
    panel_ghosts(axs[0, 0], sweeps)
    panel_noise(axs[0, 1], sweeps)
    panel_smoothing(axs[1, 0], sweeps)
    if a.control:
        panel_control(axs[1, 1], load(a.control[0]), load(a.control[1]), a.control_label)
    else:
        axs[1, 1].axis('off')
    for ax in axs.flat:
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout(h_pad=4, w_pad=3)
    fig.savefig(a.out, dpi=110, facecolor='white')


if __name__ == '__main__':
    main()
