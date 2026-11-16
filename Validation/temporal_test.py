"""Tier 3: temporal behaviour of the 5-frame input.

SHINE predicts the centre frame from the centre frame (spatial blind spot) and four fully visible
neighbouring frames, so it learns to lean on the neighbours. These tests edit the RAW input of one
frame t0 (real noise and context stay) and measure the change in the outputs of t0-2 ... t0+2:

  insert  - a Gaussian blob added to frame t0 only. Output t0: is a one-frame event kept?
            Outputs t0+-1, t0+-2: does it leak into frames where it does not exist?
  remove  - a real small particle subtracted from frame t0 only (profile from a registered
            temporal mean that excludes the window). Output t0: is it filled back in?
  jump    - the same particle moved by d px in frame t0 only. Output t0: where is it put?
  neighbours - output change and accuracy when the four neighbours are replaced (far frames,
            noise only, copies of the centre), and the noise transfer kernel: the fraction of
            frame t+k noise that passes into output t.

Responses are differences between edited and unedited outputs of the same model, so the noise
cancels and the measurement is exact for that model. Amplitudes are fitted with the known profile
(least squares, with a constant background where needed).

Subcommands
  sweep         RAW DEN CKPT OUT [--null_summary J]   all tests with one model
  make-control  RAW DEN OUT [--null_summary J]        edited stack + event list for retraining
  eval-control  RAW DEN CKPT_PROD CKPT_RETRAINED CONTROL_DIR   same events, both models
"""
import argparse
import json
import os
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import list_tifs, read_stack, load_shine, denoise_window, affine_fit
from Utils.Utils import idxreturn
from residual_test import fourier_shift, register
from null_test import blob_score, shaped_noise
from skimage.feature import peak_local_max

AMP_MULT = (0.5, 1, 2, 4)      # insertion amplitude, x (2 x null floor) ~ x null-floor blob score
INSERT_SIGMAS = (2, 5)
JUMPS = (2, 5, 10, 20)
OFFSETS = (-2, -1, 0, 1, 2)


# ----------------------------------------------------------------------------- setup helpers
class Stack:
    def __init__(self, raw_dir, den_dir):
        self.names = list_tifs(raw_dir)
        self.raw, _ = read_stack(raw_dir, self.names)
        self.den, _ = read_stack(den_dir, self.names)
        self.T, self.H, self.W = self.raw.shape
        self._shift = {}

    def shift_to(self, s, t):
        """(dy, dx) moving frame s onto frame t (registration of the production outputs)."""
        key = (s, t)
        if key not in self._shift:
            dy, dx, _ = register(self.den[t], self.den[s])
            self._shift[key] = (dy, dx)
        return self._shift[key]

    def temporal_mean(self, t0, half=12, exclude=2):
        """Registered mean of raw frames around t0, excluding the frames of t0's input windows."""
        acc, n = np.zeros((self.H, self.W)), 0
        for s in range(max(0, t0 - half), min(self.T, t0 + half + 1)):
            if abs(s - t0) <= exclude + 2:
                continue
            dy, dx = self.shift_to(s, t0)
            acc += fourier_shift(self.raw[s], dy, dx)
            n += 1
        return acc / n


class Model:
    """Production-equivalent inference in z units, with the per-frame raw affine map of the
    unedited output, so edited and unedited outputs share one intensity scale."""

    def __init__(self, ckpt, stack, model='3x3_blind', frame_num=5):
        self.net = load_shine(ckpt, model, frame_num=frame_num)
        self.F = frame_num
        self.st = stack
        self.base = {}
        self.coef = {}

    def window(self, t):
        return self.st.raw[idxreturn(t, self.st.T, self.F)].copy()

    def unedited(self, t):
        if t not in self.base:
            out = denoise_window(self.net, self.window(t), rescale=False)
            m = 40
            self.coef[t] = affine_fit(out[m:-m, m:-m], self.st.raw[t][m:-m, m:-m])
            self.base[t] = out
        a, b = self.coef[t]
        return a * self.base[t] + b

    def edited(self, t, edits):
        """edits: {frame index: additive image}. Returns the output of t on the raw scale."""
        self.unedited(t)
        idx = idxreturn(t, self.st.T, self.F)
        win = self.window(t)
        for k, s in enumerate(idx):
            if s in edits:
                win[k] = win[k] + edits[s]
        a, b = self.coef[t]
        return a * denoise_window(self.net, win, rescale=False) + b

    def edited_window(self, t, win):
        self.unedited(t)
        a, b = self.coef[t]
        return a * denoise_window(self.net, win, rescale=False) + b


def gauss(shape, cy, cx, s):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))


def disk(shape, cy, cx, r):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r * r


def background_sites(stack, n=8, size=128, sep=200, border=96):
    tm = stack.raw.mean(0)
    sc, _ = blob_score(cv.GaussianBlur(tm, (0, 0), 2))
    cand = []
    for y in range(border, stack.H - border - size + 1, 32):
        for x in range(border, stack.W - border - size + 1, 32):
            cand.append((np.percentile(sc[y:y + size, x:x + size], 99.5), y + size // 2, x + size // 2))
    cand.sort()
    sites = []
    for _, y, x in cand:
        if all(np.hypot(y - yy, x - xx) >= sep for yy, xx in sites):
            sites.append((y, x))
        if len(sites) == n:
            break
    return sites


def den_raw(stack, t, cache):
    """Production output of frame t on the raw intensity scale."""
    if t not in cache:
        m = 40
        a, b = affine_fit(stack.den[t][m:-m, m:-m], stack.raw[t][m:-m, m:-m])
        cache[t] = a * stack.den[t] + b
    return cache[t]


def particle_sites(stack, t, cache, n=8, border=96, r_min=4, r_max=50, gap=20, frac=0.15):
    """Isolated particles in frame t: connected components of the smoothed production output
    above background + frac x (99.9th percentile - background), equivalent radius r_min..r_max px,
    no other component within `gap` px. Returns (y, x, radius)."""
    img = cv.GaussianBlur(den_raw(stack, t, cache), (0, 0), 2)
    bg = np.median(img)
    thr = bg + frac * (np.percentile(img, 99.9) - bg)
    lab_n, lab, stats, cent = cv.connectedComponentsWithStats((img > thr).astype(np.uint8), connectivity=8)
    kern = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
    out = []
    order = sorted(range(1, lab_n), key=lambda i: -stats[i, cv.CC_STAT_AREA])
    for i in order:
        r = np.sqrt(stats[i, cv.CC_STAT_AREA] / np.pi)
        cx, cy = cent[i]
        if not r_min <= r <= r_max:
            continue
        if not (border <= cy < stack.H - border and border <= cx < stack.W - border):
            continue
        own = cv.dilate((lab == i).astype(np.uint8), kern).astype(bool)
        if (own & (lab > 0) & (lab != i)).any():
            continue
        out.append((int(round(cy)), int(round(cx)), float(r)))
        if len(out) == n:
            break
    return out


def particle_profile(stack, t0, y, x, r, cache):
    """Particle signal in frame t0: production output (raw scale) minus the annulus background,
    tapered to zero between r + 2 and r + 6. Returns (profile, crop slices)."""
    M = den_raw(stack, t0, cache)
    R0, Rc = r + 2, r + 10
    h = int(Rc + 26)
    sl = (slice(y - h, y + h + 1), slice(x - h, x + h + 1))
    sub = M[sl]
    yy, xx = np.indices(sub.shape)
    rr = np.hypot(yy - h, xx - h)
    bg = np.median(sub[(rr >= Rc - 4) & (rr <= Rc + 2)])
    taper = np.clip((Rc - 4 - rr) / (Rc - 4 - R0), 0, 1)
    taper = 0.5 - 0.5 * np.cos(np.pi * taper)
    p = np.clip(sub - bg, 0, None) * taper
    return p, sl


def lsq(y, cols):
    X = np.stack([c.ravel() for c in cols], 1)
    return np.linalg.lstsq(X, y.ravel(), rcond=None)[0]


def frames_t0(stack, n=10):
    return np.linspace(6, stack.T - 7, n).astype(int)


# ----------------------------------------------------------------------------- tests
def test_insert(model, stack, sites, floor, t0s, amp_mult=AMP_MULT, sigmas=INSERT_SIGMAS):
    rows, example = [], None
    for s in sigmas:
        for mult in amp_mult:
            A = 2 * floor * mult
            for t0 in t0s:
                add = np.zeros((stack.H, stack.W), np.float32)
                for (y, x) in sites:
                    add += A * gauss(add.shape, y, x, s)
                for k in OFFSETS:
                    out_e = model.edited(t0 + k, {t0: add})
                    d = out_e - model.unedited(t0 + k)
                    for i, (y, x) in enumerate(sites):
                        R = int(3 * s + 4)
                        sl = (slice(y - R, y + R + 1), slice(x - R, x + R + 1))
                        g = gauss(add.shape, y, x, s)[sl]
                        amp = lsq(d[sl], [g])[0]
                        rows.append(dict(test='insert', sigma=s, mult=mult, A=A, t0=int(t0), k=k, site=i,
                                         frac=float(amp / A)))
                if s == sigmas[-1] and mult == 2 and t0 == t0s[len(t0s) // 2]:
                    y, x = sites[0]
                    sl = (slice(y - 40, y + 41), slice(x - 40, x + 41))
                    example = dict(raw=[(stack.raw[t0 + k] + (add if k == 0 else 0))[sl] for k in OFFSETS],
                                   out=[model.edited(t0 + k, {t0: add})[sl] for k in OFFSETS])
    return rows, example


def particle_events(stack, t0s, cache):
    ev = []
    for t0 in t0s:
        for i, (y, x, s) in enumerate(particle_sites(stack, int(t0), cache)):
            p, sl = particle_profile(stack, t0, y, x, s, cache)
            ev.append(dict(t0=int(t0), part=i, y=y, x=x, s=s, p=p, sl=sl))
    return ev


def place(shape, p, sl, dy=0, dx=0):
    img = np.zeros(shape, np.float32)
    img[sl] = np.roll(p, (dy, dx), (0, 1))
    return img


def test_remove(model, stack, events):
    rows, example = [], None
    by_t0 = {}
    for e in events:
        by_t0.setdefault(e['t0'], []).append(e)
    for t0, evs in by_t0.items():
        sub = sum(place((stack.H, stack.W), e['p'], e['sl']) for e in evs)
        for k in OFFSETS:
            out_u = model.unedited(t0 + k)
            out_e = model.edited(t0 + k, {t0: -sub})
            for e in evs:
                sl, p = e['sl'], e['p']
                one = np.ones_like(p)
                a_u = lsq(out_u[sl], [p, one])[0]
                a_e = lsq(out_e[sl], [p, one])[0]
                # how much of the particle is left in the edited raw frame t0 (0 = fully removed)
                r_u = lsq(stack.raw[t0][sl], [p, one])[0]
                r_e = lsq((stack.raw[t0] - sub)[sl], [p, one])[0]
                rows.append(dict(test='remove', t0=t0, k=k, part=e['part'], radius=e['s'],
                                 amp_unedited=float(a_u), frac=float(a_e / a_u) if a_u > 0 else np.nan,
                                 raw_left=float(r_e / r_u) if r_u > 0 else np.nan))
        if example is None:
            e = evs[0]
            sl = e['sl']
            example = dict(raw=[(stack.raw[t0 + k] - (sub if k == 0 else 0))[sl] for k in OFFSETS],
                           out=[model.edited(t0 + k, {t0: -sub})[sl] for k in OFFSETS],
                           out_u=[model.unedited(t0 + k)[sl] for k in OFFSETS])
    return rows, example


def test_jump(model, stack, events, jumps=JUMPS):
    rows, examples = [], {}
    by_t0 = {}
    for e in events:
        by_t0.setdefault(e['t0'], []).append(e)
    for d in jumps:
        for axis, (dy, dx) in (('x', (0, d)), ('y', (d, 0))):
            for t0, evs in by_t0.items():
                edit = sum(place((stack.H, stack.W), e['p'], e['sl'], dy, dx) -
                           place((stack.H, stack.W), e['p'], e['sl']) for e in evs)
                for k in OFFSETS:
                    out_u = model.unedited(t0 + k)
                    out_e = model.edited(t0 + k, {t0: edit})
                    for e in evs:
                        sl, p = e['sl'], e['p']
                        pn = np.roll(p, (dy, dx), (0, 1))
                        one = np.ones_like(p)
                        a_u = lsq(out_u[sl], [p, one])[0]
                        a_old, a_new, _ = lsq(out_e[sl], [p, pn, one])
                        rows.append(dict(test='jump', d=d, axis=axis, t0=t0, k=k, part=e['part'],
                                         amp_unedited=float(a_u), old=float(a_old / a_u) if a_u > 0 else np.nan,
                                         new=float(a_new / a_u) if a_u > 0 else np.nan))
                if axis == 'x' and d == 10 and d not in examples:
                    e = evs[0]
                    sl = e['sl']
                    examples[d] = dict(raw=[(stack.raw[t0 + k] + (edit if k == 0 else 0))[sl] for k in OFFSETS],
                                       out=[model.edited(t0 + k, {t0: edit})[sl] for k in OFFSETS])
    return rows, examples


def test_neighbours(model, stack, ts, amp_spec, nstd, rng):
    rows = []
    m = 40
    c = (slice(m, -m), slice(m, -m))
    for t in ts:
        if t + 4 >= stack.T:
            continue
        dy, dx = stack.shift_to(t + 3, t)
        target = fourier_shift(stack.raw[t + 3], dy, dx)
        dy4, dx4 = stack.shift_to(t + 4, t)
        nv = 0.5 * np.var((target - fourier_shift(stack.raw[t + 4], dy4, dx4))[c])
        base = model.window(t)
        ref = model.unedited(t)
        if model.F == 1:      # no neighbours: only the own-frame noise pass-through is defined
            delta = shaped_noise(base[0].shape, amp_spec, 0.5 * nstd, rng).astype(np.float32)
            dout = model.edited_window(t, base + delta) - ref
            rows.append(dict(test='kernel', t=int(t), k=0,
                             passthrough=float(np.sum(dout[c] * delta[c]) / np.sum(delta[c] ** 2)),
                             power_transfer=float(np.var(dout[c]) / np.var(delta[c]))))
            continue
        far = [(t + stack.T // 2 + j) % stack.T for j in (-2, -1, 1, 2)]
        variants = {'real neighbours': base}
        w = base.copy()
        w[[0, 1, 3, 4]] = stack.raw[far]
        variants['far frames as neighbours'] = w
        w = base.copy()
        for k in (0, 1, 3, 4):
            w[k] = stack.raw[t].mean() + shaped_noise(base[k].shape, amp_spec, nstd, rng)
        variants['noise-only neighbours'] = w
        w = base.copy()
        w[[0, 1, 3, 4]] = base[2]
        variants['centre copied as neighbours'] = w
        ref = model.unedited(t)
        for name, win in variants.items():
            out = ref if name == 'real neighbours' else model.edited_window(t, win)
            rows.append(dict(test='neighbours', t=int(t), variant=name,
                             held_out=float((np.mean((out[c] - target[c]) ** 2) - nv) / nv),
                             rms_change_sigma=float(np.sqrt(np.mean((out - ref)[c] ** 2)) / nstd)))
        # noise transfer kernel: realistic noise (0.5 sigma) added to one input frame
        for k in OFFSETS:
            delta = shaped_noise(base[0].shape, amp_spec, 0.5 * nstd, rng).astype(np.float32)
            win = base.copy()
            win[k + 2] += delta
            dout = model.edited_window(t, win) - ref
            rows.append(dict(test='kernel', t=int(t), k=k,
                             passthrough=float(np.sum(dout[c] * delta[c]) / np.sum(delta[c] ** 2)),
                             power_transfer=float(np.var(dout[c]) / np.var(delta[c]))))
    return rows


def noise_model(stack):
    """Full-frame noise amplitude spectrum and std from consecutive-frame differences."""
    spec = np.zeros((stack.H, stack.W))
    n = []
    for t in range(0, stack.T - 1, max(1, stack.T // 20)):
        dy, dx = stack.shift_to(t + 1, t)
        d = (stack.raw[t] - fourier_shift(stack.raw[t + 1], dy, dx)) / np.sqrt(2)
        spec += np.abs(np.fft.fft2(d - d.mean())) ** 2
        n.append(d[40:-40, 40:-40].std())
    return np.sqrt(spec / spec.mean()), float(np.mean(n))


# ----------------------------------------------------------------------------- figures
def strip(ax, ex, title, keys=('raw', 'out')):
    lo, hi = np.percentile(np.concatenate([e.ravel() for e in ex['out']]), [1, 99.5])
    for r, key in enumerate(keys):
        imgs = ex[key]
        row = np.hstack([np.pad(i, ((0, 0), (0, 3)), constant_values=hi) for i in imgs])
        if key == 'raw':
            l2, h2 = np.percentile(row, [1, 99.5])
            ax[r].imshow(row, cmap='gray', vmin=l2, vmax=h2)
        else:
            ax[r].imshow(row, cmap='gray', vmin=lo, vmax=hi)
        ax[r].set_title(f'{title}: {key} frames t0-2 ... t0+2')
        ax[r].axis('off')


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def col(rows, key):
    return np.array([r[key] for r in rows], float)


def uniq(rows, key):
    return sorted({r[key] for r in rows})


def by(rows, keys, val):
    """{key tuple: (mean, std, n)} of val over rows grouped by keys."""
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r[val])
    return {k: (float(np.nanmean(v)), float(np.nanstd(v)), len(v)) for k, v in sorted(out.items())}


def summarise_and_plot(rows, examples, out_dir, floor, nstd):
    with open(os.path.join(out_dir, 'temporal_rows.json'), 'w') as fh:
        json.dump(rows, fh)
    summary = {}
    fig = plt.figure(figsize=(24, 26))
    gs = fig.add_gridspec(6, 4)

    ins = sel(rows, test='insert')
    if ins:
        g = by(ins, ('sigma', 'mult', 'k'), 'frac')
        summary['insert'] = {}
        for (s, m, k), (mu, sd, n) in g.items():
            summary['insert'].setdefault(f'sigma{s:g}_x{m:g}', {})[int(k)] = round(mu, 4)
        leak = [abs(r['frac'] * r['A']) / (2 * floor) for r in ins if r['k'] != 0]
        summary['insert_max_leak_over_floor_amplitude'] = float(np.max(leak))
        for j, s in enumerate(uniq(ins, 'sigma')):
            ax = fig.add_subplot(gs[0, j])
            for m in uniq(ins, 'mult'):
                ks = [k for k in OFFSETS if (s, m, k) in g]
                ax.errorbar(ks, [g[(s, m, k)][0] for k in ks], [g[(s, m, k)][1] for k in ks], marker='o',
                            capsize=3, label=f'A = {2 * floor * m:.0f} ({2 * floor * m / nstd:.2f} sigma)')
            ax.axhline(0, color='k', lw=0.5)
            ax.axhline(1, color='gray', lw=0.5, ls='--')
            ax.set_xlabel('output frame - t0')
            ax.set_ylabel('fitted amplitude / inserted amplitude')
            ax.set_title(f'inserted blob sigma {s:g} px, frame t0 only')
            ax.legend(fontsize=8)
    rem = sel(rows, test='remove')
    if rem:
        g = by(rem, ('k',), 'frac')
        summary['remove'] = {int(k[0]): round(v[0], 4) for k, v in g.items()}
        summary['remove_unedited_amplitude_median'] = float(np.median(col(sel(rem, k=0), 'amp_unedited')))
        summary['remove_raw_left_median'] = float(np.nanmedian(col(sel(rem, k=0), 'raw_left')))
        summary['remove_n_events'] = len(sel(rem, k=0))
        ax = fig.add_subplot(gs[0, 2])
        ax.errorbar([k[0] for k in g], [v[0] for v in g.values()], [v[1] for v in g.values()], marker='o', capsize=3)
        ax.axhline(0, color='k', lw=0.5)
        ax.axhline(1, color='gray', lw=0.5, ls='--')
        ax.set_xlabel('output frame - t0')
        ax.set_ylabel('particle amplitude, edited / unedited')
        ax.set_title('real particle removed from frame t0 only\n(ideal: 0 at t0, 1 elsewhere)')
    jmp = sel(rows, test='jump')
    if jmp:
        j0 = sel(jmp, k=0)
        go, gn = by(j0, ('axis', 'd'), 'old'), by(j0, ('axis', 'd'), 'new')
        summary['jump_t0'] = {f'{a}{d}': dict(old=round(go[(a, d)][0], 3), new=round(gn[(a, d)][0], 3)) for a, d in go}
        jn = [r for r in jmp if r['k'] != 0]
        no, nn = by(jn, ('d',), 'old'), by(jn, ('d',), 'new')
        summary['jump_neighbours'] = {int(d[0]): dict(old=round(no[d][0], 3), new=round(nn[d][0], 3)) for d in no}
        ax = fig.add_subplot(gs[0, 3])
        for axis, mk in (('x', 'o'), ('y', 's')):
            ds = [d for d in JUMPS if (axis, d) in go]
            ax.plot(ds, [gn[(axis, d)][0] for d in ds], marker=mk, label=f'new position ({axis})')
            ax.plot(ds, [go[(axis, d)][0] for d in ds], marker=mk, ls='--', label=f'old position ({axis})')
        ax.axhline(1, color='gray', lw=0.5, ls='--')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xscale('log')
        ax.set_xticks(JUMPS)
        ax.set_xticklabels(JUMPS)
        ax.set_xlabel('jump d (px), frame t0 only')
        ax.set_ylabel('amplitude in output t0 / unedited')
        ax.set_title('particle moved in frame t0 only\n(ideal: new 1, old 0)')
        ax.legend(fontsize=8)
    nb = sel(rows, test='neighbours')
    if nb:
        gh, gr = by(nb, ('variant',), 'held_out'), by(nb, ('variant',), 'rms_change_sigma')
        names = [v[0] for v in gh]
        summary['neighbours'] = {v[0]: dict(held_out=round(gh[v][0], 4), rms_change_sigma=round(gr[v][0], 4)) for v in gh}
        ax = fig.add_subplot(gs[1, 0])
        ax.barh(names, [gh[(n,)][0] for n in names])
        ax.set_xlabel('held-out error vs raw t+3 / noise variance')
        ax.set_title('which neighbours (output of frame t)')
        ax = fig.add_subplot(gs[1, 1])
        ax.barh(names, [gr[(n,)][0] for n in names])
        ax.set_xlabel('RMS change from real-neighbour output (noise sigma)')
        ax.set_title('dependence on neighbours')
    ker = sel(rows, test='kernel')
    if ker:
        gp, gt = by(ker, ('k',), 'passthrough'), by(ker, ('k',), 'power_transfer')
        ks = np.array([k[0] for k in gp])
        summary['kernel'] = {int(k[0]): dict(passthrough=round(gp[k][0], 4), power_transfer=round(gt[k][0], 4)) for k in gp}
        ax = fig.add_subplot(gs[1, 2])
        ax.bar(ks - 0.2, [v[0] for v in gp.values()], 0.4, label='same-pixel pass-through')
        ax.bar(ks + 0.2, [v[0] for v in gt.values()], 0.4, label='noise power transfer')
        ax.set_xlabel('input frame - output frame')
        ax.legend()
        ax.set_title('noise transfer kernel (0.5 sigma realistic noise added)')
    row = 2
    for name, ex in examples.items():
        if ex is None or row > 5:
            continue
        ax0 = fig.add_subplot(gs[row, :2])
        ax1 = fig.add_subplot(gs[row, 2:])
        strip([ax0, ax1], ex, name)
        row += 1
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'temporal_test.png'), dpi=65)
    return summary


# ----------------------------------------------------------------------------- control
def control_events(stack, floor, tm_cache):
    sites = background_sites(stack)
    t0s = frames_t0(stack, 8)
    gap = max(5, int(np.median(np.diff(t0s))) // 2)
    t0s_p = [int(t) + gap for t in t0s if t + gap <= stack.T - 7]
    ins = [dict(test='insert', t0=int(t0), site=i, y=y, x=x, sigma=INSERT_SIGMAS[i % 2], A=2 * floor * 2)
           for t0 in t0s for i, (y, x) in enumerate(sites)]
    rem = [dict(test='remove', t0=t0, part=i, y=y, x=x, s=s) for t0 in t0s_p
           for i, (y, x, s) in enumerate(particle_sites(stack, t0, tm_cache))]
    return ins, rem


def control_edits(stack, ins, rem, tm_cache):
    edits = {}
    for e in ins:
        edits.setdefault(e['t0'], np.zeros((stack.H, stack.W), np.float32))
        edits[e['t0']] += e['A'] * gauss((stack.H, stack.W), e['y'], e['x'], e['sigma'])
    profiles = {}
    for e in rem:
        p, sl = particle_profile(stack, e['t0'], e['y'], e['x'], e['s'], tm_cache)
        profiles[(e['t0'], e['part'])] = (p, sl)
        edits.setdefault(e['t0'], np.zeros((stack.H, stack.W), np.float32))
        edits[e['t0']] -= place((stack.H, stack.W), p, sl)
    return edits, profiles


def eval_control_model(model, stack, ins, rem, edits, profiles):
    rows = []
    for t0 in sorted(edits):
        for k in OFFSETS:
            out_e = model.edited(t0 + k, {t0: edits[t0]})
            out_u = model.unedited(t0 + k)
            d = out_e - out_u
            for e in [e for e in ins if e['t0'] == t0]:
                s = e['sigma']
                R = int(3 * s + 4)
                sl = (slice(e['y'] - R, e['y'] + R + 1), slice(e['x'] - R, e['x'] + R + 1))
                g = gauss((stack.H, stack.W), e['y'], e['x'], s)[sl]
                rows.append(dict(test='insert', sigma=s, t0=t0, k=k, site=e['site'],
                                 frac=float(lsq(d[sl], [g])[0] / e['A'])))
            for e in [e for e in rem if e['t0'] == t0]:
                p, sl = profiles[(t0, e['part'])]
                one = np.ones_like(p)
                a_u = lsq(out_u[sl], [p, one])[0]
                a_e = lsq(out_e[sl], [p, one])[0]
                r_u = lsq(stack.raw[t0][sl], [p, one])[0]
                r_e = lsq((stack.raw[t0] + edits[t0])[sl], [p, one])[0]
                rows.append(dict(test='remove', t0=t0, k=k, part=e['part'], amp_unedited=float(a_u),
                                 frac=float(a_e / a_u) if a_u > 0 else np.nan,
                                 raw_left=float(r_e / r_u) if r_u > 0 else np.nan))
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    a1 = sp.add_parser('sweep')
    a1.add_argument('raw_dir'); a1.add_argument('den_dir'); a1.add_argument('ckpt'); a1.add_argument('out_dir')
    a1.add_argument('--null_summary', required=True)
    a1.add_argument('--model', default='3x3_blind')
    a1.add_argument('--frame_num', type=int, default=5)
    a1.add_argument('--n_t0', type=int, default=10)
    a1.add_argument('--t0s', type=int, nargs='+', default=None, help='explicit event frames (overrides --n_t0)')
    a1.add_argument('--tests', default='insert,remove,jump,neighbours')
    a1.add_argument('--seed', type=int, default=0)
    a2 = sp.add_parser('make-control')
    a2.add_argument('raw_dir'); a2.add_argument('den_dir'); a2.add_argument('out_dir')
    a2.add_argument('--null_summary', required=True)
    a3 = sp.add_parser('eval-control')
    a3.add_argument('raw_dir'); a3.add_argument('den_dir')
    a3.add_argument('control_dir')
    a3.add_argument('--prod', nargs='+', required=True, help='production checkpoints')
    a3.add_argument('--retrained', nargs='+', required=True, help='checkpoints trained on the edited stack')
    a3.add_argument('--model', default='3x3_blind')
    a = ap.parse_args()

    if a.cmd == 'sweep':
        os.makedirs(a.out_dir, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        floor = json.load(open(a.null_summary))['null_floor_score']
        st = Stack(a.raw_dir, a.den_dir)
        model = Model(a.ckpt, st, a.model, a.frame_num)
        tests = a.tests.split(',')
        t0s = np.array(a.t0s) if a.t0s else frames_t0(st, a.n_t0)
        sites = background_sites(st)
        amp_spec, nstd = noise_model(st)
        tm_cache = {}
        rows, examples = [], {}
        meta = dict(floor=floor, noise_sigma=nstd, t0s=t0s.tolist(), sites=sites)
        if 'insert' in tests:
            r, ex = test_insert(model, st, sites, floor, t0s)
            rows += r
            examples['insert (sigma 5, A = 4 x floor)'] = ex
            print('insert done', flush=True)
        if 'remove' in tests or 'jump' in tests:
            events = particle_events(st, t0s, tm_cache)
            meta['particle_events'] = [(e['t0'], e['y'], e['x'], round(e['s'], 1)) for e in events]
        if 'remove' in tests:
            r, ex = test_remove(model, st, events)
            rows += r
            examples['remove'] = ex
            print('remove done', flush=True)
        if 'jump' in tests:
            r, exs = test_jump(model, st, events)
            rows += r
            for d, ex in exs.items():
                examples[f'jump {d} px (x)'] = ex
            print('jump done', flush=True)
        if 'neighbours' in tests:
            rows += test_neighbours(model, st, t0s, amp_spec, nstd, rng)
            print('neighbours done', flush=True)
        summary = summarise_and_plot(rows, examples, a.out_dir, floor, nstd)
        summary['meta'] = meta
        with open(os.path.join(a.out_dir, 'temporal_summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=1))

    elif a.cmd == 'make-control':
        floor = json.load(open(a.null_summary))['null_floor_score']
        st = Stack(a.raw_dir, a.den_dir)
        tm_cache = {}
        ins, rem = control_events(st, floor, tm_cache)
        edits, _ = control_edits(st, ins, rem, tm_cache)
        sd = os.path.join(a.out_dir, 'stack')
        os.makedirs(sd, exist_ok=True)
        for t, n in enumerate(st.names):
            img = st.raw[t] + edits.get(t, 0)
            Image.fromarray(img.astype(np.float32)).save(os.path.join(sd, n))
        with open(os.path.join(a.out_dir, 'control_events.json'), 'w') as fh:
            json.dump(dict(insert=ins, remove=rem, floor=floor), fh, indent=1)
        print(f'{len(ins)} insertions, {len(rem)} removals in {len(edits)} frames -> {sd}')

    elif a.cmd == 'eval-control':
        ev = json.load(open(os.path.join(a.control_dir, 'control_events.json')))
        st = Stack(a.raw_dir, a.den_dir)
        tm_cache = {}
        edits, profiles = control_edits(st, ev['insert'], ev['remove'], tm_cache)
        res = {}
        runs = [('production', ck) for ck in a.prod] + [('retrained', ck) for ck in a.retrained]
        for group, ck in runs:
            label = f'{group} {os.path.splitext(os.path.basename(ck))[0]}'
            model = Model(ck, st, a.model)
            res[label] = eval_control_model(model, st, ev['insert'], ev['remove'], edits, profiles)
            print(label, 'done', flush=True)
        out = {}
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        for label, rows in res.items():
            with open(os.path.join(a.control_dir, f'control_rows_{label.replace(" ", "_")}.json'), 'w') as fh:
                json.dump(rows, fh)
            ls = '-' if label.startswith('production') else '--'
            for j, s in enumerate(INSERT_SIGMAS):
                g = by(sel(rows, test='insert', sigma=s), ('k',), 'frac')
                out[f'{label} insert sigma{s}'] = {int(k[0]): round(v[0], 4) for k, v in g.items()}
                ax[j].plot([k[0] for k in g], [v[0] for v in g.values()], marker='o', ls=ls, label=label)
                ax[j].set_title(f'inserted blob sigma {s} px (A = 4 x floor)')
            g = by(sel(rows, test='remove'), ('k',), 'frac')
            out[f'{label} remove'] = {int(k[0]): round(v[0], 4) for k, v in g.items()}
            out[f'{label} remove raw_left'] = float(np.nanmedian(col(sel(rows, test='remove', k=0), 'raw_left')))
            ax[2].plot([k[0] for k in g], [v[0] for v in g.values()], marker='o', ls=ls, label=label)
            ax[2].set_title('particle removed from frame t0')
        for x in ax:
            x.axhline(0, color='k', lw=0.5)
            x.axhline(1, color='gray', lw=0.5, ls='--')
            x.set_xlabel('output frame - t0')
            x.legend()
        ax[0].set_ylabel('response (fraction)')
        plt.tight_layout()
        plt.savefig(os.path.join(a.control_dir, 'control_comparison.png'), dpi=70)
        with open(os.path.join(a.control_dir, 'control_summary.json'), 'w') as fh:
            json.dump(out, fh, indent=2)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
