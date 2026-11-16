"""Tier 2, test 2: what does the model make out of pure noise?

A square patch is replaced, in all frames of the 5-frame input window, by a flat level plus
noise. Two kinds of noise:
  synthetic - Gaussian with the measured noise power spectrum (incl. the fast-scan correlation)
              and the noise variance at that level
  real      - real noise realisations, (raw_t - shift(raw_t+1)) / sqrt(2) cut from the same place
The rest of the frame stays real, so the network sees the same normalisation and context as in
production. Whatever bright blob the model puts inside the patch is a false detection.

The same multi-scale blob detector (scale-normalised LoG, score ~ blob amplitude / 2 in raw
units) runs on the null outputs and on every production denoised frame. The largest null score
is the false-detection floor: real detections below it cannot be told apart from noise.

A stress case runs whole windows of pure synthetic noise (per-frame z-score then inflates the
noise relative to production) and reports how much of the input noise survives.

Usage: python null_test.py RAW_DIR DENOISED_DIR CKPT OUT_DIR [--model 3x3_blind]
"""
import argparse
import json
import os
import numpy as np
import cv2 as cv
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import list_tifs, read_stack, load_shine, denoise_window, affine_fit
from Utils.Utils import idxreturn
from residual_test import fourier_shift, register

SIGMAS = (1.5, 2, 3, 4, 6, 8, 12)


def blob_score(img):
    """Max over scales of the scale-normalised LoG (bright blobs positive) and the best scale."""
    st = np.stack([-s * s * ndi.gaussian_laplace(img.astype(np.float64), s) for s in SIGMAS])
    return st.max(0), np.asarray(SIGMAS)[st.argmax(0)]


def peaks(score, sig, mask, min_distance=3):
    yx = peak_local_max(np.where(mask, score, -np.inf), min_distance=min_distance, threshold_abs=0,
                        exclude_border=False)
    return [(float(score[y, x]), float(sig[y, x]), int(y), int(x)) for y, x in yx]


def shaped_noise(shape, amp_spec, std, rng):
    w = rng.standard_normal(shape)
    n = np.real(np.fft.ifft2(np.fft.fft2(w) * amp_spec))
    return n / n.std() * std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir')
    ap.add_argument('den_dir')
    ap.add_argument('ckpt')
    ap.add_argument('out_dir')
    ap.add_argument('--model', default='3x3_blind')
    ap.add_argument('--frame_num', type=int, default=5)
    ap.add_argument('--patch', type=int, default=256)
    ap.add_argument('--n_patches', type=int, default=6)
    ap.add_argument('--n_frames', type=int, default=24, help='frames used for the null patches')
    ap.add_argument('--guard', type=int, default=40, help='px excluded inside patch / frame edges')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    names = list_tifs(a.raw_dir)
    raw, _ = read_stack(a.raw_dir, names)
    den, _ = read_stack(a.den_dir, names)
    T, H, W = raw.shape
    P, g = a.patch, a.guard
    net = load_shine(a.ckpt, a.model, frame_num=a.frame_num)
    F, c = a.frame_num, a.frame_num // 2

    # noise realisations (sign-alternating not needed: each is used on its own)
    noise = np.zeros_like(raw)
    for t in range(T - 1):
        dy, dx, _ = register(den[t], den[t + 1])
        noise[t] = (raw[t] - fourier_shift(raw[t + 1], dy, dx)) / np.sqrt(2)
    noise[-1] = noise[-2]

    # patch locations: particle-free background (lowest blob score in the time-mean raw)
    tm = raw.mean(0)
    tm_score, _ = blob_score(cv.GaussianBlur(tm, (0, 0), 2))
    cand = []
    for y in range(g, H - P - g + 1, 32):
        for x in range(g, W - P - g + 1, 32):
            cand.append((np.percentile(tm_score[y:y + P, x:x + P], 99.9), y, x))
    cand.sort()
    locs = []
    for _, y, x in cand:
        if all(abs(y - yy) >= P or abs(x - xx) >= P for yy, xx in locs):
            locs.append((y, x))
        if len(locs) == a.n_patches:
            break

    # noise power spectrum (patch-sized) and level from the noise realisations at those places
    spec = np.zeros((P, P))
    for (y, x) in locs:
        for t in range(0, T - 1, 4):
            spec += np.abs(np.fft.fft2(noise[t, y:y + P, x:x + P] - noise[t, y:y + P, x:x + P].mean())) ** 2
    amp_spec = np.sqrt(spec / spec.mean())

    frames = np.linspace(2, T - 3, a.n_frames).astype(int)
    inner = np.zeros((H, W), bool)
    rows = []
    examples = {}
    for kind in ('synthetic', 'real'):
        for (y, x) in locs:
            level = float(np.median(tm[y:y + P, x:x + P]))
            std = float(noise[:, y:y + P, x:x + P].std())
            for t in frames:
                idx = idxreturn(t, T, F)
                win = raw[idx].copy()
                for k, tt in enumerate(idx):
                    if kind == 'synthetic':
                        win[k, y:y + P, x:x + P] = level + shaped_noise((P, P), amp_spec, std, rng)
                    else:
                        win[k, y:y + P, x:x + P] = level + noise[tt, y:y + P, x:x + P]
                out = denoise_window(net, win)
                mask = np.ones((H, W), bool)
                mask[y:y + P, x:x + P] = False
                aa, bb = affine_fit(out, win[c], mask)
                fitted = aa * out + bb
                sc, sg = blob_score(fitted)
                inner[:] = False
                inner[y + g:y + P - g, x + g:x + P - g] = True
                pk = peaks(sc, sg, inner)
                patch_out = fitted[y + g:y + P - g, x + g:x + P - g]
                rows.append(dict(kind=kind, y=y, x=x, t=int(t), level=level, noise_std=std,
                                 out_std=float(patch_out.std()), out_mean_minus_level=float(patch_out.mean() - level),
                                 max_score=max([p[0] for p in pk], default=0.0),
                                 scores=[p[0] for p in pk]))
                if (y, x) == locs[0] and t == frames[len(frames) // 2]:
                    ey, ex = slice(max(0, y - 64), y + P + 64), slice(max(0, x - 64), x + P + 64)
                    examples[kind] = (win[c, ey, ex].copy(), fitted[ey, ex].copy())

    # stress: whole windows of pure synthetic noise (full-frame spectrum from the same shape)
    full_spec = np.zeros((H, W))
    for t in range(0, T - 1, 8):
        full_spec += np.abs(np.fft.fft2(noise[t] - noise[t].mean())) ** 2
    full_amp = np.sqrt(full_spec / full_spec.mean())
    bg = float(np.median(tm))
    nstd = float(noise.std())
    stress = []
    for r in range(8):
        win = np.stack([bg + shaped_noise((H, W), full_amp, nstd, rng) for _ in range(F)])
        out = denoise_window(net, win, rescale=False) * win[c].std() + win[c].mean()
        sc, sg = blob_score(out)
        m = np.zeros((H, W), bool)
        m[g:H - g, g:W - g] = True
        pk = peaks(sc, sg, m)
        stress.append(dict(out_std=float(out[g:-g, g:-g].std()), noise_std=nstd,
                           max_score=max([p[0] for p in pk], default=0.0)))
        if r == 0:
            examples['stress'] = (win[c, :P + 128, :P + 128].copy(), out[:P + 128, :P + 128].copy())

    # real detections on the production output
    real = []
    m = np.zeros((H, W), bool)
    m[g:H - g, g:W - g] = True
    for t in range(T):
        aa, bb = affine_fit(den[t][g:-g, g:-g], raw[t][g:-g, g:-g])
        sc, sg = blob_score(aa * den[t] + bb)
        for s, sgm, yy, xx in peaks(sc, sg, m):
            real.append((t, s, sgm, yy, xx))
    real = np.array(real)

    null_scores = np.concatenate([r['scores'] for r in rows])
    floor = float(max(r['max_score'] for r in rows))
    floor_syn = float(max(r['max_score'] for r in rows if r['kind'] == 'synthetic'))
    floor_real = float(max(r['max_score'] for r in rows if r['kind'] == 'real'))
    null_area = len(rows) * (P - 2 * g) ** 2
    real_area = T * (H - 2 * g) * (W - 2 * g)
    above = real[real[:, 1] > floor]
    summary = dict(
        patch_locations=locs, frames=frames.tolist(), n_null_patches=len(rows),
        null_floor_score=floor, null_floor_synthetic=floor_syn, null_floor_real_noise=floor_real,
        null_score_p999=float(np.percentile(null_scores, 99.9)),
        null_out_std_over_noise_std_median=float(np.median([r['out_std'] / r['noise_std'] for r in rows])),
        null_out_mean_minus_level_median=float(np.median([r['out_mean_minus_level'] for r in rows])),
        null_peaks_per_Mpx=float(len(null_scores) / null_area * 1e6),
        real_peaks_per_Mpx=float(len(real) / real_area * 1e6),
        real_peaks_above_floor=int(len(above)),
        real_peaks_above_floor_per_frame_median=float(np.median(np.bincount(above[:, 0].astype(int), minlength=T))),
        stress_out_std_over_noise_std=float(np.mean([s['out_std'] / s['noise_std'] for s in stress])),
        stress_max_score=float(max(s['max_score'] for s in stress)),
    )
    with open(os.path.join(a.out_dir, 'null_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(a.out_dir, 'null_patches.json'), 'w') as fh:
        json.dump([{k: v for k, v in r.items() if k != 'scores'} for r in rows], fh)
    np.save(os.path.join(a.out_dir, 'real_detections.npy'), real)

    # ---- figure ----
    fig, ax = plt.subplots(2, 4, figsize=(22, 11))
    for k, kind in enumerate(('synthetic', 'real', 'stress')):
        inp, out = examples[kind]
        lo, hi = np.percentile(inp, [1, 99])
        ax[0, k].imshow(np.hstack([inp, np.full((inp.shape[0], 8), hi), out]), cmap='gray', vmin=lo, vmax=hi)
        ax[0, k].set_title(f'{kind} null: input (left) / output (right), same scale')
        ax[0, k].axis('off')
    tmid = T // 2
    ax[0, 3].imshow(den[tmid], cmap='gray')
    for (y, x) in locs:
        ax[0, 3].add_patch(plt.Rectangle((x, y), P, P, fill=False, color='c', lw=1))
    sel = real[(real[:, 0] == tmid) & (real[:, 1] > floor)]
    ax[0, 3].scatter(sel[:, 4], sel[:, 3], s=(sel[:, 2] * 3) ** 2, facecolors='none', edgecolors='r', lw=1)
    ax[0, 3].set_title(f'frame {tmid}: null patch places (cyan), detections above floor (red)')
    ax[0, 3].axis('off')

    bins = np.linspace(0, max(real[:, 1].max(), floor) * 1.05, 120)
    ax[1, 0].hist(null_scores, bins, density=True, alpha=0.6, label=f'null patches ({len(null_scores)} peaks)')
    ax[1, 0].hist(real[:, 1], bins, density=True, alpha=0.6, label=f'real frames ({len(real)} peaks)')
    ax[1, 0].axvline(floor, color='r', label=f'null floor {floor:.0f}')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_xlabel('blob score (~ amplitude/2, raw units)')
    ax[1, 0].legend()
    ax[1, 0].set_title('blob score distributions')
    for kind, c in (('synthetic', 'C0'), ('real', 'C1')):
        ax[1, 1].hist([r['max_score'] for r in rows if r['kind'] == kind], 30, alpha=0.6, color=c, label=kind)
    ax[1, 1].set_xlabel('largest blob score in a null patch')
    ax[1, 1].legend()
    ax[1, 1].set_title('per-patch maximum')
    cnt = np.bincount(above[:, 0].astype(int), minlength=T)
    ax[1, 2].plot(cnt)
    ax[1, 2].set_xlabel('frame')
    ax[1, 2].set_ylabel('detections above floor')
    ax[1, 2].set_title('real detections above the null floor')
    ax[1, 3].scatter(real[:, 2] + rng.uniform(-0.3, 0.3, len(real)), real[:, 1], s=1, alpha=0.2, label='real')
    ax[1, 3].scatter(np.full(len(null_scores), 0.8) + rng.uniform(-0.3, 0.3, len(null_scores)), null_scores, s=1,
                     alpha=0.2, label='null (x=0.8)')
    ax[1, 3].axhline(floor, color='r')
    ax[1, 3].set_xlabel('best LoG sigma (px)')
    ax[1, 3].set_ylabel('blob score')
    ax[1, 3].legend()
    ax[1, 3].set_title('score vs blob size')
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, 'null_test.png'), dpi=70)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
