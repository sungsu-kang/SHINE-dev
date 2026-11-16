"""Tier 2, test 3 (analysis): do two independently trained models agree?

Model E saw only even frames, model O only odd frames. Even frame t (denoised by E) is compared
with odd frame t+1 (denoised by O) after registration. Their noise, training data and inputs are
independent, so structure that one model makes from noise does not appear in the other output.
The one-frame time step also contains real changes; every number is shown next to the same number
for raw frames, which see the same changes.

  1. FRC(den_E, den_O) vs FRC(raw_t, raw_t+1)
  2. held-out error: mean (est_t - raw_t+1)^2 - noise variance ~ true error + real change, for
     SHINE and for simple estimators built from the even half only (Gaussian blur, 5-frame mean)
  3. disagreement map: time-mean |den_E - den_O|
  4. blob reproducibility: detections above the null floor in den_E matched in den_O, by score
  5. spectral error E(k) = ring power of F[den_E] - F[raw_t+1], minus the noise ring power from
     frame differences, relative to the signal power S(k) = raw power - noise power, and the FRC
     between den_E and raw t+1. Sharp scan-interference peaks are masked out of the rings.
     E(k)/S(k) < 1 marks the frequencies the output gets right; like the held-out error it is an
     upper bound, since the real change between t and t+1 counts as error.
  6. optional --roi y0,y1,x0,x1 (repeatable): FRC and image crops for a region such as a particle,
     to check whether lattice fringes in the denoised output are reproduced by the other model

Usage: python evenodd_test.py RAW_DIR EVEN_DEN_DIR ODD_DEN_DIR OUT_DIR [--floor F | --null_summary J]
"""
import argparse
import json
import os
import numpy as np
import cv2 as cv
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import list_tifs, read_tif, affine_fit
from residual_test import fourier_shift, register
from null_test import blob_score, peaks


def frc(a, b, nbins=64):
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))
    fb = np.fft.fftshift(np.fft.fft2((b - b.mean()) * win))
    h, w = a.shape
    y, x = np.indices(a.shape)
    r = np.hypot((y - h // 2) / h, (x - w // 2) / w)
    bins = np.linspace(0, 0.5, nbins + 1)
    i = np.digitize(r.ravel(), bins) - 1
    ok = (i >= 0) & (i < nbins)
    num = np.bincount(i[ok], np.real(fa * np.conj(fb)).ravel()[ok], nbins)
    da = np.bincount(i[ok], (np.abs(fa) ** 2).ravel()[ok], nbins)
    db = np.bincount(i[ok], (np.abs(fb) ** 2).ravel()[ok], nbins)
    return 0.5 * (bins[1:] + bins[:-1]), num / np.sqrt(da * db + 1e-30)


def ring_average(p2, mask, nbins=64):
    """Mean of a centred 2D spectrum over rings, ignoring masked-out pixels."""
    h, w = p2.shape
    y, x = np.indices(p2.shape)
    r = np.hypot((y - h // 2) / h, (x - w // 2) / w)
    bins = np.linspace(0, 0.5, nbins + 1)
    i = np.digitize(r.ravel(), bins) - 1
    ok = (i >= 0) & (i < nbins) & mask.ravel()
    return 0.5 * (bins[1:] + bins[:-1]), np.bincount(i[ok], p2.ravel()[ok], nbins) / np.maximum(
        np.bincount(i[ok], minlength=nbins), 1)


def interference_mask(p_raw, dc=6, thresh=10):
    """False at sharp isolated spectral peaks (scan interference), True elsewhere."""
    from scipy.ndimage import median_filter, binary_dilation
    q = p_raw / median_filter(p_raw, size=9)
    h, w = p_raw.shape
    q[h // 2 - dc:h // 2 + dc + 1, w // 2 - dc:w // 2 + dc + 1] = 0
    return ~binary_dilation(q > thresh, iterations=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir')
    ap.add_argument('even_dir')
    ap.add_argument('odd_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--floor', type=float, default=None, help='blob score floor (default: from null summary)')
    ap.add_argument('--null_summary', default=None)
    ap.add_argument('--margin', type=int, default=40)
    ap.add_argument('--roi', action='append', default=[], help='y0,y1,x0,x1 (repeatable)')
    ap.add_argument('--skip_edges', type=int, default=2, help='pairs dropped at each end (mirrored windows)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    floor = a.floor
    if floor is None and a.null_summary:
        floor = json.load(open(a.null_summary))['null_floor_score']
    floor = floor or 0.0

    names = list_tifs(a.raw_dir)
    raw = np.stack([read_tif(os.path.join(a.raw_dir, n)) for n in names])
    T, H, W = raw.shape
    even = {n: read_tif(os.path.join(a.even_dir, n)) for n in names[0::2]}
    odd = {n: read_tif(os.path.join(a.odd_dir, n)) for n in names[1::2]}
    m = a.margin
    crop = (slice(m, H - m), slice(m, W - m))

    # noise variance (for the held-out error) from consecutive raw frames
    pairs = [(t, t + 1) for t in range(0, T - 1, 2)]
    pairs = pairs[a.skip_edges:len(pairs) - a.skip_edges]

    sig_list = (1, 2, 3)
    f = None
    frc_den, frc_raw = [], []
    held = {k: [] for k in ['SHINE (E model)'] + [f'Gaussian blur {s} px' for s in sig_list] + ['5-frame mean (even half)', 'raw frame']}
    noise_var = []
    dis_sum = np.zeros((H - 2 * m, W - 2 * m))
    dis_noise_sum = np.zeros_like(dis_sum)
    match_rows = []
    example = None
    rois = [tuple(int(v) for v in r.split(',')) for r in a.roi]
    roi_frc = {r: ([], []) for r in rois}
    roi_examples = {r: [] for r in rois}
    ch, cw = H - 2 * m, W - 2 * m
    win = np.outer(np.hanning(ch), np.hanning(cw))
    spec_keys = ('err', 'noise', 'A', 'B', 'AB', 'err_blur3', 'err_mean5')
    acc = {key: np.zeros((ch, cw)) for key in spec_keys}

    def wfft(x):
        x = x[crop]
        return np.fft.fftshift(np.fft.fft2((x - x.mean()) * win))

    for k, (t, s) in enumerate(pairs):
        dE, dO = even[names[t]], odd[names[s]]
        aE = affine_fit(dE[crop], raw[t][crop])
        aO = affine_fit(dO[crop], raw[s][crop])
        dE = aE[0] * dE + aE[1]
        dO = aO[0] * dO + aO[1]
        dy, dx, _ = register(dE, dO)
        dO_r = fourier_shift(dO, dy, dx)
        rO_r = fourier_shift(raw[s], dy, dx)
        f, c1 = frc(dE[crop], dO_r[crop])
        _, c2 = frc(raw[t][crop], rO_r[crop])
        frc_den.append(c1)
        frc_raw.append(c2)

        # noise variance of a raw frame from the raw pair itself
        nv = 0.5 * np.var((raw[t] - rO_r)[crop])
        noise_var.append(nv)
        target = rO_r[crop]
        ests = {'SHINE (E model)': dE}
        for sg in sig_list:
            ests[f'Gaussian blur {sg} px'] = cv.GaussianBlur(raw[t], (0, 0), sg)
        ev = [i for i in range(t - 4, t + 5, 2) if 0 <= i < T]
        ests['5-frame mean (even half)'] = raw[ev].mean(0)
        ests['raw frame'] = raw[t]
        for key, e in ests.items():
            held[key].append(float(np.mean((e[crop] - target) ** 2) - nv))

        FA, FB = wfft(dE), wfft(rO_r)
        FN = wfft((raw[t] - rO_r) / np.sqrt(2))
        acc['err'] += np.abs(FA - FB) ** 2
        acc['noise'] += np.abs(FN) ** 2
        acc['A'] += np.abs(FA) ** 2
        acc['B'] += np.abs(FB) ** 2
        acc['AB'] += np.real(FA * np.conj(FB))
        acc['err_blur3'] += np.abs(wfft(ests['Gaussian blur 3 px']) - FB) ** 2
        acc['err_mean5'] += np.abs(wfft(ests['5-frame mean (even half)']) - FB) ** 2

        for r in rois:
            sl = (slice(r[0], r[1]), slice(r[2], r[3]))
            fr, cd = frc(dE[sl], dO_r[sl], nbins=32)
            _, cr = frc(raw[t][sl], rO_r[sl], nbins=32)
            roi_frc[r][0].append(cd)
            roi_frc[r][1].append(cr)
            if k in (len(pairs) // 4, len(pairs) // 2, 3 * len(pairs) // 4):
                roi_examples[r].append((t, raw[t][sl].copy(), dE[sl].copy(), dO_r[sl].copy()))

        dis_sum += np.abs(dE - dO_r)[crop]
        dis_noise_sum += np.abs(raw[t] - rO_r)[crop]

        # blob reproducibility
        mask = np.zeros((H, W), bool)
        mask[m:H - m, m:W - m] = True
        sE, gE = blob_score(dE)
        sO, gO = blob_score(dO_r)
        pE = [p for p in peaks(sE, gE, mask) if p[0] > floor * 0.5]
        pO = [p for p in peaks(sO, gO, mask) if p[0] > floor * 0.5]
        if pO:
            tree = cKDTree([(p[2], p[3]) for p in pO])
        for p in pE:
            if p[0] <= floor:
                continue
            ok = False
            if pO:
                d, j = tree.query((p[2], p[3]))
                ok = d <= max(2.0, p[1])
                match_rows.append((t, p[0], p[1], p[2], p[3], float(ok), float(d), pO[j][0]))
            else:
                match_rows.append((t, p[0], p[1], p[2], p[3], 0.0, np.inf, 0.0))
        if k == len(pairs) // 2:
            example = (t, s, raw[t].copy(), rO_r.copy(), dE.copy(), dO_r.copy())

    frc_den = np.array(frc_den)
    frc_raw = np.array(frc_raw)
    match_rows = np.array(match_rows)
    dis_map = dis_sum / len(pairs)
    dis_noise_map = dis_noise_sum / len(pairs)
    nvm = float(np.mean(noise_var))
    sigma = np.sqrt(nvm)

    # spectral error, per-pixel variance units (window and pair count divided out)
    norm = len(pairs) * (win ** 2).sum()
    acc = {key: v / norm for key, v in acc.items()}
    smask = interference_mask(acc['B'])
    rings = {}
    for key in spec_keys:
        fk, rings[key] = ring_average(acc[key], smask)
    N_k = rings['noise']
    S_k = rings['B'] - N_k
    E_k = {'SHINE (E model)': rings['err'] - N_k, 'Gaussian blur 3 px': rings['err_blur3'] - N_k,
           '5-frame mean (even half)': rings['err_mean5'] - N_k}
    # relative error only where a single frame carries measurable signal (S > 2% of noise power)
    valid = S_k > 0.02 * N_k
    rel_k = {key: np.where(valid, v / np.where(valid, S_k, 1), np.nan) for key, v in E_k.items()}
    beyond = np.where(~valid & (fk > 0.01))[0]
    signal_limit = float(fk[beyond[0]]) if len(beyond) else 0.5
    frc_AB = rings['AB'] / np.sqrt(rings['A'] * rings['B'])

    def crossing(rel):
        # first frequency below the signal limit where the relative error reaches 1;
        # None = stays below 1 throughout the band with measurable signal
        above = np.where((rel >= 1) & (fk > 0.01) & (fk < signal_limit))[0]
        return float(fk[above[0]]) if len(above) else None

    # reproducibility vs score
    edges = np.unique(np.percentile(match_rows[:, 1], np.linspace(0, 100, 21)))
    idx = np.digitize(match_rows[:, 1], edges[1:-1])
    rep_curve = [(float(match_rows[idx == i, 1].mean()), float(match_rows[idx == i, 5].mean()), int((idx == i).sum()))
                 for i in range(len(edges) - 1) if (idx == i).any()]
    summary = dict(
        pairs_used=len(pairs), floor=floor, noise_var=nvm,
        held_out_error=dict((k, float(np.mean(v))) for k, v in held.items()),
        held_out_error_over_noise_var=dict((k, float(np.mean(v)) / nvm) for k, v in held.items()),
        frc_den_mean_by_band={f'{lo}-{hi}': float(frc_den.mean(0)[(f >= lo) & (f < hi)].mean())
                              for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]},
        frc_raw_mean_by_band={f'{lo}-{hi}': float(frc_raw.mean(0)[(f >= lo) & (f < hi)].mean())
                              for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]},
        detections_above_floor=int(len(match_rows)),
        reproduced_fraction=float(match_rows[:, 5].mean()) if len(match_rows) else None,
        reproduced_fraction_by_score=rep_curve,
        disagreement_over_raw_pair_median=float(np.median(dis_map / dis_noise_map)),
        noise_sigma=sigma,
        e_minus_o_std_over_sigma=float(np.std(example[4][crop] - example[5][crop]) / sigma),
        interference_pixels_masked=int((~smask).sum()),
        signal_measurable_below=signal_limit,
        relative_error_reaches_1_at=dict((key, crossing(v)) for key, v in rel_k.items()),
        relative_error_by_band=dict((key, {f'{lo}-{hi}': float(np.nanmedian(v[(fk >= lo) & (fk < hi)]))
                                           if np.isfinite(v[(fk >= lo) & (fk < hi)]).any() else None
                                           for lo, hi in [(0.01, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2)]})
                                    for key, v in rel_k.items()),
        frc_den_vs_raw_other_by_band={f'{lo}-{hi}': float(frc_AB[(fk >= lo) & (fk < hi)].mean())
                                      for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]},
    )
    for r in rois:
        cd, cr = np.array(roi_frc[r][0]), np.array(roi_frc[r][1])
        summary[f'roi_{r}'] = dict(f=fr.tolist(), frc_den_mean=cd.mean(0).tolist(), frc_raw_mean=cr.mean(0).tolist())
    with open(os.path.join(a.out_dir, 'evenodd_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(os.path.join(a.out_dir, 'evenodd_data.npz'), f=f, frc_den=frc_den, frc_raw=frc_raw,
                        match_rows=match_rows, dis_map=dis_map, dis_noise_map=dis_noise_map,
                        held=np.array([held[k] for k in held]), held_keys=np.array(list(held)),
                        fk=fk, N_k=N_k, S_k=S_k, frc_AB=frc_AB, smask=smask,
                        **{'E_k_' + key.split(' ')[0]: v for key, v in E_k.items()})

    # ---- figure ----
    t, s, rE, rO, dE, dO = example
    fig, ax = plt.subplots(4, 4, figsize=(22, 21))
    lo, hi = np.percentile(dE[crop], [0.5, 99.8])
    for k, (img, title) in enumerate([(rE, f'raw {t} (even)'), (rO, f'raw {s} (odd, registered)'),
                                      (dE, f'E model, frame {t}'), (dO, f'O model, frame {s} (registered)')]):
        ax[0, k].imshow(img[crop], cmap='gray', vmin=lo if k > 1 else None, vmax=hi if k > 1 else None)
        ax[0, k].set_title(title)
        ax[0, k].axis('off')
    zoom = (slice(H // 2 - 128, H // 2 + 128), slice(W // 2 - 128, W // 2 + 128))
    dd = (dE - dO)[zoom]
    ax[1, 0].imshow(np.hstack([dE[zoom], np.full((256, 6), hi), dO[zoom]]), cmap='gray', vmin=lo, vmax=hi)
    ax[1, 0].set_title('centre 256 px: E (left) / O (right)')
    ax[1, 0].axis('off')
    im = ax[1, 1].imshow(dd / sigma, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    cb = plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    cb.set_label('E - O  (units of noise sigma per raw pixel)')
    ax[1, 1].set_title(f'E - O, centre 256 px (noise sigma = {sigma:.0f} raw units)')
    ax[1, 1].axis('off')
    im = ax[1, 2].imshow(dis_map / sigma, cmap='magma', vmin=0, vmax=0.2)
    cb = plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
    cb.set_label('units of noise sigma')
    ax[1, 2].set_title(f'time-mean |E - O| over {len(pairs)} pairs')
    ax[1, 2].axis('off')
    sel = match_rows[match_rows[:, 0] == t]
    ax[1, 3].imshow(dE[crop], cmap='gray', vmin=lo, vmax=hi)
    for r in sel:
        ax[1, 3].add_patch(plt.Circle((r[4] - m, r[3] - m), 2 * r[2] + 2, fill=False,
                                      color='lime' if r[5] else 'r', lw=1))
    ax[1, 3].set_title(f'frame {t} detections > floor: reproduced (green) / not (red)')
    ax[1, 3].axis('off')

    ax[2, 0].plot(fk, frc_AB, label='denoised E vs raw t+1 (accuracy)')
    ax[2, 0].plot(f, frc_den.mean(0), label='denoised E vs denoised O (reproducibility)')
    ax[2, 0].plot(f, frc_raw.mean(0), label='raw t vs raw t+1')
    ax[2, 0].axhline(0, color='k', lw=0.5)
    ax[2, 0].set_xlabel('spatial frequency (cycles/px)')
    ax[2, 0].set_ylabel('FRC')
    ax[2, 0].legend()
    ax[2, 0].set_title('Fourier ring correlation')
    keys = list(held)
    vals = [np.mean(held[k]) / nvm for k in keys]
    ax[2, 1].barh(keys, vals)
    ax[2, 1].set_xlabel('held-out error / noise variance (lower is better)')
    ax[2, 1].set_title('error against the other half (true error + real change)')
    rc = np.array(rep_curve)
    ax[2, 2].plot(rc[:, 0], rc[:, 1], 'o-')
    ax[2, 2].set_xlabel('blob score in E')
    ax[2, 2].set_ylabel('fraction found in O')
    ax[2, 2].set_ylim(0, 1.02)
    ax[2, 2].axvline(floor, color='r', lw=0.8, label='null floor')
    ax[2, 2].legend()
    ax[2, 2].set_title('blob reproducibility vs score')
    ax[2, 3].hist(match_rows[match_rows[:, 5] == 1, 1], 50, alpha=0.6, label='reproduced')
    ax[2, 3].hist(match_rows[match_rows[:, 5] == 0, 1], 50, alpha=0.6, label='not reproduced')
    ax[2, 3].set_yscale('log')
    ax[2, 3].set_xlabel('blob score')
    ax[2, 3].legend()
    ax[2, 3].set_title('detections above the null floor')
    ax[3, 0].semilogy(fk, S_k, label='signal S(k) = raw power - noise')
    ax[3, 0].semilogy(fk, N_k, label='noise N(k) (frame differences)')
    for key, c in zip(E_k, ('C3', 'C4', 'C5')):
        ax[3, 0].semilogy(fk, np.clip(E_k[key], 1e-3 * N_k.min(), None), color=c, label=f'error E(k), {key}')
    ax[3, 0].set_xlabel('spatial frequency (cycles/px)')
    ax[3, 0].set_ylabel('power per pixel (raw units^2)')
    ax[3, 0].legend(fontsize=8)
    ax[3, 0].set_title('spectral error vs signal and noise')
    for key, c in zip(rel_k, ('C3', 'C4', 'C5')):
        ax[3, 1].semilogy(fk, np.clip(rel_k[key], 1e-3, None), color=c, label=key)
    ax[3, 1].axhline(1, color='k', lw=0.8)
    ax[3, 1].axvline(signal_limit, color='gray', ls='--', lw=0.8)
    ax[3, 1].text(signal_limit, 2e2, ' no measurable signal ->', fontsize=8)
    ax[3, 1].set_ylim(1e-3, 1e3)
    ax[3, 1].set_xlabel('spatial frequency (cycles/px)')
    ax[3, 1].set_ylabel('E(k) / S(k)')
    ax[3, 1].legend(fontsize=8)
    ax[3, 1].set_title('relative error (<1: frequency reproduced correctly)')
    k0 = 160
    sl2 = (slice(ch // 2 - k0, ch // 2 + k0), slice(cw // 2 - k0, cw // 2 + k0))
    ax[3, 2].imshow(np.log10(acc['B'][sl2]), cmap='magma',
                    extent=[-k0 / cw, k0 / cw, k0 / ch, -k0 / ch])
    yy, xx = np.nonzero(~smask[sl2])
    ax[3, 2].scatter((xx - k0) / cw, (yy - k0) / ch, s=2, c='c')
    ax[3, 2].set_title('raw power spectrum, masked interference (cyan)')
    ax[3, 2].set_xlabel('fx')
    ax[3, 2].set_ylabel('fy')
    ax[3, 3].axis('off')
    txt = [f'single-frame signal measurable below {signal_limit:.3f} cycles/px', '',
           'relative error reaches 1 at (cycles/px):'] + [
        f'  {key}: ' + ('stays < 1' if crossing(v) is None else f'{crossing(v):.3f}') for key, v in rel_k.items()]
    txt += ['', f'held-out error / noise variance:'] + [f'  {key}: {np.mean(v) / nvm:.4f}' for key, v in held.items()]
    ax[3, 3].text(0, 1, '\n'.join(txt), va='top', family='monospace', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, 'evenodd_test.png'), dpi=70)
    for r in rois:
        ex = roi_examples[r]
        cd, cr = np.array(roi_frc[r][0]), np.array(roi_frc[r][1])
        fig, ax = plt.subplots(len(ex), 5, figsize=(25, 5 * len(ex)), squeeze=False)
        for i, (t, rr, e, o) in enumerate(ex):
            lo, hi = np.percentile(e, [0.5, 99.8])
            ps = lambda x: np.log10(np.abs(np.fft.fftshift(np.fft.fft2((x - x.mean()) * np.outer(
                np.hanning(x.shape[0]), np.hanning(x.shape[1]))))) ** 2 + 1)
            for j, (img, title, kw) in enumerate([
                    (rr, f'raw {t}', {}), (e, f'E model {t}', dict(vmin=lo, vmax=hi)),
                    (o, f'O model {t + 1} (registered)', dict(vmin=lo, vmax=hi))]):
                ax[i, j].imshow(img, cmap='gray', **kw)
                ax[i, j].set_title(title)
                ax[i, j].axis('off')
            pe, po = ps(e), ps(o)
            v0, v1 = np.percentile(pe, [50, 99.9])
            ax[i, 3].imshow(np.hstack([pe, np.full((pe.shape[0], 4), v1), po]), cmap='magma', vmin=v0, vmax=v1)
            ax[i, 3].set_title('power spectrum E (left) / O (right)')
            ax[i, 3].axis('off')
            if i == 0:
                ax[i, 4].plot(fr, cd.mean(0), label='denoised E vs O')
                ax[i, 4].fill_between(fr, np.percentile(cd, 5, 0), np.percentile(cd, 95, 0), alpha=0.2)
                ax[i, 4].plot(fr, cr.mean(0), label='raw t vs t+1')
                ax[i, 4].axhline(0, color='k', lw=0.5)
                ax[i, 4].set_xlabel('spatial frequency (cycles/px)')
                ax[i, 4].set_ylabel('FRC in region')
                ax[i, 4].legend()
                ax[i, 4].set_title(f'region {r}')
            else:
                ax[i, 4].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(a.out_dir, 'evenodd_roi_{}_{}_{}_{}.png'.format(*r)), dpi=70)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
