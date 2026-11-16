"""Tier 2, test 1: is raw - denoised pure noise?

The residual of every frame is compared with an independent noise estimate taken from
consecutive raw frames: n_t = (raw_t - shift(raw_t+1)) / sqrt(2), where the shift (from
phase correlation of the denoised frames) is applied in Fourier space so the noise power
spectrum is untouched. If the denoiser removed only noise, the residual matches n_t in
level-dependent variance, autocorrelation and power spectrum, averages to zero everywhere,
and is uncorrelated with the denoised image.

The production output is min/max rescaled to the raw frame, so its intensity scale is not
the raw scale; each frame gets a 2-parameter affine fit (raw ~ a*den + b) before the residual
is taken. A global affine map cannot create or remove structure.

Usage: python residual_test.py RAW_DIR DENOISED_DIR OUT_DIR [--margin 32]
"""
import argparse
import json
import os
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import list_tifs, read_stack, affine_fit


def fourier_shift(img, dy, dx):
    h, w = img.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (ky * dy + kx * dx))))


def register(a, b):
    """Shift (dy, dx) that moves b onto a."""
    win = cv.createHanningWindow(a.shape[::-1], cv.CV_32F)
    (dx, dy), resp = cv.phaseCorrelate(b.astype(np.float32), a.astype(np.float32), win)
    return dy, dx, resp


def autocorr(x, n=4):
    x = x - x.mean()
    f = np.fft.fft2(x)
    ac = np.real(np.fft.ifft2(np.abs(f) ** 2))
    ac /= ac[0, 0]
    return np.fft.fftshift(ac)[ac.shape[0] // 2 - n:ac.shape[0] // 2 + n + 1,
                               ac.shape[1] // 2 - n:ac.shape[1] // 2 + n + 1]


def psd2(x):
    x = x - x.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    return np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2


def radial(p, nbins=64):
    h, w = p.shape
    y, x = np.indices(p.shape)
    r = np.hypot((y - h // 2) / h, (x - w // 2) / w)  # cycles/px
    bins = np.linspace(0, 0.5, nbins + 1)
    i = np.digitize(r.ravel(), bins) - 1
    ok = (i >= 0) & (i < nbins)
    s = np.bincount(i[ok], p.ravel()[ok], nbins) / np.maximum(np.bincount(i[ok], minlength=nbins), 1)
    return 0.5 * (bins[1:] + bins[:-1]), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir')
    ap.add_argument('den_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--margin', type=int, default=32, help='border excluded (Fourier-shift wrap)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    names = list_tifs(a.raw_dir)
    raw, _ = read_stack(a.raw_dir, names)
    den, _ = read_stack(a.den_dir, names)
    T, H, W = raw.shape
    m = a.margin
    crop = (slice(m, H - m), slice(m, W - m))

    # per-frame affine fit raw ~ a*den + b
    coef = np.array([affine_fit(den[t][crop], raw[t][crop]) for t in range(T)])
    fit = coef[:, 0, None, None] * den + coef[:, 1, None, None]
    res = (raw - fit)[:, crop[0], crop[1]]
    level = fit[:, crop[0], crop[1]]

    # independent noise from consecutive frames
    shifts, noise, nlevel = [], [], []
    for t in range(T - 1):
        dy, dx, resp = register(den[t], den[t + 1])
        shifts.append((dy, dx, resp))
        if abs(dy) > m / 2 or abs(dx) > m / 2:
            continue
        n = (raw[t] - fourier_shift(raw[t + 1], dy, dx)) / np.sqrt(2)
        noise.append(n[crop])
        nlevel.append(fit[t][crop])
    shifts = np.array(shifts)
    noise = np.stack(noise)
    nlevel = np.stack(nlevel)

    # 1. variance vs signal level
    edges = np.percentile(level, np.linspace(0.5, 99.9, 30))
    def var_by_level(x, lv):
        i = np.digitize(lv.ravel(), edges)
        xs = x.ravel()
        out = []
        for k in range(1, len(edges)):
            sel = xs[i == k]
            out.append(sel.var() if sel.size > 1000 else np.nan)
        return np.array(out)
    centers = 0.5 * (edges[1:] + edges[:-1])
    v_res = var_by_level(res, level)
    v_noise = var_by_level(noise, nlevel)

    # 2. temporal-mean residual map (should be zero everywhere)
    mean_res = res.mean(0)
    sem_expected = np.sqrt(noise.var(0).mean() / T)
    # same statistic for pure noise; non-overlapping pairs so the differences do not telescope
    mean_noise = noise[::2].mean(0) * np.sqrt(len(noise[::2]) / T)

    # 3. autocorrelation, 4. power spectra
    sub = list(range(0, T, max(1, T // 40)))
    ac_res = np.mean([autocorr(res[t]) for t in sub], 0)
    ac_noise = np.mean([autocorr(noise[min(t, len(noise) - 1)]) for t in sub], 0)
    p_res = np.mean([psd2(res[t]) for t in sub], 0)
    p_noise = np.mean([psd2(noise[min(t, len(noise) - 1)]) for t in sub], 0)
    p_den = np.mean([psd2(fit[t][crop]) for t in sub], 0)
    p_raw = np.mean([psd2(raw[t][crop]) for t in sub], 0)
    f, r_res = radial(p_res)
    _, r_noise = radial(p_noise)
    _, r_den = radial(p_den)
    _, r_raw = radial(p_raw)

    # 5. residual-vs-denoised correlation, per frame and in local 32 px windows
    cc_frame = np.array([np.corrcoef(res[t].ravel(), level[t].ravel())[0, 1] for t in range(T)])
    cc_noise = np.array([np.corrcoef(noise[t].ravel(), nlevel[t].ravel())[0, 1] for t in range(len(noise))])
    b = 32
    hh, ww = (res.shape[1] // b) * b, (res.shape[2] // b) * b
    def blocks(x):
        return x[:, :hh, :ww].reshape(x.shape[0], hh // b, b, ww // b, b)
    rb, lb = blocks(res), blocks(level)
    rb = rb - rb.mean((2, 4), keepdims=True)
    lb = lb - lb.mean((2, 4), keepdims=True)
    cc_local = ((rb * lb).sum((0, 2, 4)) /
                np.sqrt((rb ** 2).sum((0, 2, 4)) * (lb ** 2).sum((0, 2, 4)) + 1e-12))

    # 6. intensity bias vs level and where it sits (time-mean residual in 16 px blocks; the
    #    time-mean raw frame is a near noise-free reference for the static parts of the scene)
    i = np.digitize(level.ravel(), edges)
    bias = np.array([res.ravel()[i == k].mean() for k in range(1, len(edges))])
    bb = 16
    h2, w2 = (mean_res.shape[0] // bb) * bb, (mean_res.shape[1] // bb) * bb
    # block s.e. from the noise reference blocks themselves (includes the scan-direction correlation)
    nb = noise[::2, :h2, :w2].reshape(-1, h2 // bb, bb, w2 // bb, bb).mean((2, 4))
    block_se = nb.std(0).mean() / np.sqrt(T)
    block_z = mean_res[:h2, :w2].reshape(h2 // bb, bb, w2 // bb, bb).mean((1, 3)) / block_se
    lowpass_diff = cv.GaussianBlur(raw.mean(0)[crop], (0, 0), 4) - cv.GaussianBlur(fit.mean(0)[crop], (0, 0), 4)
    tm = cv.GaussianBlur(raw.mean(0)[crop], (0, 0), 4)
    particle_contrast = float(np.percentile(tm, 99.5) - np.median(tm))

    # 7. sharp spectral peaks (periodic scan interference): power kept in the denoised output
    N0, N1 = p_raw.shape
    q = p_raw / np.median(p_raw)
    q[N0 // 2 - 5:N0 // 2 + 6, N1 // 2 - 5:N1 // 2 + 6] = 0
    peaks = np.argwhere(q > 30)
    peak_list = [dict(fy=round((y - N0 // 2) / N0, 4), fx=round((x - N1 // 2) / N1, 4),
                      raw_rel=round(float(q[y, x]), 1), den_over_raw=round(float(p_den[y, x] / p_raw[y, x]), 3))
                 for y, x in peaks if (y, x) >= (N0 // 2, N1 // 2)]
    # peaks present in the denoised output but not in raw
    qd = p_den / np.maximum(p_raw, 1e-12)
    qd[N0 // 2 - 5:N0 // 2 + 6, N1 // 2 - 5:N1 // 2 + 6] = 0

    summary = dict(
        frames=int(T), affine_a_median=float(np.median(coef[:, 0])), affine_b_median=float(np.median(coef[:, 1])),
        drift_px_median=float(np.median(np.hypot(shifts[:, 0], shifts[:, 1]))),
        noise_pairs_used=int(len(noise)),
        var_ratio_res_over_noise_median=float(np.nanmedian(v_res / v_noise)),
        var_ratio_res_over_noise_min=float(np.nanmin(v_res / v_noise)),
        var_ratio_res_over_noise_max=float(np.nanmax(v_res / v_noise)),
        mean_residual_map_std=float(mean_res.std()), mean_residual_expected_std=float(sem_expected),
        mean_noise_map_std=float(mean_noise.std()),
        mean_residual_map_absmax_sigma=float(np.abs(mean_res).max() / sem_expected),
        acf_res_dx1=float(ac_res[4, 5]), acf_noise_dx1=float(ac_noise[4, 5]),
        acf_res_dy1=float(ac_res[5, 4]), acf_noise_dy1=float(ac_noise[5, 4]),
        psd_ratio_low_f_lt_0p05=float(np.mean(r_res[f < 0.05] / r_noise[f < 0.05])),
        psd_ratio_mid_f=float(np.mean(r_res[(f >= 0.05) & (f < 0.25)] / r_noise[(f >= 0.05) & (f < 0.25)])),
        psd_ratio_high_f_ge_0p25=float(np.mean(r_res[f >= 0.25] / r_noise[f >= 0.25])),
        corr_res_vs_den_frame_median=float(np.median(cc_frame)),
        corr_noise_vs_den_frame_median=float(np.median(cc_noise)),
        corr_res_vs_den_local_absmax=float(np.abs(cc_local).max()),
        bias_vs_level_min=float(np.nanmin(bias)), bias_vs_level_max=float(np.nanmax(bias)),
        block16_z_absmax=float(np.abs(block_z).max()), block16_frac_absz_gt4=float((np.abs(block_z) > 4).mean()),
        timemean_lowpass_diff_absmax=float(np.abs(lowpass_diff).max()),
        timemean_lowpass_diff_p99=float(np.percentile(np.abs(lowpass_diff), 99)),
        particle_contrast_timemean=particle_contrast,
        sharp_peaks=peak_list,
        max_den_over_raw_power_ratio_outside_dc=float(qd.max()),
    )
    with open(os.path.join(a.out_dir, 'residual_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(os.path.join(a.out_dir, 'residual_data.npz'), centers=centers, v_res=v_res, v_noise=v_noise,
                        mean_res=mean_res, mean_noise=mean_noise, ac_res=ac_res, ac_noise=ac_noise, f=f,
                        r_res=r_res, r_noise=r_noise, r_den=r_den, r_raw=r_raw, cc_frame=cc_frame,
                        cc_local=cc_local, shifts=shifts, coef=coef, bias=bias, block_z=block_z,
                        lowpass_diff=lowpass_diff)

    # ---- figure ----
    t0 = T // 2
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    vr = np.percentile(np.abs(res[t0]), 99)
    for k, (img, title, kw) in enumerate([
            (raw[t0][crop], f'raw frame {t0}', dict(cmap='gray')),
            (fit[t0][crop], 'denoised (affine-fitted to raw)', dict(cmap='gray')),
            (res[t0], 'residual = raw - denoised', dict(cmap='gray', vmin=-vr, vmax=vr)),
            (noise[min(t0, len(noise) - 1)], 'noise reference (raw t - raw t+1)/sqrt2', dict(cmap='gray', vmin=-vr, vmax=vr))]):
        ax[0, k].imshow(img, **kw)
        ax[0, k].set_title(title)
        ax[0, k].axis('off')
    lim = 4 * sem_expected
    im = ax[1, 0].imshow(mean_res, cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax[1, 0].set_title(f'residual averaged over {T} frames\n(colour range +/-4 x expected noise s.e.)')
    ax[1, 0].axis('off')
    plt.colorbar(im, ax=ax[1, 0], fraction=0.046)
    im = ax[1, 1].imshow(mean_noise, cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax[1, 1].set_title('noise reference averaged (same scale)')
    ax[1, 1].axis('off')
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    im = ax[1, 2].imshow(cc_local, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax[1, 2].set_title('local corr(residual, denoised), 32 px blocks')
    ax[1, 2].axis('off')
    plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
    ax[1, 3].plot(cc_frame, label='residual vs denoised')
    ax[1, 3].plot(cc_noise, label='noise ref vs denoised', alpha=0.7)
    ax[1, 3].axhline(0, color='k', lw=0.5)
    ax[1, 3].set_xlabel('frame')
    ax[1, 3].set_ylabel('Pearson r')
    ax[1, 3].legend()
    ax[1, 3].set_title('whole-frame correlation with denoised')

    ax[2, 0].plot(centers, v_noise, 'o-', label='noise reference')
    ax[2, 0].plot(centers, v_res, 's-', label='residual')
    ax[2, 0].set_xlabel('signal level (denoised, raw units)')
    ax[2, 0].set_ylabel('variance')
    ax[2, 0].legend()
    ax[2, 0].set_title('variance vs signal level')
    ax[2, 1].plot(centers, v_res / v_noise, 'o-')
    ax[2, 1].axhline(1, color='k', lw=0.5)
    ax[2, 1].set_xlabel('signal level')
    ax[2, 1].set_ylabel('residual / noise variance')
    ax[2, 1].set_title('>1: signal removed, <1: noise kept')
    ax[2, 2].semilogy(f, r_raw, label='raw')
    ax[2, 2].semilogy(f, r_den, label='denoised')
    ax[2, 2].semilogy(f, r_noise, label='noise reference')
    ax[2, 2].semilogy(f, r_res, '--', label='residual')
    ax[2, 2].set_xlabel('spatial frequency (cycles/px)')
    ax[2, 2].set_ylabel('power')
    ax[2, 2].legend()
    ax[2, 2].set_title('radial power spectra')
    ax[2, 3].plot(f, r_res / r_noise)
    ax[2, 3].axhline(1, color='k', lw=0.5)
    ax[2, 3].set_xlabel('spatial frequency (cycles/px)')
    ax[2, 3].set_ylabel('residual / noise power')
    ax[2, 3].set_title('spectral ratio (1 = residual is noise)')
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, 'residual_test.png'), dpi=70)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for k, (ac, title) in enumerate([(ac_res, 'residual'), (ac_noise, 'noise reference')]):
        ac = ac.copy()
        ac[4, 4] = np.nan
        im = ax[k].imshow(ac, cmap='RdBu_r', vmin=-0.15, vmax=0.15, extent=[-4.5, 4.5, 4.5, -4.5])
        ax[k].set_title(f'{title} autocorrelation (zero lag hidden)')
        ax[k].set_xlabel('dx (px, fast scan)')
        ax[k].set_ylabel('dy (px)')
        for (i, j), v in np.ndenumerate(ac):
            if not np.isnan(v) and abs(i - 4) <= 2 and abs(j - 4) <= 2:
                ax[k].text(j - 4, i - 4, f'{v:.3f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.savefig(os.path.join(a.out_dir, 'residual_autocorr.png'), dpi=80, bbox_inches='tight')
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    ax[0, 0].plot(centers, bias, 'o-')
    ax[0, 0].axhline(0, color='k', lw=0.5)
    ax[0, 0].set_xlabel('signal level (denoised, raw units)')
    ax[0, 0].set_ylabel('mean residual')
    ax[0, 0].set_title(f'intensity bias vs level (particle contrast ~{particle_contrast:.0f})')
    im = ax[0, 1].imshow(block_z, cmap='RdBu_r', vmin=-10, vmax=10)
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title('time-mean residual, 16 px blocks (z vs noise reference)')
    ax[0, 1].axis('off')
    im = ax[0, 2].imshow(lowpass_diff, cmap='RdBu_r', vmin=-150, vmax=150)
    plt.colorbar(im, ax=ax[0, 2])
    ax[0, 2].set_title('time-mean raw - time-mean denoised, blurred 4 px')
    ax[0, 2].axis('off')
    k0 = 160
    sl = (slice(N0 // 2 - k0, N0 // 2 + k0), slice(N1 // 2 - k0, N1 // 2 + k0))
    ext = [-k0 / N1, k0 / N1, k0 / N0, -k0 / N0]
    for k, (P, title) in enumerate([(p_raw, 'raw'), (p_den, 'denoised'), (p_res, 'residual')]):
        ax[1, k].imshow(np.log10(P[sl]), cmap='magma', extent=ext)
        ax[1, k].set_title(f'2D power spectrum, {title}')
        ax[1, k].set_xlabel('fx (cycles/px, fast scan)')
        ax[1, k].set_ylabel('fy')
    plt.tight_layout()
    plt.savefig(os.path.join(a.out_dir, 'residual_diagnostics.png'), dpi=70)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
