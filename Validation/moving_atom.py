"""Tier 3, realistic motion: a faint atom-like blob that is present in every frame and moves.

The panel-C blob (Gaussian sigma 5 px, amplitude = blob_mult x detection threshold) is added to the
RAW frames t0-5 ... t0+5 on flat background (real noise stays). Outputs t0-3 ... t0+3 are
measured; their whole 5-frame input lies inside the motion, so every measured frame sees the same
situation. Conditions:

  static      - blob at the same place in all frames (baseline: what a persistent atom gives)
  drift_x/y   - blob moves `speed` px/frame along x (or y); true position x(t) = x0 + v (t - t0)
  hop_D       - blob at site A before t0 and at site B = A + D px (x) from t0 on, and stays

Measurements on the output change d = edited - unedited (noise cancels; exact for that model):
  drift: least-squares split of d onto blobs at the true positions of frames t+k (k = -2..2).
         Ideal: all weight at k = 0. Weight at k != 0 = the atom shown where it was / will be.
         Also the centroid error along the motion and the length/width ratio of d.
  hop:   split of d onto blobs at A and B. Ideal: A = 1, B = 0 before t0; A = 0, B = 1 from t0.

Usage: python moving_atom.py RAW DEN OUT_DIR --null_summary J --model_spec "LABEL=CKPT[:FRAME_NUM]" ...
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from temporal_test import Stack, Model, background_sites, frames_t0, gauss, lsq

HALF_PRESENT, HALF_MEASURE = 5, 3
K = (-2, -1, 0, 1, 2)


def trajectory(cond, t, t0, y, x, speed):
    """True blob centre in raw frame t."""
    if cond == 'static':
        return y, x
    if cond == 'drift_x':
        return y, x + speed * (t - t0)
    if cond == 'drift_y':
        return y + speed * (t - t0), x
    D = int(cond.split('_')[1])                   # hop_D: centred on the site
    return (y, x - D / 2) if t < t0 else (y, x + D / 2)


def edits_for(cond, t0, sites, A, s, shape, speed):
    ed = {}
    for t in range(t0 - HALF_PRESENT, t0 + HALF_PRESENT + 1):
        img = np.zeros(shape, np.float32)
        for (y, x) in sites:
            cy, cx = trajectory(cond, t, t0, y, x, speed)
            img += A * gauss(shape, cy, cx, s)
        ed[t] = img
    return ed


def moments(d, cy, cx, R, along):
    """Centroid offset along the motion and length/width ratio of the positive part of d."""
    sl = (slice(int(cy) - R, int(cy) + R + 1), slice(int(cx) - R, int(cx) + R + 1))
    w = np.clip(d[sl], 0, None)
    w = np.where(w > 0.2 * w.max(), w, 0)
    yy, xx = np.indices(w.shape, dtype=float)
    yy += sl[0].start
    xx += sl[1].start
    m = w.sum()
    my, mx = (w * yy).sum() / m, (w * xx).sum() / m
    vy, vx = (w * (yy - my) ** 2).sum() / m, (w * (xx - mx) ** 2).sum() / m
    if along == 'x':
        return mx - cx, np.sqrt(vx / vy)
    return my - cy, np.sqrt(vy / vx)


def run_model(model, st, sites, t0s, A, s, speed, hops):
    shape = (st.H, st.W)
    R = int(3 * s + 2 * speed + 4)
    conds = ['static', 'drift_x', 'drift_y'] + [f'hop_{D}' for D in hops]
    rows, example = [], {}
    for cond in conds:
        for t0 in t0s:
            ed = edits_for(cond, t0, sites, A, s, shape, speed)
            for m in range(-HALF_MEASURE, HALF_MEASURE + 1):
                t = t0 + m
                d = model.edited(t, ed) - model.unedited(t)
                if t0 == t0s[len(t0s) // 2]:
                    y, x = sites[0]
                    example.setdefault(cond, {})[m] = dict(
                        d=d[y - 64:y + 65, x - 64:x + 65],
                        raw=(st.raw[t] + ed[t])[y - 64:y + 65, x - 64:x + 65])
                for i, (y, x) in enumerate(sites):
                    sl = (slice(y - R - 12, y + R + 13), slice(x - R - 12, x + R + 13))
                    row = dict(cond=cond, t0=int(t0), m=m, site=i)
                    if cond == 'static' or cond.startswith('drift'):
                        cols = [gauss(shape, *trajectory(cond, t + k, t0, y, x, speed), s)[sl] for k in K]
                        if cond == 'static':
                            row['amp'] = float(lsq(d[sl], [cols[2]])[0] / A)
                        else:
                            w = lsq(d[sl], cols) / A
                            row.update({f'w{k:+d}': float(v) for k, v in zip(K, w)})
                            cy, cx = trajectory(cond, t, t0, y, x, speed)
                            off, el = moments(d, cy, cx, R, cond[-1])
                            row.update(offset=float(off), elong=float(el))
                        if cond == 'static':
                            off, el = moments(d, y, x, R, 'x')
                            row.update(elong=float(el))
                    else:
                        D = int(cond.split('_')[1])
                        gA, gB = gauss(shape, y, x - D / 2, s)[sl], gauss(shape, y, x + D / 2, s)[sl]
                        a, b = lsq(d[sl], [gA, gB]) / A
                        row.update(A_site=float(a), B_site=float(b))
                    rows.append(row)
        print(f'  {cond} done', flush=True)
    return rows, example


def summarise(rows, hops):
    def mean(cond, key, **kw):
        v = [r[key] for r in rows if r['cond'] == cond and all(r[a] == b for a, b in kw.items())]
        return float(np.mean(v))

    out = dict(static=dict(amp=mean('static', 'amp'), elong=mean('static', 'elong')))
    dr = [r for r in rows if r['cond'].startswith('drift')]
    out['drift'] = {f'w{k:+d}': float(np.mean([r[f'w{k:+d}'] for r in dr])) for k in K}
    out['drift'].update(offset=float(np.mean([r['offset'] for r in dr])),
                        abs_offset=float(np.mean([abs(r['offset']) for r in dr])),
                        elong=float(np.mean([r['elong'] for r in dr])),
                        total=float(np.mean([sum(r[f'w{k:+d}'] for k in K) for r in dr])))
    out['hop'] = {str(D): {str(m): dict(A=mean(f'hop_{D}', 'A_site', m=m), B=mean(f'hop_{D}', 'B_site', m=m))
                           for m in range(-HALF_MEASURE, HALF_MEASURE + 1)} for D in hops}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir'); ap.add_argument('den_dir'); ap.add_argument('out_dir')
    ap.add_argument('--null_summary', required=True, help='production null summary: sets the threshold')
    ap.add_argument('--model_spec', action='append', required=True, help='LABEL=CKPT[:FRAME_NUM]')
    ap.add_argument('--blob_mult', type=float, default=4)
    ap.add_argument('--sigma', type=float, default=5)
    ap.add_argument('--speed', type=float, default=8)
    ap.add_argument('--hops', type=int, nargs='+', default=[8, 20])
    ap.add_argument('--n_t0', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    floor = json.load(open(a.null_summary))['null_floor_score']
    A = 2 * floor * a.blob_mult                     # score ~ amplitude / 2, as in temporal_test
    st = Stack(a.raw_dir, a.den_dir)
    sites = background_sites(st)
    t0s = [int(t) for t in frames_t0(st, a.n_t0)]
    res = dict(meta=dict(floor=floor, A=A, sigma=a.sigma, speed=a.speed, hops=a.hops, t0s=t0s, sites=sites,
                         present=f't0-{HALF_PRESENT}..t0+{HALF_PRESENT}',
                         measured=f't0-{HALF_MEASURE}..t0+{HALF_MEASURE}'), models={})
    examples = {}
    for spec in a.model_spec:
        lab, ck = spec.split('=', 1)
        ck, fn = (ck.rsplit(':', 1) if ck.rsplit(':', 1)[-1].isdigit() else (ck, '5'))
        print(lab, flush=True)
        model = Model(ck, st, frame_num=int(fn))
        rows, ex = run_model(model, st, sites, t0s, A, a.sigma, a.speed, a.hops)
        res['models'][lab] = summarise(rows, a.hops)
        examples[lab] = ex
        with open(os.path.join(a.out_dir, f'rows_{len(res["models"])}.json'), 'w') as fh:
            json.dump(dict(label=lab, rows=rows), fh)
        del model
    with open(os.path.join(a.out_dir, 'moving_atom_summary.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    np.save(os.path.join(a.out_dir, 'examples.npy'), examples, allow_pickle=True)
    print(json.dumps(res['models'], indent=1))


if __name__ == '__main__':
    main()
