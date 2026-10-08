"""
Export a fully corrected 4D Camera stack (*_gaincorr.h5 from process_scan.py) to a
SHINE-ready TIF folder. No further correction is applied - only defective (zero)
frames are skipped, so they are not used for training or as temporal neighbors.

Usage:
  python3 Utils/gaincorr_h5_to_tif.py --h5 /pscratch/.../MgCl2_corr/data_scan0000000035_gaincorr.h5 \
                                      --out_dir /pscratch/.../MgCl2_tif/
  -> /pscratch/.../MgCl2_tif/data_scan0000000035/data_scan_000000frame.tif ...
"""
import argparse
import os
import re

import cv2 as cv
import h5py
import numpy as np
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True, nargs='+')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--start', type=int, default=0, help='first frame to export')
    p.add_argument('--stop', type=int, default=None, help='last frame (exclusive)')
    p.add_argument('--step', type=int, default=1, help='export every n-th frame')
    args = p.parse_args()

    for fn in args.h5:
        m = re.search(r'scan(\d+)', os.path.basename(fn))
        scan = int(m.group(1)) if m else 0
        outpath = os.path.join(args.out_dir, f'data_scan{scan:010}')
        os.makedirs(outpath, exist_ok=True)
        with h5py.File(fn, 'r') as f:
            dset = f['frames']
            defects = set(f['defect_list'][()].tolist()) if 'defect_list' in f else set()
            stop = dset.shape[0] if args.stop is None else min(args.stop, dset.shape[0])
            n_written, skipped = 0, []
            for ii in tqdm(range(args.start, stop, args.step), desc=os.path.basename(fn)):
                if ii in defects:
                    skipped.append(ii)
                    continue
                im = dset[ii]
                if not np.any(im) or not np.isfinite(im).all():   # safety net
                    skipped.append(ii)
                    continue
                cv.imwrite(os.path.join(outpath, f'data_scan_{ii:06d}frame.tif'), im.astype(np.float32))
                n_written += 1
        print(f'{fn}: wrote {n_written} TIFs to {outpath}, skipped {len(skipped)} defective frames')


if __name__ == '__main__':
    main()
