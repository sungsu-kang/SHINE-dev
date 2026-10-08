#!/usr/bin/env python
"""
Batch version of h5_gain_fix.ipynb.

Pipeline (matches the notebook, minus visualization and the TIF-export option):
  1. Load raw frames + first-pass row/column dark correction
  2. Defect-frame detection -> save defect list as .txt
  3. Estimate dark/gain reference from non-defective frames
  4. Full per-frame gain/defect correction, in parallel, streamed into
     a single output HDF5 file (Option 2 from the notebook)

Usage:
    python process_scan.py --file_path /pscratch/sd/s/skang2/4dcam/ \
                            --scanMater MgCl2 --scanNum 16

    # or process many scanNums for the same material in one process:
    python process_scan.py --file_path /pscratch/sd/s/skang2/4dcam/ \
                            --scanMater MgCl2 --scanNum 12 13 14 15 16
"""

import argparse
import multiprocessing
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def load_and_dark_correct(h5_path):
    """Cell 2: load raw frames, apply first-pass row/column dark correction."""
    with h5py.File(h5_path, "r") as f0:
        sz, sx, sy = f0["frames"].shape
        print(f0["frames"].shape)
        dd_refined = f0["frames"][:sz, :, :]

    dd_rowdark = np.zeros(dd_refined.shape, dtype=np.float32)
    for ii, im in tqdm(enumerate(dd_refined[:, :, :]), total=sz, desc="dark-correct"):
        im = im.astype(np.float32)

        # column dark
        linestat = np.zeros_like(im)
        linestat[: 576 // 2, :] = np.median(im[: 576 // 2, :], axis=-2, keepdims=True)
        linestat[576 // 2 :, :] = np.median(im[576 // 2 :, :], axis=-2, keepdims=True)
        linestat = linestat / np.mean(linestat, axis=(0, 1), keepdims=True)
        im /= linestat

        # row dark
        linestat = np.zeros_like(im)
        linestat[:, : 576 // 2] = np.median(im[:, : 576 // 2], axis=-1, keepdims=True)
        linestat[:, 576 // 2 :] = np.median(im[:, 576 // 2 :], axis=-1, keepdims=True)
        linestat = linestat / np.median(linestat, axis=(0, 1), keepdims=True)
        im /= linestat

        no_zeros = im[np.where(im == 0)]
        zero_ratio = len(no_zeros) / len(im)
        if zero_ratio > 0.2:
            continue
        if ~np.all(~np.isnan(im)):
            continue

        dd_rowdark[ii, :, :] = im

    del dd_refined  # free the raw stack; nothing downstream needs it
    return dd_rowdark, sz


def detect_defects(data, sz, file_path, scanMater, scanNum):
    """Cell 4: defect-frame detection + save list."""
    defect_list = []
    for i in range(0, sz):
        if np.mean(data[i, :, :]) == 0:
            defect_list.append(i)

    listpath = file_path / f"{scanMater}_corr"
    os.makedirs(listpath, exist_ok=True)
    print(listpath)

    print(f"Total number of frames with defect: {len(defect_list)}")
    if len(defect_list) > 0:
        txt_name = "data_scan_{:010}_defect_list.txt".format(scanNum)
        np.savetxt(listpath / txt_name, defect_list, fmt="%d")

    return defect_list


def estimate_dark_gain(data, defect_list):
    """Cell 6: estimate dark and gain reference from non-defective frames."""
    mask = np.ones(data.shape[0], dtype=bool)
    mask[defect_list] = False
    nonzero_array = data[mask]
    print(nonzero_array.shape)

    estimated_dark = np.min(nonzero_array[:, :, :], axis=0)
    estimated_gain = np.mean(nonzero_array[:, :, :], axis=0) - estimated_dark
    return estimated_dark, estimated_gain


## Cell 10: per-frame gain/defect correction, run in worker processes.
##
## multiprocessing.Pool must pickle whatever callable it dispatches to
## workers. A closure (a function defined inside another function) is NOT
## picklable, so these have to be plain module-level globals + a module-level
## function, set up once per worker via the Pool's `initializer`.
_worker_data = None
_worker_defect_set = None
_worker_dark = None
_worker_gain = None


def _init_worker(data, defect_set, estimated_dark, estimated_gain):
    """Runs once in each worker process; stashes shared arrays as globals
    so process_frame_h5 doesn't need to be (and can't be) pickled per task."""
    global _worker_data, _worker_defect_set, _worker_dark, _worker_gain
    _worker_data = data
    _worker_defect_set = defect_set
    _worker_dark = estimated_dark
    _worker_gain = estimated_gain


def process_frame_h5(ii):
    data = _worker_data
    defect_list = _worker_defect_set
    estimated_dark = _worker_dark
    estimated_gain = _worker_gain

    if ii in defect_list:
        return ii, np.zeros((576, 576), dtype=np.float32)

    im_d = data[ii, :, :].astype(np.float32)
    im_d = (im_d - estimated_dark) / estimated_gain

    im_d[183:186, 288:] = np.nan
    im_d[387:390, 288:] = np.nan

    avg = np.nanmean(im_d)
    w, h = np.shape(im_d)

    linestat = np.zeros_like(im_d)
    linestat[:, h // 2 :] = np.nanmedian(im_d[:, h // 2 :], axis=-2, keepdims=True)
    linestat[:, : h // 2] = np.nanmedian(im_d[:, : h // 2], axis=-2, keepdims=True)
    im_d /= linestat / avg

    linestat = np.zeros_like(im_d)
    linestat[:, : h // 2] = np.nanmedian(im_d[:, : h // 2], axis=-1, keepdims=True)
    linestat[:, h // 2 :] = np.nanmedian(im_d[:, h // 2 :], axis=-1, keepdims=True)
    im_d /= linestat / avg

    linestat = np.zeros_like(im_d)
    linestat[: w // 4, : h // 2] = np.nanmedian(im_d[: w // 4, : h // 2], keepdims=True)
    linestat[w // 4 : w // 2, : h // 2] = np.nanmedian(im_d[w // 4 : w // 2, : h // 2], keepdims=True)
    linestat[w // 2 : w // 4 * 3, : h // 2] = np.nanmedian(im_d[w // 2 : w // 4 * 3, : h // 2], keepdims=True)
    linestat[w // 4 * 3 :, : h // 2] = np.nanmedian(im_d[w // 4 * 3 :, : h // 2], keepdims=True)
    linestat[: w // 4, h // 2 :] = np.nanmedian(im_d[: w // 4, h // 2 :], keepdims=True)
    linestat[w // 4 : w // 2, h // 2 :] = np.nanmedian(im_d[w // 4 : w // 2, h // 2 :], keepdims=True)
    linestat[w // 2 : w // 4 * 3, h // 2 :] = np.nanmedian(im_d[w // 2 : w // 4 * 3, h // 2 :], keepdims=True)
    linestat[w // 4 * 3 :, h // 2 :] = np.nanmedian(im_d[w // 4 * 3 :, h // 2 :], keepdims=True)
    im_d /= linestat / avg

    arr = im_d
    indices = np.where(np.isnan(arr))
    radius = 25
    for i, j in zip(indices[0], indices[1]):
        findarray = arr[i - radius : i + radius + 1, j - radius : j + radius + 1]
        findarray = findarray[~np.isnan(findarray)].flatten()
        if len(findarray) > 0:
            arr[i, j] = np.random.choice(findarray)
    im_g = arr

    im_max = np.percentile(im_g, 98)
    im_min = np.percentile(im_g, 0)
    im_g = np.clip(im_g, im_min, im_max)
    im_g = np.nan_to_num(im_g, nan=0.0)

    if np.isnan(im_g).any():
        print(f"nan slice is detected: {ii}")
    return ii, im_g.astype(np.float32)


def export_single_h5(data, sz, defect_list, estimated_dark, estimated_gain,
                      file_path, scanMater, scanNum, n_procs):
    """Cell 10: parallel per-frame correction streamed into one HDF5 file."""
    h5_out = file_path / f"{scanMater}_corr" / "data_scan{:010}_gaincorr.h5".format(scanNum)
    os.makedirs(h5_out.parent, exist_ok=True)

    defect_set = set(defect_list)  # O(1) membership checks in workers

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(
        processes=n_procs,
        initializer=_init_worker,
        initargs=(data, defect_set, estimated_dark, estimated_gain),
    )

    with h5py.File(h5_out, "w") as fout:
        dset = fout.create_dataset(
            "frames",
            shape=(sz, 576, 576),
            dtype="float32",
            chunks=(1, 576, 576),
            compression="lzf",
        )
        fout.create_dataset("estimated_dark", data=estimated_dark.astype(np.float32), compression="lzf")
        fout.create_dataset("estimated_gain", data=estimated_gain.astype(np.float32), compression="lzf")
        fout.create_dataset("defect_list", data=np.array(defect_list, dtype=np.int64))
        dset.attrs["scanMater"] = scanMater
        dset.attrs["scanNum"] = scanNum
        dset.attrs["n_defect_frames"] = len(defect_list)
        dset.attrs["note"] = "gain-corrected; defective frames stored as zero images"

        with tqdm(total=sz, desc="gain-correct+write") as pbar:
            for ii, frame in pool.imap_unordered(process_frame_h5, range(sz)):
                dset[ii] = frame
                pbar.update()

    pool.close()
    pool.join()
    print(f"Saved gain-corrected stack to: {h5_out}")


def run_one(file_path, scanMater, scanNum, n_procs):
    h5_path = file_path / scanMater / "data_scan{:010}.h5".format(scanNum)
    print(f"\n=== Processing {h5_path} ===")

    data, sz = load_and_dark_correct(h5_path)
    defect_list = detect_defects(data, sz, file_path, scanMater, scanNum)
    estimated_dark, estimated_gain = estimate_dark_gain(data, defect_list)
    export_single_h5(data, sz, defect_list, estimated_dark, estimated_gain,
                      file_path, scanMater, scanNum, n_procs)


def main():
    parser = argparse.ArgumentParser(description="Batch dark/gain correction -> single H5 export")
    parser.add_argument("--file_path", type=str, required=True,
                         help="Base directory, e.g. /pscratch/sd/s/skang2/4dcam/")
    parser.add_argument("--scanMater", type=str, required=True,
                         help="Scan material/folder name, e.g. MgCl2")
    parser.add_argument("--scanNum", type=int, nargs="+", required=True,
                         help="One or more scan numbers to process, e.g. 12 13 14")
    parser.add_argument("--n_procs", type=int, default=None,
                         help="Worker processes for per-frame correction "
                              "(default: all CPUs allocated to the job)")
    args = parser.parse_args()

    n_procs = args.n_procs or multiprocessing.cpu_count()
    file_path = Path(args.file_path)

    for scanNum in args.scanNum:
        run_one(file_path, args.scanMater, scanNum, n_procs)


if __name__ == "__main__":
    main()
