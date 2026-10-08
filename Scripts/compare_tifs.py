"""Compare two folders of denoised TIFs: python Scripts/compare_tifs.py OLD_DIR NEW_DIR"""
import os, sys
import numpy as np
import cv2 as cv

old, new = sys.argv[1], sys.argv[2]
names = sorted(f for f in os.listdir(old) if f.lower().endswith(('.tif', '.tiff')))
missing = [n for n in names if not os.path.exists(os.path.join(new, n))]
worst = 0.0
for n in names:
    if n in missing:
        continue
    a = cv.imread(os.path.join(old, n), cv.IMREAD_UNCHANGED).astype(np.float64)
    b = cv.imread(os.path.join(new, n), cv.IMREAD_UNCHANGED).astype(np.float64)
    rel = np.abs(a - b).max() / max(np.ptp(a), 1e-12)
    worst = max(worst, rel)
print(f'{len(names)-len(missing)}/{len(names)} files compared, missing in new: {len(missing)}')
print(f'worst max|diff| / range = {worst:.3e}  ({"bit-identical" if worst == 0 else "differs"})')
