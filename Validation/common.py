"""Shared helpers for the Tier 2 validation tests: frame I/O, model loading, inference.

Inference reproduces TEM_denoiser_main.predict_step (z-score per frame, whole-frame forward,
min/max rescale to the raw centre frame) so results match the production outputs exactly.
"""
import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Nets.Blindspot_Net import SHINE  # noqa: E402
from Utils.Utils import idxreturn  # noqa: E402


def list_tifs(d):
    return sorted(n for n in os.listdir(d) if n.lower().endswith(('.tif', '.tiff')))


def read_tif(path):
    return np.asarray(Image.open(path)).astype(np.float32)


def read_stack(d, names=None):
    names = names or list_tifs(d)
    return np.stack([read_tif(os.path.join(d, n)) for n in names]), names


def load_shine(ckpt_path, model='3x3_blind', frame_num=5, filter=64, blocks=14, device='cuda'):
    di = int(model.split('x')[0]) // 2
    dj = int(model.split('x')[1].split('_')[0]) // 2
    net = SHINE(1, 1, add_dilation=(di, dj), frame_num=frame_num, filter=filter, blocks=blocks, Bias=False)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    net.load_state_dict({k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')})
    return net.to(device).eval()


def _zscore(x):
    m = x.mean(dim=(-2, -1), keepdim=True)
    s = x.std(dim=(-2, -1), keepdim=True)
    return (x - m) / s


@torch.no_grad()
def denoise_window(net, window, rescale=True, amp=True, device='cuda'):
    """window: (frame_num,H,W) float array, centre frame = the one denoised.

    rescale=True matches production (min/max of the raw centre frame); False returns the
    raw network output in z-score units of the centre frame. amp=True runs fp16 autocast
    like production (--precision=16).
    """
    x = torch.from_numpy(np.ascontiguousarray(window)).float().unsqueeze(0).to(device)
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        out = net(_zscore(x)).float()[0, 0]
    if not rescale:
        return out.cpu().numpy()
    raw = x[0, x.shape[1] // 2]
    out = (out - out.min()) / (out.max() - out.min())
    return (out * (raw.max() - raw.min()) + raw.min()).cpu().numpy()


def denoise_stack(net, stack, frame_num=5, rescale=True, idx=None, amp=True, device='cuda'):
    """Denoise frames `idx` (default all) of a (T,H,W) stack with production edge mirroring."""
    idx = range(len(stack)) if idx is None else idx
    return np.stack([denoise_window(net, stack[idxreturn(i, len(stack), frame_num)], rescale, amp, device)
                     for i in idx])


def affine_fit(x, y, mask=None):
    """Least-squares y ~ a*x + b in float64 (optionally over mask only)."""
    if mask is not None:
        x, y = x[mask], y[mask]
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).mean() / ((x - xm) ** 2).mean()
    return a, ym - a * xm
