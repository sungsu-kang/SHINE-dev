import torch
import numpy as np
import torch.fft
import torch.nn as nn
import numba

def idxreturn(idx, stack_num, frame_num):
    """Indices of the frame_num frames centred on idx; at the stack edges the window is mirrored."""
    half = frame_num//2
    if idx > (stack_num-half-1):
        return [idx-abs(i) for i in range(-half, half+1)]
    if idx < half:
        return [idx+abs(i) for i in range(-half, half+1)]
    return [idx+i for i in range(-half, half+1)]

@torch.jit.script
def torch_zscore_normalize(image):
    if image.dim() == 2:
        # Compute mean and std over the entire image for 2D images
        mean = image.mean()
        std = image.std()
        normalized_image = (image - mean) / std
    elif image.dim() == 3 or image.dim() == 4:
        # Compute mean and std over each channel (of each image in the batch for 4D)
        mean = image.mean(dim=(-2,-1), keepdim=True)
        std = image.std(dim=(-2,-1), keepdim=True)
        normalized_image = (image - mean) / std
    else:
        raise ValueError("Unsupported image dimension. Image must be 2D, 3D, or 4D.")

    return normalized_image

@numba.jit(nopython=True, cache=True)
def numpy_zscore_normalize(image):
    output = np.empty_like(image)
    channels = image.shape[0]
    for c in range(channels):
        v = image[c,:,:].flatten()
        mean = v.mean()
        std = v.std()
        output[c,:,:] = (image[c,:,:] - mean) / std
    return output

@numba.jit(nopython=True, cache=True)
def clip_top_3_percent(img):
    """Clip every channel of a (C,H,W) array at its 99.7th percentile."""
    assert img.ndim == 3, "Input must be a 3D array"
    c, h, w = img.shape
    clipped_img = np.zeros_like(img)
    for i in range(c):
        channel = img[i, :, :]
        threshold = np.percentile(channel, 99.7)
        clipped_img[i, :, :] = np.clip(channel, None, threshold)
    return clipped_img

def numpy_zscore_normalize_test(image):
    """Z-score every frame (last two dims). Returns (normalized, un-normalized center frame).

    Works on a new tensor; the input is not modified. Statistics are per
    (batch, channel) frame, so the result does not depend on the batch size.
    """
    if image.dim() == 2:
        image_nonscale = image.clone()
    elif image.dim() == 3:
        image_nonscale = image[image.shape[0]//2, :, :].clone()
    elif image.dim() == 4:
        image_nonscale = image[:, image.shape[1]//2, :, :].clone()
    else:
        raise ValueError("Unsupported image dimension. Image must be 2D, 3D, or 4D.")
    mean = image.mean(dim=(-2, -1), keepdim=True)
    std = image.std(dim=(-2, -1), keepdim=True)
    return (image - mean) / std, image_nonscale

def numpy_zscore_recover(image, original):
    """Rescale each denoised frame to the min/max range of its own original frame.

    image:    (H,W), (C,H,W) or (N,C,H,W) denoised output
    original: un-normalized center frame, (H,W) or (N,H,W) matching image's batch dim
    The old implementation used the min/max of the whole batch of originals, which
    is only correct for batch size 1.
    """
    v2min = image.amin(dim=(-2, -1), keepdim=True)
    v2max = image.amax(dim=(-2, -1), keepdim=True)
    if image.dim() == 4 and original.dim() == 3 and original.shape[0] == image.shape[0]:
        omin = original.amin(dim=(-2, -1)).reshape(-1, 1, 1, 1)
        omax = original.amax(dim=(-2, -1)).reshape(-1, 1, 1, 1)
    else:
        omin = original.min()
        omax = original.max()
    return (image - v2min) / (v2max - v2min) * (omax - omin) + omin

class MixedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self,img1,img2):
        loss_a = self.mae(img1,img2)
        loss_s = self.mse(img1,img2)
        return loss_a + loss_s

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm='ortho')
        return torch.stack([freq.real, freq.imag], -1)

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight