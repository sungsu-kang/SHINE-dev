import os
import cv2 as cv
import mrcfile
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils.Utils import (idxreturn, clip_top_3_percent, numpy_zscore_normalize_test,
                         numpy_zscore_recover, MixedLoss, FocalFrequencyLoss)

class L1_Charbonnier_loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
    
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error) 
        return loss

class TEM_denoiser_main(pl.LightningModule):
    def __init__(self,
                network,
                in_channels,
                out_channels,
                frame_num,
                img_size,
                training_path,
                save_folder,
                time_stamp,
                model_type,
                learning_rate,
                batch_size, 
                lossF,
                beta1,
                beta2,
                eps,
                weight_decay,
                total_epochs,
                trainset,
                validationset,
                testset,
                additional_dilation_i,
                additional_dilation_j,
                num_workers=8,
                predict_batch_size=1,
                tile_size=0):
        super(TEM_denoiser_main, self).__init__()
        self.frame_num=frame_num
        self.prepare_data_per_node=False
        self.save_folder=save_folder
        self.time_stamp=time_stamp
        self.model_type=model_type
        self.learning_rate=learning_rate
        self.loss_F=lossF
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.weight_decay=weight_decay
        self.batch_size=batch_size
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.training_path=training_path
        self.img_size=img_size
        self.total_epochs=total_epochs
        self.Validationset=validationset
        self.Trainset=trainset
        self.Testset=testset
        self.loss_function_forward=self.loss_function()
        self.model=network
        self.additional_dilation=np.maximum(additional_dilation_i, additional_dilation_j)
        self.num_workers=num_workers
        self.predict_batch_size=predict_batch_size
        # tile_size <= 0: predict runs whole frames up to 1024 px (512 px tiles above that);
        # test (large mrc stacks) keeps 1024 px tiles. A positive value forces that tile size.
        self.tile_size=tile_size
        
    def forward(self, x, shuffle=0):
        out = self.model(x, shuffle=shuffle)
        return out

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(
            self.parameters(),   
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        return optimizer1
                
    def loss_function(self, *args):
        if self.loss_F == 'L2':
            return nn.MSELoss(*args)
        elif self.loss_F == 'L1':
            return nn.L1Loss(*args)
        elif self.loss_F == 'SL1':
            return nn.SmoothL1Loss(*args)
        elif self.loss_F == 'BCE':
            return nn.BCEWithLogitsLoss()
        elif self.loss_F == 'FFL':
            return FocalFrequencyLoss()
        elif self.loss_F == 'NLL':
            return nn.PoissonNLLLoss(log_input=True)
        elif self.loss_F == 'charbonnier':
            return L1_Charbonnier_loss()
        elif self.loss_F == 'Mix':
            return MixedLoss()
        raise ValueError(f'Unknown loss function: {self.loss_F}')

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/metric_1": 0, "hp/metric_2": 0})

    def train_dataloader(self):
        train_loader = DataLoader(self.Trainset, batch_size=self.batch_size, shuffle=True, pin_memory=False, 
                                    num_workers=self.num_workers, drop_last=True, persistent_workers=self.num_workers > 0)
        return train_loader

    def val_dataloader(self):
        validation_loader = DataLoader(self.Validationset, batch_size=self.batch_size, shuffle=False, pin_memory=False,
                                    num_workers=self.num_workers, drop_last=False, persistent_workers=self.num_workers > 0)
        return validation_loader

    def test_dataloader(self):
        # one stack per batch: test_step writes one mrc file per item
        test_loader = DataLoader(self.Testset, batch_size=1, shuffle=False, pin_memory=False, num_workers=min(self.num_workers, 4))
        return test_loader

    def predict_dataloader(self):
        batch_size = max(1, self.predict_batch_size)
        workers = min(self.num_workers, max(1, len(self.Testset) // batch_size))
        # no pinned memory: frames can be large and pinning fails with CUDA OOM on WSL2
        test_loader = DataLoader(self.Testset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                    num_workers=workers, persistent_workers=False)
        return test_loader

    @staticmethod
    def _split_batch(batch):
        # Datasets return either frames or a (frames, mask) pair; default_collate turns the
        # pair into a list. A plain tensor of batch size 2 must not be mistaken for a pair.
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        return batch, None

    def _prepare_frames(self, frames):
        if len(frames.shape)==3:
            frames = frames.unsqueeze(1)
        b,c,h,w = frames.shape
        now = frames[:,self.frame_num//2,:,:].unsqueeze(1)
        if c!=self.frame_num:
            frames = frames[:,:self.frame_num,:,:].squeeze(1).squeeze(2)
        return frames, now

    def _split_output(self, out, mask):
        # networks that return (denoised, mask) instead of a single tensor
        if isinstance(out, (tuple, list)):
            return out[0], out[1]
        return out, mask

    def training_step(self, batch, batch_nb):
        frames, mask = self._split_batch(batch)
        frames, now = self._prepare_frames(frames)
        out = self.forward(frames, 0) 
        out, mask = self._split_output(out, mask)
        if mask is not None:
            mask = mask.requires_grad_(False)
            out = out*(1-mask)
            now = now*(1-mask)
        # BCE uses BCEWithLogitsLoss, which expects raw logits
        loss = self.loss_function_forward(out,now)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        num_list = [0,1,2,3,4,5]
        if batch_nb == 0 and (self.current_epoch%5==0 or self.current_epoch in num_list):
            # None of the networks use the `shuffle` argument, so a second forward pass
            # would return exactly `out`; reuse it for the "revisible" image.
            shown = torch.sigmoid(out) if self.loss_F=='BCE' else out
            sample_imgs = now[0,:,:,:]
            denoised_imgs = shown[0,:,:,:].detach()
            revisible_imgs = denoised_imgs
            grid_sample = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True, scale_each=True)
            grid_denoised = torchvision.utils.make_grid(denoised_imgs, nrow =4, normalize=True, scale_each=True)
            grid_revisible = torchvision.utils.make_grid(revisible_imgs, nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.add_image('example_images', grid_sample, global_step = self.global_step)
            self.logger.experiment.add_image('denoised_image', grid_denoised, global_step = self.global_step)
            self.logger.experiment.add_image('revisible_image', grid_revisible, global_step = self.global_step)
        return loss

    def validation_step(self, batch, batch_nb):
        frames, mask = self._split_batch(batch)
        frames, now = self._prepare_frames(frames)
        out = self.forward(frames)
        out, mask = self._split_output(out, mask)
        if mask is not None:
            out = out*(1-mask)
            now = now*(1-mask)
        val_loss = self.loss_function_forward(out,now)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        num_list = [0,1,2,3,4,5]
        if batch_nb == 0 and (self.current_epoch%5==0 or self.current_epoch in num_list):
            # as in training, the blind forward pass equals `out`
            denoised_imgs = (torch.sigmoid(out) if self.loss_F=='BCE' else out).detach()
            sample_imgs = now[0,:,:,:]
            grid_sample = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True, scale_each =True)
            grid_denoised = torchvision.utils.make_grid(denoised_imgs, nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.add_image('val_example_images', grid_sample, global_step = self.global_step)
            self.logger.experiment.add_image('val_denoised_image', grid_denoised, global_step = self.global_step)
        return val_loss
    
    def _tile_size(self, height, width, auto_full_max, auto_fallback):
        if self.tile_size > 0:
            return self.tile_size
        # auto: one forward pass per frame when it fits, tiles otherwise
        return max(height, width) if max(height, width) <= auto_full_max else auto_fallback

    def _denoise_tiled(self, frames, img_size, **forward_kwargs):
        """Run the network on overlapping tiles (64 px halo) and stitch the centre frame.

        frames: (N,C,H,W) normalized frames on the model device. Returns (N,1,H,W) on the same device.
        """
        slice = 64
        denoised = torch.zeros(frames.size(0), 1, frames.size(2), frames.size(3), dtype=frames.dtype, device=frames.device)
        for i in range(0, frames.size(2), img_size):
            for j in range(0, frames.size(3), img_size):
                si = max(0, i - slice)
                ei = min(frames.size(2), i + img_size + slice)

                sj = max(0, j - slice)
                ej = min(frames.size(3), j + img_size + slice)

                xij = frames[:,:,si:ei,sj:ej]
                yij = self.forward(xij, **forward_kwargs)

                top_pad = i - si
                left_pad = j - sj
                bottom_pad = ei - (i + img_size)
                right_pad = ej - (j + img_size)

                yij = yij[:,:,top_pad:yij.shape[-2] - bottom_pad, left_pad:yij.shape[-1] - right_pad]
                denoised[:,:,i:i+yij.shape[-2],j:j+yij.shape[-1]] = yij.detach().to(denoised.dtype)
        return denoised

    def test_step(self, batch, batch_idx):
        img, idx, img_name, gain_value = batch
        denoised_dir = self.save_folder+self.time_stamp+'/denoised'
        img_name = ''.join(img_name)
        if not os.path.exists(denoised_dir):
            os.makedirs(denoised_dir, exist_ok=True)
        img_size = self._tile_size(img.shape[-2], img.shape[-1], auto_full_max=0, auto_fallback=1024)
        # clip each frame once instead of once per window it appears in
        clipped = torch.from_numpy(clip_top_3_percent(img[0].detach().cpu().numpy())).unsqueeze(0).to(img.device)
        mrc = mrcfile.new_mmap(os.path.join(denoised_dir, img_name), img.shape[-3:], mrc_mode=2, overwrite=True)
        for idx in tqdm(range(img.shape[1]),desc = 'Denoising Fraction'):
            idxlist= idxreturn(idx,img.shape[1],self.frame_num)
            frames = clipped[:,idxlist,:,:]
            frames, nonsacle = numpy_zscore_normalize_test(frames)
            denoised = self._denoise_tiled(frames.type_as(gain_value), img_size, shuffle=False).cpu()
            denoised = numpy_zscore_recover(denoised,nonsacle.cpu())
            mrc.data[idx,:,:] = np.array(denoised.squeeze(), dtype=np.float32)
        mrc.close()


    def predict_step(self, batch, batch_idx):
        frames, idx, img_names = batch
        if isinstance(img_names, str):
            img_names = [img_names]
        N, C, nH, nW = frames.shape
        frames, nonsacle = numpy_zscore_normalize_test(frames)
        img_size = self._tile_size(nH, nW, auto_full_max=1024, auto_fallback=512)
        denoised = self._denoise_tiled(frames, img_size, shuffle=self.additional_dilation+1).cpu()
        denoised = numpy_zscore_recover(denoised, nonsacle.cpu())
        denoised_dir = self.save_folder + self.time_stamp + '/denoised'
        os.makedirs(denoised_dir, exist_ok=True)
        for k, name in enumerate(img_names):
            name = os.path.splitext(name)[0] + '.tif'
            image = denoised[k].squeeze().numpy()
            cv.imwrite(os.path.join(denoised_dir, name), image.astype(np.float32))
