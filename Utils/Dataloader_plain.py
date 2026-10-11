"""Datasets for folders of TIF frames (file_type='Image'): training, validation and inference."""
import os
import random
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T2
from Utils.Utils import torch_zscore_normalize, idxreturn

# Bounded: with maxsize=None every DataLoader worker would cache the whole training set.
# TestLoader_plain pre-decodes its own frames, so it does not rely on this cache.
@lru_cache(maxsize=128)
def imageloader(image_path):
    return np.asarray(Image.open(image_path)).astype(np.float32)

def _list_images(d):
    # Skip stray files such as '*.tif:Zone.Identifier'.
    return sorted(n for n in os.listdir(d) if n.lower().endswith(('.tif', '.tiff')))

def Sequentialloader_plain(image_dir, image_size, validation_length=1, recursive_factor=1, frame_num=5):
    image_list = _list_images(image_dir)
    total_length = len(image_list)
    training_length = total_length-validation_length
    index = random.sample(list(range(0, total_length)), total_length)
    t_index = index[0:training_length]
    v_index = index[training_length:training_length+validation_length]
    paths = [os.path.join(image_dir, n) for n in image_list]

    Trainset = TrainLoader_plain(paths, training_length, total_length,image_size,t_index,recursive_factor,frame_num)
    Validationset = ValidationLoader_plain(paths, validation_length, total_length,image_size,v_index,frame_num)
    return Trainset, Validationset

@torch.jit.script
def gauss_noise_torch(img: torch.Tensor) -> torch.Tensor:
    sigma = torch.rand(1) * 0.5 + 0.25

    # Generate Gaussian noise
    noise = sigma * torch.randn_like(img)

    # Add noise to the image
    out = img + noise
    out = torch_zscore_normalize(out)

    return out

class TrainLoader_plain(Dataset):
    def __init__(self, paths, training_length, total_length,image_size,index,recursive_factor,frame_num):
        self.paths = paths
        self.total_length = total_length
        self.training_length = training_length
        self.transforms = T2.Compose([
        T2.RandomResize(max(256,image_size),max(512,image_size+1),interpolation=T2.InterpolationMode.NEAREST),
        T2.RandomCrop(image_size),
        T2.RandomHorizontalFlip(0.50),
        T2.RandomVerticalFlip(0.5)])
        self.index = index
        self.recursive_factor =recursive_factor
        self.frame_num = frame_num

    def __len__(self):
        return int(self.training_length*self.recursive_factor)
    
    def __getitem__(self, idx):
        idx = idx % self.training_length
        train_idx = self.index[idx]

        idxlist = idxreturn(train_idx, self.total_length, self.frame_num)

        images_array = np.stack([imageloader(self.paths[i]) for i in idxlist])
        return gauss_noise_torch(torch_zscore_normalize(self.transforms(torch.from_numpy(images_array))))

class ValidationLoader_plain(Dataset):
    def __init__(self, paths, validation_length, total_length,image_size,index,frame_num):
        self.paths = paths
        self.total_length = total_length
        self.validation_length = validation_length
        self.transforms = T2.FiveCrop(image_size)
        self.index = index
        self.frame_num = frame_num

    def __len__(self):
        return int(self.validation_length) * 4 * 5

    def __getitem__(self, idx):
        idx_img = idx//20
        crop = idx%5
        val_index = self.index[idx_img]
        idxlist = idxreturn(val_index,self.total_length,self.frame_num)
        images_array = np.stack([imageloader(self.paths[i]) for i in idxlist])
        batches = torch_zscore_normalize(self.transforms(torch.from_numpy(images_array))[crop])
        if random.random() < 0.5:
            batches = torch.flip(batches, dims=[0])
        return batches

class TestLoader_plain(Dataset):
    def __init__(self,image_dir,frame_num=5):
        self.image_dir = image_dir
        self.image_list = _list_images(image_dir)
        self.total_length = len(self.image_list)
        self.frame_num = frame_num
        # Decode every frame once: neighboring samples share most of their frames.
        self.frames = [imageloader(os.path.join(image_dir, n)) for n in self.image_list]

    def __len__(self):
        return int(self.total_length)        

    def __getitem__(self, idx):
        idxlist = idxreturn(idx,self.total_length,self.frame_num)
        img_name = self.image_list[idxlist[self.frame_num//2]]
        images_array = np.stack([self.frames[i] for i in idxlist])
        return torch.from_numpy(images_array), idx, img_name
