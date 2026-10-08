"""Datasets for the patch-based pipelines (file_type: mrc, dm4, large, large_dm4, single, single_dm4).

Training reads the *.npz patches written by Utils/patch_generator_5frame.py; testing reads
the raw stacks / frame folders directly. Plain TIF folders ('Image') live in Dataloader_plain.py.
"""
import os
import random
from functools import lru_cache
import numpy as np
import torch
import cv2 as cv
import mrcfile
import ncempy.io.dm as dm
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from Utils.Utils import torch_zscore_normalize, idxreturn

def _list_tifs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(('.tif', '.tiff')))

def imageloader(path,index,total_length,list):
    image_path = os.path.join(path,list[index])
    image_in = torch.Tensor((cv.imread(image_path,flags=cv.IMREAD_UNCHANGED).astype(np.float32))).unsqueeze(0)
    #image_in = torch.Tensor((Image.open(image_path).astype(np.float32))).unsqueeze(0)
    
    return image_in

def Sequentialloader(image_dir, image_size,gt_path=None,validation_length=1,recursive_factor=10,frame_num=5):
    total_length = len(os.listdir(image_dir))
    training_length = total_length-validation_length
    image_size = image_size
    index = random.sample(list(range(0, total_length)), total_length)
    t_index = index[0:training_length]
    v_index = index[training_length:training_length+validation_length]
    if gt_path is None:
        gt_path = image_dir
    Trainset = TrainLoader(image_dir, training_length, gt_path,total_length,image_size,t_index,recursive_factor)
    Validationset = ValidationLoader(image_dir, validation_length, gt_path,total_length,image_size,v_index)
    #transforms = torch.jit.script(transforms)
    return Trainset, Validationset


class TrainLoader(Dataset):
    def __init__(self, image_dir, training_length, gt_path,total_length,image_size,index,recursive_factor):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(self.image_dir))
        self.training_length = training_length
        self.recursive_factor = recursive_factor
        self.transforms = T.Compose([
        T.RandomResize(max(256,image_size),max(1024,image_size+1),interpolation=T.InterpolationMode.NEAREST),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(0.50),
        T.RandomVerticalFlip(0.5),
        T.RandomApply([T.RandomRotation((90, 90))], 0.5),
        #T.ConvertImageDtype(torch.float),
        #T.Normalize([0], [1])
        ])
        self.gt_path = gt_path
        self.datalength = total_length
        self.index = index

    def __len__(self):
        return int(self.training_length*self.recursive_factor)


    def __getitem__(self, idx):
        idx = idx%self.training_length
        idxhat = self.index[idx]
        image_path = os.path.join(self.image_dir, self.image_list[idxhat])
        batch_image = np.load(image_path)['data']
        batch_image = (batch_image).astype(np.float32)
        batch_processed = (batch_image)
        batch_processed = torch.from_numpy(batch_processed)
        batch_processed = batch_processed
        batch_processed = (torch_zscore_normalize(self.transforms(batch_processed)))
        return batch_processed

class ValidationLoader(Dataset):
    def __init__(self,image_dir, validation_length, gt_path,total_length,image_size,index):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = total_length
        self.validation_length = validation_length
        self.transforms = T.Compose([
        T.CenterCrop(image_size),
        #T.ConvertImageDtype(torch.float),
        #T.Normalize([0], [1])
        ])
        self.gt_path = gt_path
        #self.gt_list = sorted(os.listdir(gt_path))
        self.index = index

    def __len__(self):
        return int(self.validation_length)        

    def __getitem__(self, idx):
        val_index = self.index[idx]
        image_path = os.path.join(self.image_dir, self.image_list[val_index])
        batch_image = torch.Tensor((np.load(image_path)['data']))
        batch_image = batch_image
        batch_processed = self.transforms(batch_image)
        batch_processed = torch_zscore_normalize(batch_processed)
        return batch_processed

def _list_dm4(image_dir):
    """Relative paths of all .dm4 files below image_dir (same search as the patch generator)."""
    return sorted(os.path.relpath(os.path.join(root, f), image_dir)
                  for root, _, files in os.walk(image_dir) for f in files if f.endswith('.dm4'))

def _load_gain(gain_dir):
    if gain_dir is None or gain_dir == 'None':
        return None
    if gain_dir.endswith('.dm4'):
        g_data = dm.fileDM(gain_dir).getDataset(0)
        g_data = g_data['data'] if isinstance(g_data, dict) else g_data
    else:
        with mrcfile.open(gain_dir, permissive=True) as mrc:
            g_data = mrc.data
    return np.flipud(g_data).astype(np.float32).copy()

@lru_cache(maxsize=16)
def _read_dm4(path):
    data = dm.fileDM(path).getDataset(0)
    return (data['data'] if isinstance(data, dict) else data).astype(np.float32)

def Sequentialloader_dm4(image_dir, image_size, patch_size=1024, gain_dir=None, validation_length=1, recursive_factor=1):
    """Train/validation sets that read single dm4 frames directly (no *.npz patches needed)."""
    image_list = _list_dm4(image_dir)
    total_length = len(image_list)
    training_length = total_length - validation_length
    index = random.sample(list(range(0, total_length)), total_length)
    gain = _load_gain(gain_dir)
    Trainset = TrainLoader_dm4(image_dir, image_list, index[:training_length], image_size, patch_size, gain, recursive_factor)
    Validationset = ValidationLoader_dm4(image_dir, image_list, index[training_length:], image_size, gain)
    return Trainset, Validationset

class TrainLoader_dm4(Dataset):
    """Single-frame training samples cropped on the fly from dm4 files.

    Equivalent to TrainLoader on patches from generate_patch_dm4_frames(frames=1): a random
    patch_size x patch_size region is cut from a frame, then the same resize/crop/flip/rotation
    augmentation is applied.
    """
    def __init__(self, image_dir, image_list, index, image_size, patch_size, gain, recursive_factor):
        self.image_dir = image_dir
        self.image_list = image_list
        self.index = index
        self.patch_size = patch_size
        self.gain = gain
        self.recursive_factor = recursive_factor
        self.transforms = T.Compose([
        T.RandomResize(max(256,image_size),max(1024,image_size+1),interpolation=T.InterpolationMode.NEAREST),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(0.50),
        T.RandomVerticalFlip(0.5),
        T.RandomApply([T.RandomRotation((90, 90))], 0.5),
        ])

    def __len__(self):
        return int(len(self.index)*self.recursive_factor)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.image_list[self.index[idx % len(self.index)]])
        img = _read_dm4(path)
        if self.gain is not None:
            img = img * self.gain
        h, w = img.shape
        ph, pw = min(self.patch_size, h), min(self.patch_size, w)
        i, j = random.randint(0, h-ph), random.randint(0, w-pw)
        patch = torch.from_numpy(np.ascontiguousarray(img[i:i+ph, j:j+pw])).unsqueeze(0)
        return torch_zscore_normalize(self.transforms(patch))

class ValidationLoader_dm4(Dataset):
    def __init__(self, image_dir, image_list, index, image_size, gain):
        self.image_dir = image_dir
        self.image_list = image_list
        self.index = index
        self.gain = gain
        self.transforms = T.CenterCrop(image_size)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img = _read_dm4(os.path.join(self.image_dir, self.image_list[self.index[idx]]))
        if self.gain is not None:
            img = img * self.gain
        return torch_zscore_normalize(self.transforms(torch.from_numpy(img).unsqueeze(0)))

class TestLoader_large(Dataset):
    def __init__(self,image_dir,subset=None,frame_num=5):
        self.image_dir = image_dir
        self.image_list = _list_tifs(image_dir)
        self.total_length = len(self.image_list)
        self.frame_num = frame_num
        self.subset = subset if subset is not None else self.total_length

    def __len__(self):
        return int(self.total_length)

    def __getitem__(self, idx):
        idx_list = idxreturn(idx,self.subset,frame_num=self.frame_num)
        img_name = self.image_list[idx_list[self.frame_num//2]]
        batch_image = torch.cat([imageloader(self.image_dir,i,self.total_length,self.image_list) for i in idx_list], 0)
        return batch_image, idx, img_name

class TestLoader_mrc(Dataset):
    def __init__(self,image_dir,subset=None,gain_dir=None):
        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.total_length = len(os.listdir(image_dir))
        if subset is not None:
            self.subset=subset
        else:
            self.subset=self.total_length
        if isinstance(gain_dir, type(None)):
            self.gain_value = None
        else:
            gain = mrcfile.mmap(gain_dir, mode='r')
            gain_value = gain.data 
            gain_value = np.flipud(gain_value)
            self.gain_value = torch.tensor(gain_value.copy())

    def __len__(self):
        return int(self.total_length)       

    def __getitem__(self, idx):
        im0 = mrcfile.open(os.path.join(self.image_dir,self.image_list[idx]))
        image = torch.tensor(np.array(im0.data,dtype=np.float32))
        if isinstance(self.gain_value, type(None)):
            gain_value = torch.ones_like(image[0,:,:])
        else:
            gain_value = self.gain_value
        data = image*gain_value
        img_name = self.image_list[idx]


        return data, idx, img_name, gain_value


class TestLoader_dm4(Dataset):
    """4D dm4 stacks (stacks1 x stacks2 x H x W): one sample per frame, with frame_num neighbours.

    Returns (frames, frame_idx, name) like the other predict loaders; name is '<stack>_<frame:05d>.dm4'
    so predict_step writes one TIF per frame.
    """
    def __init__(self, image_dir, subset=None, gain_dir=None, frames=5):
        self.image_dir = image_dir
        self.frame_num = frames
        self.image_list = sorted(f for f in os.listdir(image_dir) if f.lower().endswith('.dm4'))
        self.file_loadnumber = []
        self.file_indexes = []
        for file_num, name in enumerate(self.image_list):
            stacks1, stacks2, _, _ = dm.fileDM(os.path.join(image_dir, name)).getMemmap(0).shape
            for s in range(stacks1 * stacks2):
                self.file_loadnumber.append(file_num)
                self.file_indexes.append(idxreturn(s, stacks1*stacks2, frames))
        self.total_length = len(self.file_loadnumber)
        self.subset = subset if subset is not None else self.total_length
        self.gain_value = None
        if gain_dir is not None:
            gain_value = dm.fileDM(gain_dir).getDataset(0).astype(np.float32)
            self.gain_value = torch.tensor(np.flipud(gain_value).copy())

    def __len__(self):
        return int(self.total_length)

    def __getitem__(self, idx):
        file_num = self.file_loadnumber[idx]
        file_indexes = self.file_indexes[idx]
        memmap = dm.fileDM(os.path.join(self.image_dir, self.image_list[file_num])).getMemmap(0)
        stacks2 = memmap.shape[1]
        batch_image = torch.stack([torch.from_numpy(memmap[i//stacks2, i%stacks2].astype(np.float32)) for i in file_indexes])
        if self.gain_value is not None:
            batch_image = batch_image * self.gain_value
        frame_idx = file_indexes[self.frame_num//2]
        img_name = f'{os.path.splitext(self.image_list[file_num])[0]}_{frame_idx:05d}.dm4'
        return batch_image, frame_idx, img_name

class TestLoader_large_dm4(Dataset):
    def __init__(self, image_dir, subset=None, gain_dir=None, frame_num=5):
        self.image_dir = image_dir
        self.frame_num = frame_num
        
        self.sequences = []  
        self.image_list = [] 
        
        seq_idx = 0
        for root, _, files in sorted(os.walk(image_dir)):
            dm4_files = sorted([f for f in files if f.endswith('.dm4')])
            if dm4_files:
                self.sequences.append((root, dm4_files))
                for local_idx, f in enumerate(dm4_files):
                    self.image_list.append((seq_idx, local_idx, os.path.join(root, f)))
                seq_idx += 1
                
        self.total_length = len(self.image_list)

        if subset is not None:
            self.subset = subset
        else:
            self.subset = self.total_length

        self.gain_value = None
        if gain_dir is not None and gain_dir != 'None':
            if gain_dir.endswith('.dm4'):
                g_obj = dm.fileDM(gain_dir)
                g_data = g_obj.getDataset(0)
                g_data = g_data['data'] if isinstance(g_data, dict) else g_data
            else:
                with mrcfile.open(gain_dir, permissive=True) as mrc:
                    g_data = mrc.data
            
            self.gain_value = np.flipud(g_data).astype(np.float32)

    def __len__(self):
        return int(self.subset)

    def __getitem__(self, idx):
        seq_idx, local_idx, fpath = self.image_list[idx]
        root, dm4_files = self.sequences[seq_idx]
        
        idx_list = idxreturn(local_idx, len(dm4_files), frame_num=self.frame_num)
        
        first = True
        img_series = None
        
        # 'folderA/image.dm4' -> 'folderA_image.tif' (flat, so the output folder needs no subdirectories)
        img_name = os.path.relpath(fpath, self.image_dir).replace('.dm4', '.tif')
        img_name = img_name.replace('/', '_').replace('\\', '_')

        for neighbor_idx in idx_list:
            fname = dm4_files[neighbor_idx]
            neighbor_fpath = os.path.join(root, fname)
            
            try:
                dm_obj = dm.fileDM(neighbor_fpath)
                img_data = dm_obj.getDataset(0)
                img = img_data['data'] if isinstance(img_data, dict) else img_data
                img = img.astype(np.float32)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                img = np.zeros(self.gain_value.shape if self.gain_value is not None else (1024,1024), dtype=np.float32)

            if self.gain_value is not None:
                img = img * self.gain_value

            img = np.expand_dims(img, axis=0)

            if first:
                first = False
                img_series = img
            else:
                img_series = np.concatenate((img_series, img), axis=0)

        batch_image = torch.from_numpy(img_series)
        return batch_image, idx, img_name

class TestLoader_single(Dataset):
    def __init__(self,image_dir,subset=None):
        self.image_dir = image_dir
        self.image_list = _list_tifs(image_dir)
        self.total_length = len(self.image_list)
        if subset is not None:
            self.subset=subset
        else:
            self.subset=self.total_length

    def __len__(self):
        return int(self.total_length)       

    def __getitem__(self, idx): 
        previous_in = imageloader(self.image_dir,idx,self.total_length,self.image_list)
        img_name = self.image_list[idx]
        return previous_in, idx, img_name
    
class TestLoader_single_dm4(Dataset):
    def __init__(self, image_dir, subset=None, gain_dir=None):
        self.image_dir = image_dir
        
        # all .dm4 files below image_dir (same search as the patch generator and large_dm4)
        self.image_list = _list_dm4(image_dir)
        self.total_length = len(self.image_list)

        if subset is not None:
            self.subset = subset
        else:
            self.subset = self.total_length

        self.gain_value = _load_gain(gain_dir)

    def __len__(self):
        return int(self.subset)        

    def __getitem__(self, idx): 
        rel_path = self.image_list[idx]
        file_path = os.path.join(self.image_dir, rel_path)
        # 'folderA/image.dm4' -> 'folderA_image.tif' (flat output folder)
        img_name = rel_path.replace('.dm4', '.tif').replace('/', '_').replace('\\', '_')
        
        try:
            dm_obj = dm.fileDM(file_path)
            img_data = dm_obj.getDataset(0)
            img = img_data['data'] if isinstance(img_data, dict) else img_data
            img = img.astype(np.float32)
        except Exception as e:
            print(f"Error reading {rel_path}: {e}")
            img = np.zeros((1024, 1024), dtype=np.float32) # Assume standard size or handle dynamic

        if self.gain_value is not None:
            img = img * self.gain_value

        batch_image = torch.from_numpy(img).unsqueeze(0)
        return batch_image, idx, img_name