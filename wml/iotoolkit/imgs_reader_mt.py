import wml.img_utils as wmli
import wml.wml_utils as wmlu
from wml.wtorch.data.dataloader import DataLoader
from wml.wtorch.data._utils.collate import null_convert,default_collate
import numpy as np
import random
import sys
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

class MaxImgLongSize:
    def __init__(self,max_size=2048):
        self.max_size = max_size
    
    def __call__(self,img):
        if img.shape[0]>self.max_size or img.shape[1]>self.max_size:
            img = wmli.resize_long_size(img,self.max_size)
        return img

    def __repr__(self):
        return f"{type(self).__name__}: max_size={self.max_size}"

class ImgsDataset:
    def __init__(self,data_dir_or_files,transform=None,shuffle=True):
        if isinstance(data_dir_or_files,str):
            self.files = wmlu.get_files(data_dir_or_files,suffix=wmli.BASE_IMG_SUFFIX)
        else:
            self.files = data_dir_or_files
        if shuffle:
            random.shuffle(self.files)
        self.transform = transform
        print(f"ImgsDataset transform:")
        print(self.transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path = self.files[item]
        try:
            sys.stdout.flush()
            img = wmli.hpimread(path)
            if self.transform is not None:
                img = self.transform(img)
            return path,img
            #return path,wmli.gpu_imread(path)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.stdout.flush()
            return path,np.zeros([0,0,1],dtype=np.uint8)

class ImgsReader:
    def __init__(self, data_dir_or_files, thread_nr=8,transform=None,shuffle=True):
        self.dataset = ImgsDataset(data_dir_or_files,transform=transform,shuffle=shuffle)

        dataloader_kwargs = {"num_workers": thread_nr, "pin_memory": False}
        dataloader_kwargs["sampler"] = list(range(len(self.dataset)))
        dataloader_kwargs["batch_size"] = None
        dataloader_kwargs["batch_split_nr"] = 1
        dataloader_kwargs['collate_fn'] = null_convert
        #dataloader_kwargs['collate_fn'] = default_collate

        data_loader = DataLoader(self.dataset, **dataloader_kwargs)

        self.data_loader = data_loader
        self.data_loader_iter = iter(self.data_loader)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self.data_loader_iter
