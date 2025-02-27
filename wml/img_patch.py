import numpy as np
from collections.abc import Iterable
import math
import wml.img_utils as wmli

class ImagePatch:
    def __init__(self,patch_size,pad=True,pad_value=127,boundary=0) -> None:
        '''
        patch_size: (H,W)
        boundary: (bh,bw) or value
        '''
        self.patch_size = patch_size
        self.pad = pad
        self.pad_value = pad_value
        self.boundary = boundary if isinstance(boundary,Iterable) else (boundary,boundary)
        self.patch_bboxes = []
        self.src_img = None
        self.cur_idx = 0
    
    def set_src_img(self,img):
        self.src_img = img
        self.cur_idx = 0
        self.patch_bboxes = []
        self.rows = math.ceil((self.src_img.shape[0]-self.boundary[0])/(self.patch_size[0]-self.boundary[0]))
        self.cols = math.ceil((self.src_img.shape[1]-self.boundary[1])/(self.patch_size[1]-self.boundary[1]))

        x = np.array(list(range(self.cols)),dtype=np.int32)*(self.patch_size[1]-self.boundary[1])
        y = np.array(list(range(self.rows)),dtype=np.int32)*(self.patch_size[0]-self.boundary[0])
        wh = np.array([self.patch_size[1],self.patch_size[0]],dtype=np.int32)
        wh = np.reshape(wh,[-1,2])
        xv,yv = np.meshgrid(x,y,sparse=False, indexing='ij')
        x0y0 = np.stack([xv,yv],axis=-1)
        x0y0 = np.reshape(x0y0,[-1,x0y0.shape[-1]])
        x1y1 = x0y0+wh
        self.bboxes = np.concatenate([x0y0,x1y1],axis=-1)
    
    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self,idx):
        bbox = self.bboxes[idx]
        self.cur_idx = idx

        if self.pad:
            size = self.patch_size[::-1]
            return wmli.crop_and_pad(self.src_img,bbox,size,pad_color=self.pad_value)
        else:
            return wmli.crop_img(self.src_img,bbox)

    def patch_bboxes2img_bboxes(self,bboxes,idx=None):
        '''
        bboxes: [N,4] (x0,y0,x1,y1)
        '''
        if idx is None:
            idx = self.cur_idx
        bbox = self.bboxes[idx]
        offset = np.array([bbox[0],bbox[1],bbox[0],bbox[1]],dtype=bboxes.dtype)
        offset = np.reshape(offset,[-1,4])
        bboxes = bboxes+offset
        return bboxes

    def cur_bbox(self):
        return self.bboxes[self.cur_idx]

    def remove_boundary_bboxes(self,bboxes,boundary=None):
        '''
        bboxes: [N,4] (x0,y0,x1,y1), in patch img
        '''
        if boundary is None:
            boundary = self.boundary
        if not isinstance(boundary,Iterable):
            boundary = (boundary,boundary)
        
        value = (boundary[0]/2,boundary[1]/2)
        
        cxy = (bboxes[...,:2]+bboxes[...,2:])/2

        mask0 = cxy[...,0]<value[1]
        mask1 = cxy[...,1]<value[0]
        mask2 = cxy[...,0]>(self.patch_size[1]-value[1])
        mask3 = cxy[...,1]>(self.patch_size[0]-value[0])
        _mask0 = np.logical_or(mask0,mask1)
        _mask2 = np.logical_or(mask2,mask3)
        mask = np.logical_or(_mask0,_mask2) 
        keep = np.logical_not(mask)

        return keep




    
