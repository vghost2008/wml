import numpy as np
import copy
import cv2
import wtorch.utils as wtu
from .mask_utils import get_bboxes_by_contours

class WPolygonMaskItem:
    HORIZONTAL = 'horizontal'
    VERTICAL =  'vertical'
    DIAGONAL =  'diagonal'
    def __init__(self,points,*,width=None,height=None):
        '''
        points:  list[[N,2]]
        '''
        self.points = copy.deepcopy(points)
        self.width = width
        self.height = height

    def bitmap(self,width=None,height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        mask = np.zeros(shape=[height,width],dtype=np.uint8)
        segmentation = cv2.drawContours(mask,self.points,-1,color=(1),thickness=cv2.FILLED)
        return segmentation

    def resize(self,size,width=None,height=None):
        '''
        size:[w,h]
        '''
        if len(self.points)==0:
            return self
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        w_scale = size[0]/width
        h_scale = size[1]/height
        scale = np.array([[w_scale,h_scale]],dtype=np.float)
        ori_type = self.points[0].dtype
        points = [p.astype(np.float)*scale for p in self.points]
        self.points = [p.astype(ori_type) for p in points]
        self.width = size[0]
        self.height = size[1]

        return self

    def flip(self,direction,width=None,height=None):
        if len(self.points)==0:
            return self
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        if direction == self.HORIZONTAL:
            new_points = []
            for p in self.points:
                p[:,0] = width-p[:,0]
                p = p[::-1,:]
                new_points.append(p)
        elif direction == self.VERTICAL:
            for p in self.points:
                p = p[::-1,:]
                p[:,1] = height-p[:,0]
                new_points.append(p)
        elif direction == self.DIAGONAL:
            new_points = []
            for p in self.points:
                p[:,0] = width-p[:,0]
                p[:,1] = height-p[:,0]
                new_points.append(p)
        else:
            info = f"unknow flip direction {direction}"
            print(f"ERROR: {info}")
            raise RuntimeError(info)
        
        self.points = new_points

        return self


class WPolygonMasks:
    def __init__(self,masks,*,width=None,height=None,exclusion=None) -> None:
        self.masks = copy.deepcopy(masks)
        self.width = width
        self.height = height
        self.exclusion = exclusion

    def __getitem__(self,idxs):
        if isinstance(idxs,(list,np.ndarray,tuple)):
            idxs = np.array(idxs)
            print("__item__",idxs.dtype,idxs)
            if idxs.dtype == np.bool:
                idxs = np.where(idxs)[0]
                print("__item__",idxs)
            masks = [self.masks[idx] for idx in idxs]
            return WPolygonMasks(masks,width=self.width,height=self.height,exclusion=self.exclusion)
        else:
            return self.masks[idxs]

    def __len__(self):
        return len(self.masks)

    def bitmap(self,exclusion=None):
        '''
        exclusion: 如果exclusion那么一个像素只能为一个类别，索引大的优先级高，即masks[i+1]会覆盖masks[i]的重叠区域
        '''
        if len(self.masks) == 0:
            return np.zeros(shape=[0,self.height,self.width],dtype=np.uint8)

        masks = [m.bitmap(width=self.width,height=self.height) for m in self.masks]
        
        if exclusion is None:
            exclusion = self.exclusion
        if exclusion is None:
            exclusion = False
        if exclusion and len(masks)>1:
            mask = 1 - masks[-1]
            for i in reversed(range(len(masks) - 1)):
                masks[i] = np.logical_and(masks[i], mask)
                mask = np.logical_and(mask, 1 - masks[i])

        return np.stack(masks,axis=0)

    def resize(self,size):
        '''
        size:[w,h]
        '''
        self.masks = [m.resize(size,width=self.width,height=self.height) for m in self.masks]
        self.width = size[0]
        self.height = size[1]
        print(f"resize mask")
        return self

    def flip(self,direction):
        print(f"FLIP mask")
        [m.flip(direction,width=self.width,height=self.height) for m in self.masks]
        return self

    @classmethod
    def from_bitmap_masks(cls,bitmap_masks):
        masks,bboxes,keep = bitmap_masks.polygon()
        items = [WPolygonMaskItem(points=m,width=bitmap_masks.width,height=bitmap_masks.height) for m in masks]
        return cls(items,width=bitmap_masks.width,height=bitmap_masks.height)

    
class WBitmapMasks:
    def __init__(self,masks,*,width=None,height=None):
        '''
        masks: [N,H,W]
        '''
        assert len(masks.shape)==3 and masks.shape[1]>0 and masks.shape[2]>0, f"ERROR: error points shape {masks.shape}"
        if masks.dtype != np.uint8:
            print("WARNING: masks dtype is not uint8")
            masks = masks.astype(np.uint8)
        self.masks = copy.deepcopy(masks)
        self.width = width if width is not None else masks.shape[2]
        self.height = height if height is not None else masks.shape[1]

    def polygon(self,bboxes=None):
        t_masks = []
        keep = np.ones([self.masks.shape[0]],dtype=np.bool)
        res_bboxes = []
        for i in range(self.masks.shape[0]):
            if bboxes is not None:
                contours = wtu.find_contours_in_bbox(self.masks[i],bboxes[i])
            else:
                contours,hierarchy = cv2.findContours(self.masks[i],cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            t_bbox = get_bboxes_by_contours(contours)
            if len(contours)==0 or not np.all(t_bbox[2:]-t_bbox[:2]>1):
                keep[i] = False
            t_masks.append(contours)
            res_bboxes.append(t_bbox)

        if len(res_bboxes) == 0:
            res_bboxes = np.zeros([0,4],dtype=np.float32)
        else:
            res_bboxes = np.array(res_bboxes,dtype=np.float32)

        return t_masks,res_bboxes,keep

    @classmethod
    def from_polygon_masks(cls,polygon_masks):
        return cls(masks=polygon_masks.bitmap())
