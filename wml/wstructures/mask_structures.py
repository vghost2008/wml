import numpy as np
import copy
import cv2
from wml.semantic import basic_toolkit as bmt
from wml.semantic.mask_utils import get_bboxes_by_contours,npresize_mask
import wml.object_detection2.bboxes as odb
import wml.basic_img_utils as bwmli
import wml.object_detection2.bboxes as odb
from wml.semantic.basic_toolkit import findContours
from .common import WBaseMaskLike
import traceback
import sys

WBaseMask = WBaseMaskLike


class WPolygonMaskItem:
    '''
    用于表示一张图上的一个mask实例
    '''
    def __init__(self,points,*,width=None,height=None):
        '''
        points:  list[[N,2]],
        example:
        [np.zeros(3,2),np.zeros(11,2)]
        '''
        for p in points:
            if len(p.shape)!=2 or p.shape[1]!=2:
                raise RuntimeError(f"ERROR: error polygon mask item points, p shape {p.shape}, expected [N,2]")
        self.points = [p.copy().astype(np.int32) for p in points] # shape of p is [Ni,2]
        self.width = width
        self.height = height

    def copy(self):
        return WPolygonMaskItem(self.points,width=self.width,height=self.height)

    def bitmap(self,width=None,height=None):
        '''
        return: [H,W]
        '''
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        mask = np.zeros(shape=[height,width],dtype=np.uint8)
        if len(self.points)>0 and len(self.points[0])>0:
            mask = cv2.drawContours(mask,self.points,-1,color=(1),thickness=cv2.FILLED)
        return mask  

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
        scale = np.array([[w_scale,h_scale]],dtype=np.float32)
        ori_type = self.points[0].dtype
        points = [p.astype(np.float32)*scale for p in self.points]
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
        
        if direction == WBaseMask.HORIZONTAL:
            new_points = []
            for p in self.points:
                p[:,0] = width-p[:,0]
                p = p[::-1,:]
                new_points.append(p)
        elif direction == WBaseMask.VERTICAL:
            new_points = []
            for p in self.points:
                p = p[::-1,:]
                p[:,1] = height-p[:,1]
                new_points.append(p)
        elif direction == WBaseMask.DIAGONAL:
            new_points = []
            for p in self.points:
                p[:,0] = width-p[:,0]
                p[:,1] = height-p[:,1]
                new_points.append(p)
        else:
            info = f"unknow flip direction {direction}"
            print(f"ERROR: {info}")
            raise RuntimeError(info)
        
        self.points = new_points

        return self
    
    def crop(self, bbox):
        '''
        bbox: [x0,y0,x1,y1]
        offset: [xoffset,yoffset]
        如果offset is not None, 先offset再用bbox crop
        ''' 
        if not isinstance(bbox,np.ndarray):
            bbox = np.array(bbox)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.astype(np.int32).copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width-1)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height-1)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1+1, 1)
        h = np.maximum(y2 - y1+1, 1)

        if len(self.points) == 0:
            cropped_masks = WPolygonMaskItem([], height=h, width=w)
        else:
            cropped_masks = []
            for ps in self.points:
                x0 = np.min(ps[:,0])
                y0 = np.min(ps[:,1])
                x1 = np.max(ps[:,0])
                y1 = np.max(ps[:,1])
                obbox = np.array([x0,y0,x1,y1])
                iou = odb.npbboxes_intersection_of_box0(obbox[None,:],bbox[None,:])
                if iou<=1e-3: #如果剪切框与当前多边形的外包框没有交集，那么剪切的结果为空
                    continue

                e_bbox = odb.bbox_of_boxes(np.stack([bbox,obbox],axis=0)).astype(np.int32)

                if 0==bbox[0] and 0==bbox[1] and x1<=bbox[2] and y2<=bbox[3]:
                    f_poly_masks = [ps.copy()]
                elif odb.equal_bboxes(bbox,e_bbox):  #如果剪切框完全包围当前多边形，要么只需要简单的剪切处理
                    sub_cropped_masks = WPolygonMaskItem([ps.copy()],width=self.width,height=self.height)
                    f_cropped_masks = sub_cropped_masks.simple_crop(e_bbox)
                    f_poly_masks = f_cropped_masks.points
                else:
                    sub_cropped_masks = WPolygonMaskItem([ps.copy()],width=self.width,height=self.height)
                    f_cropped_masks = sub_cropped_masks.simple_crop(e_bbox)
                    f_bitmap_masks = WBitmapMasks(f_cropped_masks.bitmap()[None])
                    n_crop_bbox = bbox-np.array([e_bbox[0],e_bbox[1],e_bbox[0],e_bbox[1]])
                    f_bitmap_masks = f_bitmap_masks.crop(n_crop_bbox)
                    t_masks,res_bboxes,keep = f_bitmap_masks.polygon()
                    if not keep[0]:
                        continue
                    f_poly_masks = t_masks[0]

                cropped_masks.extend(f_poly_masks)
            cropped_masks = WPolygonMaskItem(cropped_masks, width=w,height=h)
        return cropped_masks

    def simple_crop(self, bbox):
        '''
        只处理mask完全保留或者masp完全不保留的情况
        bbox: [x0,y0,x1,y1]
        bbox的值可能为负值，如一个大的mask旋转后
        offset: [xoffset,yoffset]
        如果offset is not None, 先offset再用bbox crop
        ''' 
        if not isinstance(bbox,np.ndarray):
            bbox = np.array(bbox)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1+1, 1)
        h = np.maximum(y2 - y1+1, 1)

        if len(self.points) == 0:
            cropped_masks = WPolygonMaskItem([], height=h, width=w)
        else:
            cropped_masks = []
            for ps in self.points:
                x0 = np.min(ps[:,0])
                y0 = np.min(ps[:,1])
                x1 = np.max(ps[:,0])
                y1 = np.max(ps[:,1])
                obbox = np.array([x0,y0,x1,y1])
                iou = odb.npbboxes_intersection_of_box0(obbox[None,:],bbox[None,:])
                if iou<=1e-3: #如果剪切框与当前多边形的外包框没有交集，那么剪切的结果为空
                    continue
                ps = ps.copy()
                ps[:,0] = ps[:,0] - bbox[0]
                ps[:,1] = ps[:,1] - bbox[1]
                ps[:,0] = np.clip(ps[:,0],0,bbox[2]-bbox[0]+1)
                ps[:,1] = np.clip(ps[:,1],0,bbox[3]-bbox[1]+1)

                x0 = np.min(ps[:,0])
                y0 = np.min(ps[:,1])
                x1 = np.max(ps[:,0])
                y1 = np.max(ps[:,1])
                obbox = np.array([x0,y0,x1,y1])
                if odb.area(obbox)<=1:
                    continue

                cropped_masks.append(ps)
            cropped_masks = WPolygonMaskItem(cropped_masks, width=w,height=h)
        return cropped_masks

    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        #out_shape: [h,w]
        """See :func:`BaseInstanceMasks.rotate`."""
        if len(self.points) == 0:
            rotated_masks = WPolygonMaskItem([], width=out_shape[1],height=out_shape[0])
        else:
            rotated_masks = []
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            for p in self.points:
                coords = p.copy()
                # pad 1 to convert from format [x, y] to homogeneous
                # coordinates format [x, y, 1]
                coords = np.concatenate(
                    (coords, np.ones((coords.shape[0], 1), coords.dtype)),
                    axis=1)  # [n, 3]
                rotated_coords = np.matmul(
                    rotate_matrix[None, :, :],
                    coords[:, :, None])[..., 0]  # [n, 2, 1] -> [n, 2]
                rotated_masks.append(rotated_coords[:,:2])
            rotated_masks = WPolygonMaskItem(rotated_masks, width=out_shape[1],height=out_shape[0])
            rotated_masks = rotated_masks.crop(np.array([0,0,out_shape[1]-1,out_shape[0]-1]))
        return rotated_masks

    def warp_affine(self,M,out_shape,fill_val=0):
        #out_shape: [h,w]
        """See :func:`BaseInstanceMasks.rotate`."""
        if len(self.points) == 0:
            affined_masks = WPolygonMaskItem([], width=out_shape[1],height=out_shape[0])
        else:
            affined_masks = []
            for p in self.points:
                coords = p.copy()
                # pad 1 to convert from format [x, y] to homogeneous
                # coordinates format [x, y, 1]
                coords = np.concatenate(
                    (coords, np.ones((coords.shape[0], 1), coords.dtype)),
                    axis=1)  # [n, 3]
                affined_coords = np.matmul(
                    M[None, :, :],
                    coords[:, :, None])[..., 0]  # [n, 2, 1] -> [n, 2]
                affined_masks.append(affined_coords[:,:2])
            affined_masks = WPolygonMaskItem(affined_masks, width=out_shape[1],height=out_shape[0])
            affined_masks = affined_masks.crop(np.array([0,0,out_shape[1]-1,out_shape[0]-1]))
        return affined_masks

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        #out_shape: [h,w]
        if len(self.points) == 0:
            sheared_masks = WPolygonMaskItem([], width=out_shape[1],height=out_shape[0])
        else:
            sheared_masks = []
            if direction == 'horizontal':
                shear_matrix = np.stack([[1, magnitude],
                                         [0, 1]]).astype(np.float32)
            elif direction == 'vertical':
                shear_matrix = np.stack([[1, 0], [magnitude,
                                                  1]]).astype(np.float32)
            for p in self.points():
                p = p.copy()
                new_coords = np.matmul(shear_matrix, p.T)  # [2, n]
                new_coords[0, :] = np.clip(new_coords[0, :], 0,
                                           out_shape[1])
                new_coords[1, :] = np.clip(new_coords[1, :], 0,
                                           out_shape[0])
                sheared_masks.append(new_coords.transpose((1, 0)))
            sheared_masks = WPolygonMaskItem(sheared_masks, width=out_shape[1],height=out_shape[0])
        return sheared_masks

    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  fill_val=None,
                  interpolation=None):
        """Translate the PolygonMasks.

        Example:
            >>> self = PolygonMasks.random(dtype=np.int32)
            >>> out_shape = (self.height, self.width)
            >>> new = self.translate(out_shape, 4., direction='horizontal')
            >>> assert np.all(new.masks[0][0][1::2] == self.masks[0][0][1::2])
            >>> assert np.all(new.masks[0][0][0::2] == self.masks[0][0][0::2] + 4)  # noqa: E501
        """
        if len(self.points) == 0:
            res_masks = WPolygonMaskItem([], width=out_shape[1],height=out_shape[0])
        else:
            translated_masks = []
            for p in self.points:
                p = p.copy()
                if direction == WBaseMask.HORIZONTAL:
                    p[:,0] = np.clip(p[:,0] + offset, 0, out_shape[1])
                elif direction == WBaseMask.VERTICAL:
                    p[:,1] = np.clip(p[:,1] + offset, 0, out_shape[0])
                else:
                    info = f"error direction {direction}"
                    print(f"ERROR: {type(self).__name__} {info}")
                    raise RuntimeError(info)
                translated_masks.append(p)
            res_masks = WPolygonMaskItem(translated_masks, width=out_shape[1],height=out_shape[0])
        return res_masks

    def offset(self,
               offset):
        '''
        offset: [xoffset,yoffset]
        '''
        w = self.width+offset[0] if self.width is not None else None
        h = self.height+offset[1] if self.height is not None else None
        offset = np.reshape(np.array(offset),[1,2])
        if len(self.points) == 0:
            res_masks = WPolygonMaskItem([], width=w,height=h)
        else:
            translated_masks = []
            for p in self.points:
                p = p.copy()+offset
                translated_masks.append(p)
            res_masks = WPolygonMaskItem(translated_masks, width=w,height=h)
        return res_masks

    def get_bbox(self):
        if len(self.points) == 0:
            return np.zeros([4],dtype=np.float32)

        points = np.concatenate(self.points,axis=0)

        if len(points) == 0: #len(points)不为0，cocat的结果可能为零
            return np.zeros([4],dtype=np.float32)

        if len(points)==0:
            gt_bbox = np.zeros([4],dtype=np.float32)
        else:
            xy_min = np.min(points,axis=0)
            xy_max = np.max(points,axis=0)
            x0,y0 = xy_min[0],xy_min[1]
            x1,y1 = xy_max[0],xy_max[1]
            gt_bbox = np.array([x0,y0,x1,y1],dtype=np.float32)
        

        return gt_bbox
    
    def get_bbox(self):
        if len(self.points) == 0:
            return np.zeros([4],dtype=np.float32)

        points = np.concatenate(self.points,axis=0)

        if len(points) == 0: #len(points)不为0，cocat的结果可能为零
            return np.zeros([4],dtype=np.float32)

        if len(points)==0:
            gt_bbox = np.zeros([4],dtype=np.float32)
        else:
            xy_min = np.min(points,axis=0)
            xy_max = np.max(points,axis=0)
            x0,y0 = xy_min[0],xy_min[1]
            x1,y1 = xy_max[0],xy_max[1]
            gt_bbox = np.array([x0,y0,x1,y1],dtype=np.float32)
        

        return gt_bbox
    

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    @property
    def shape(self):
        return [len(self.points),self.height,self.width]

    def _update_shape(self,*,width=None,height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

class WPolygonMasks(WBaseMask):
    '''
    用于表示一张图上的多个Mask实例
    '''
    def __init__(self,masks,*,width=None,height=None,exclusion=None) -> None:
        super().__init__()
        self.masks = copy.deepcopy(masks)
        self.width = width
        self.height = height
        self.exclusion = exclusion
        if self.width<5 or self.height<5:
            print(f"WARNING: {self.__class__.__name__}: unnormal mask size, width={self.width}, height={self.height}")

    @classmethod
    def zeros(cls,*,width=None,height=None,shape=None):
        '''
        shape: [masks_nr,H,W]
        '''
        if shape is not None:
            width = shape[-1]
            height = shape[-2]
        return cls([],width=width,height=height)

    @classmethod
    def from_ndarray(cls,masks,*,width=None,height=None):
        _masks,bboxes,keep = WBitmapMasks.ndarray2polygon(masks)
        masks = []
        for i,k in enumerate(keep):
            if k:
                masks.append(_masks[k])

        return cls(masks=masks,width=width,height=height)

    
    @classmethod
    def from_bboxes_masks(cls,bboxes,masks,*,width=None,height=None):
        '''
        bboxes: [N,4](x0,y0,x1,y1)
        masks: [N,h,w] 仅包含bboxes内的部分
        '''
        items = []
        for bbox,mask in zip(bboxes,masks):
            x0,y0,x1,y1 = bbox
            scale = np.reshape(np.array([(x1-x0)/mask.shape[1],(y1-y0)/mask.shape[0]],dtype=np.float32),[1,2])
            offset = np.reshape(np.array([x0,y0],dtype=np.float32),[1,2])
    
            contours, hierarchy = findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            shapes = []
            for cont in contours:
                points = cont
                if len(cont.shape)==3 and cont.shape[1]==1:
                    points = np.squeeze(points,axis=1)
                points = points*scale+offset
                points = points.astype(np.int32)
                if len(points)<=2:
                    continue
                shapes.append(points)
            items.append(WPolygonMaskItem(shapes,width=width,height=height))

        return cls(masks=items,width=width,height=height)

    def copy(self):
        return WPolygonMasks(self.masks,width=self.width,height=self.height,exclusion=self.exclusion)

    def __getitem__(self,idxs):
        if isinstance(idxs,(list,tuple)) and (len(idxs)==2 or len(idxs)==3) and isinstance(idxs[0],slice):
            sx = idxs[-1]
            sy = idxs[-2]
            if self.is_none_slice(sy) and self.is_flip_slice(sx):
                return self.flip(WBaseMask.HORIZONTAL)
            elif self.is_none_slice(sx) and self.is_flip_slice(sy):
                return self.flip(WBaseMask.VERTICAL)
            elif self.is_flip_slice(sx) and self.is_flip_slice(sy):
                return self.flip(WBaseMask.DIAGONAL)
            bbox = self.slice2bbox(sx=sx,sy=sy)
            return self.crop(bbox)
        elif isinstance(idxs,(list,np.ndarray,tuple)):
            idxs = np.array(idxs)
            if idxs.dtype == bool:
                idxs = np.where(idxs)[0]
            try:
                masks = [self.masks[idx] for idx in idxs]
            except Exception as e:
                print(e)
                pass
            return WPolygonMasks(masks,width=self.width,height=self.height,exclusion=self.exclusion)
        else:
            return self.masks[idxs]

    def __setitem__(self,idxs,value):
        if isinstance(idxs,(list,tuple)) and (len(idxs)==2 or len(idxs)==3) and isinstance(idxs[0],slice):
            sx = idxs[-1]
            sy = idxs[-2]
            bbox = self.slice2bbox(sx=sx,sy=sy)
            self.copy_from(value,bbox)
        elif isinstance(idxs,(list,np.ndarray,tuple)):
            idxs = np.array(idxs)
            if idxs.dtype == bool:
                idxs = np.where(idxs)[0]
            if len(value) != len(idxs):
                info = f"idxs size not equal value's size {len(idxs)} vs {len(value)}"
                print(f"ERROR: {type(self).__name__}: {info}")
                raise RuntimeError(info)
            for i in idxs:
                self.masks[i] = value[i]
        else:
            info = f"unknow idxs type {type(idxs)}"
            print(f"ERROR: {type(self).__name__}: {info}")
            raise RuntimeError(info)

    def __len__(self):
        return len(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

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
        return self

    def flip(self,direction=WBaseMask.HORIZONTAL):
        [m.flip(direction,width=self.width,height=self.height) for m in self.masks]
        return self

    def crop(self,bbox):
        '''
        bbox: [x0,y0,x1,y1]
        '''
        bbox = np.array(bbox)
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1+1, 1)
        h = np.maximum(y2 - y1+1, 1)
        masks = [m.crop(bbox) for m in self.masks]
        return WPolygonMasks(masks,width=w,height=h)

    def slice2bbox(self,*,sx:slice,sy:slice):
        x0 = sx.start if sx.start is not None else 0
        x1 = sx.stop-1 if sx.stop is not None else self.width-1
        y0 = sy.start if sy.start is not None else 0
        y1 = sy.stop-1 if sy.stop is not None else self.height-1
        return np.array([x0,y0,x1,y1],dtype=np.int32)

    @staticmethod
    def is_flip_slice(s:slice):
        return s.start is None and s.stop is None and s.step==-1

    @staticmethod
    def is_none_slice(s:slice):
        return s.start is None and s.stop is None and s.step is None

    @staticmethod
    def get_bbox_size(bbox):
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)
        return w,h

    def copy_from(self,masks,dst_bbox=None,src_bbox=None,update_size=False):
        if src_bbox is not None:
            masks = masks.crop(src_bbox)
            if dst_bbox is not None:
                w0,h0 = self.get_bbox_size(src_bbox)
                w1,h1 = self.get_bbox_size(dst_bbox)
                if w0!=w1 or h0!=h1:
                    info = f"dst_bbox and src_bbox expected to be equal."
                    print(f"ERROR: {info}")
                    raise RuntimeError(info)
        
        if dst_bbox is not None:
            if update_size:
                w,h = self.get_bbox_size(dst_bbox)
                self.width = w
                self.height = h
            self.masks = [m.offset(dst_bbox[:2]) for m in masks]
        else:
            self.masks = masks
            if update_size:
                self.width = masks.width
                self.height = masks.height
        
        return self


    @classmethod
    def from_bitmap_masks(cls,bitmap_masks):
        masks,bboxes,keep = bitmap_masks.polygon()
        items = [WPolygonMaskItem(points=m,width=bitmap_masks.width,height=bitmap_masks.height) for m in masks]
        return cls(items,width=bitmap_masks.width,height=bitmap_masks.height)

    @staticmethod
    def concatenate(masks):
        ws = [m.width for m in masks]
        hs = [m.height for m in masks]
        ws = list(filter(lambda x:x is not None,ws))
        hs = list(filter(lambda x:x is not None,hs))
        if len(ws)>0:
            nw = np.max(ws)
        else:
            nw = None
        
        if len(hs)>0:
            nh = np.max(hs)
        else:
            nh = None
        new_masks = []
        for m in masks:
            new_masks.extend(m.masks)
        return WPolygonMasks(new_masks,width=nw,height=nh)

    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        #out_shape: [h,w]
        masks = [m.rotate(out_shape, angle, center, scale, fill_val) for m in self.masks]
        width = out_shape[1]
        height = out_shape[0]
        return WPolygonMasks(masks,width=width,height=height)
    
    def warp_affine(self,M,out_shape,fill_val=0):
        #out_shape: [h,w]
        masks = [m.warp_affine(M,out_shape,fill_val=fill_val) for m in self.masks]
        width = out_shape[1]
        height = out_shape[0]
        return WPolygonMasks(masks,width=width,height=height)
    

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        #out_shape: [h,w]
        masks = [m.shear(out_shape, magnitude, direction, border_value, interpolation) for m in self.masks]
        width = out_shape[1]
        height = out_shape[0]
        return WPolygonMasks(masks,width=width,height=height)

    def translate(self,
                  out_shape,
                  offset,
                  direction=WBaseMask.HORIZONTAL,
                  fill_val=None,
                  interpolation=None):
        '''
        out_shape: [H,W]
        '''
        masks = [m.translate(out_shape, offset, direction, fill_val,interpolation) for m in self.masks]
        width = out_shape[1]
        height = out_shape[0]
        return WPolygonMasks(masks,width=width,height=height)
    
    def pad(self, out_shape, pad_val=0):
        '''
        out_shape: [H,W]
        '''
        """padding has no effect on polygons`"""
        return WPolygonMasks(self.masks, height=out_shape[0],width=out_shape[1])

    @property
    def shape(self):
        return (len(self.masks),self.height,self.width)

    def resize_mask_in_bboxes(self,bboxes,size=None,r=None):
        '''
        mask: [N,H,W]
        bboxes: [N,4](x0,y0,x1,y1)
        size: (new_w,new_h)
        '''
        return self.resize(size),None

    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.bitmap()

    def get_bboxes(self):
        gt_bboxes = [m.get_bbox() for m in self.masks]
        if len(gt_bboxes) == 0:
            return np.zeros([0,4],dtype=np.float32)
        gt_bboxes = np.stack(gt_bboxes,axis=0)

        return gt_bboxes

    def check_consistency(self):
        for mask in self.masks:
            if mask.width != self.width or mask.height != self.height:
                info = f"Unmatch size WPolygonMasks shape {self.shape} vs WPolygonMakskItem shape {mask.shape}"
                print(info)
                raise RuntimeError(info)

    def update_shape(self,*,width=None,height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        for mask in self.masks:
            mask._update_shape(width=width,height=height)

    
class WBitmapMasks(WBaseMask):
    def __init__(self,masks,*,width=None,height=None):
        '''
        masks: [N,H,W]
        '''
        self.reinit(masks=masks,width=width,height=height)

    def reinit(self,masks,*,width=None,height=None):
        if len(masks.shape)==2:
            print(masks.shape)
            pass

        assert len(masks.shape)==3 and masks.shape[1]>0 and masks.shape[2]>0, f"ERROR: error points shape {masks.shape}"
        super().__init__()
        self.width = width if width is not None else masks.shape[2]
        self.height = height if height is not None else masks.shape[1]

        if self.width<5 or self.height<5:
            #处理WPolygonMasks剪切时，很容易出现这种情况
            #stack_info = traceback.format_stack()
            #print("\n".join(stack_info))
            sys.stdout.flush()
            print(f"WARNING: {self.__class__.__name__}: unnormal mask size, width={self.width}, height={self.height}, mask shape={masks.shape}")

        if len(masks) == 0:
            self.masks = np.zeros((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, tuple,np.ndarray))
            if isinstance(masks, (list,tuple)):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)
            if isinstance(masks,np.ndarray):
                self.masks = masks
            else:
                self.masks = np.stack(masks).reshape(-1, height, width)
        if self.masks.dtype != np.uint8:
            print("WARNING: masks dtype is not uint8")
            self.masks = self.masks.astype(np.uint8)
        
        self.masks = np.ascontiguousarray(self.masks)

    @classmethod
    def new(cls,masks,*,width=None,height=None):
        return cls(masks=masks,width=width,height=height)

    @classmethod
    def from_ndarray(cls,masks,*,width=None,height=None):
        return cls(masks=masks,width=width,height=height)

    @classmethod
    def from_bboxes_masks(cls,bboxes,masks,*,width=None,height=None):
        '''
        bboxes: [N,4](x0,y0,x1,y1)
        masks: [N,h,w] 仅包含bboxes内的部分
        '''
        masks = masks.astype(np.uint8)

        bboxes[:,0::2] = np.clip(bboxes[:,0::2],a_min=0,a_max=width)
        bboxes[:,1::2] = np.clip(bboxes[:,1::2],a_min=0,a_max=height)
        bitmap = np.zeros([bboxes.shape[0],height,width],dtype=np.uint8)
        for i,bbox in enumerate(bboxes):
            x = int(bbox[1])
            y = int(bbox[0])
            w = int((bbox[3]-bbox[1]))
            h = int((bbox[2]-bbox[0]))
            if w<=0 or h<=0:
                continue
            mask = masks[i]
            mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
            try:
                bitmap[i,y:y+h,x:x+w] = mask
            except Exception as e:
                print(f"ERROR WBitmapMasks: {e}")
                pass
    
        return cls(masks=bitmap,width=width,height=height)
    
    def __getitem__(self, index):
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        try:
            masks = self.masks[index]
            return self.new(masks)
        except Exception as e:
            print(f"ERROR WBitmapMasks: {e} index={index}, masks shape={self.masks.shape}, new masks shape {masks.shape}")
            raise e

    def __setitem__(self,idxs,value):
        if isinstance(value,WBitmapMasks):
            value = value.masks
        if isinstance(idxs,(list,tuple)) and (len(idxs)==2 or len(idxs)==3) and isinstance(idxs[0],slice):
            self.masks[idxs] = value
        elif isinstance(idxs,(list,np.ndarray,tuple)):
            idxs = np.array(idxs)
            if idxs.dtype == bool:
                idxs = np.where(idxs)[0]
            if len(value) != len(idxs) and not isinstance(value,(int,float)):
                info = f"idxs size not equal value's size {len(idxs)} vs {len(value)}"
                print(f"ERROR: {type(self).__name__}: {info}")
                raise RuntimeError(info)
            if isinstance(value,(float,int)):
                for i in idxs:
                    self.masks[i] = value
            else:
                for i in idxs:
                    self.masks[i] = value[i]
        else:
            info = f"unknow idxs type {type(idxs)}"
            print(f"ERROR: {type(self).__name__}: {info}")
            raise RuntimeError(info)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)
    

    def polygon(self,bboxes=None):
        return self.ndarray2polygon(self.masks,bboxes=bboxes)


    @staticmethod
    def ndarray2polygon(masks,bboxes=None):
        '''
        masks: [N,H,W]
        return:
        t_masks: list of list [N,2] points
        '''
        t_masks = []
        keep = np.ones([masks.shape[0]],dtype=bool)
        res_bboxes = []
        for i in range(masks.shape[0]):
            if bboxes is not None:
                contours = bmt.find_contours_in_bbox(masks[i],bboxes[i])
            else:
                contours,hierarchy = findContours(masks[i],cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
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
    
    def resize(self, size):
        '''
        size:[w,h]
        '''
        if len(self.masks) == 0:
            resized_masks = np.empty((0, size[1],size[0]), dtype=np.uint8)
        else:
            resized_masks = npresize_mask(self.masks,size)
        return self.new(resized_masks)

    def resize_mask_in_bboxes(self,bboxes,size=None,r=None):
        '''
        mask: [N,H,W]
        bboxes: [N,4](x0,y0,x1,y1)
        size: (new_w,new_h)
        '''
        if bboxes is None or self.masks.shape[0]==0:
            return self.resize(size),np.zeros([0,4],dtype=bboxes.dtype)
        mask = self.masks.copy()
        x_scale = size[0]/mask.shape[2]
        y_scale = size[1]/mask.shape[1]
        bboxes = odb.correct_bboxes(bboxes,size=[mask.shape[-1],mask.shape[-2]])
        resized_bboxes = (bboxes*np.array([[x_scale,y_scale,x_scale,y_scale]])).astype(np.int32)
        resized_bboxes = odb.correct_bboxes(resized_bboxes,size=size)
        bboxes = np.array(bboxes).astype(np.int32)

        res_mask = np.zeros([mask.shape[0],size[1],size[0]],dtype=mask.dtype)
        for i in range(mask.shape[0]):
            dbbox = resized_bboxes[i]
            dsize = (dbbox[2]-dbbox[0],dbbox[3]-dbbox[1])
            if dsize[0]<=1 or dsize[1]<=1:
                continue
            sub_mask = bwmli.crop_img_absolute_xy(mask[i],bboxes[i])
            sub_mask = np.ascontiguousarray(sub_mask)
            cur_m = cv2.resize(sub_mask,dsize=dsize,interpolation=cv2.INTER_NEAREST)
            bwmli.set_subimg(res_mask[i],cur_m,dbbox[:2])

        return self.new(res_mask),resized_bboxes

    @property
    def shape(self):
        return self.masks.shape
    
    def flip(self,direction=WBaseMask.HORIZONTAL):
        if len(self.masks)==0:
            return self
        
        if direction == WBaseMask.HORIZONTAL:
            masks = self.masks[:,:,::-1]
        elif direction == WBaseMask.VERTICAL:
            masks = self.masks[:,::-1,:]
        elif direction == WBaseMask.DIAGONAL:
            masks = self.masks[:,::-1,::-1]
        else:
            info = f"unknow flip direction {direction}"
            print(f"ERROR: {info}")
            raise RuntimeError(info)

        return self.new(masks)

    def pad(self, out_shape, pad_val=0):
        '''
        out_shape: [H,W]
        '''
        """padding has no effect on polygons`"""
        hp = max(out_shape[0]-self.height,0)
        wp = max(out_shape[1]-self.width,0)
        masks = np.pad(self.masks, [[0, 0], [0, hp], [0, wp]], constant_values=pad_val)
    
        return self.new(masks)

    def crop(self, bbox):
        '''
        bbox: [x0,y0,x1,y1]
        '''
        # clip the boundary
        bbox = bbox.copy()
        cropped_masks = bwmli.crop_masks_absolute_xy(self.masks,bbox)
        return self.new(cropped_masks)
    
    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks
        
    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            fill_val (int | float): Border value. Default 0 for masks.

        Returns:
            BitmapMasks: Rotated BitmapMasks.
        """
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = bwmli.imrotate(
                self.masks.transpose((1, 2, 0)),
                angle,
                center=center,
                scale=scale,
                border_value=fill_val)
            if rotated_masks.ndim == 2:
                # case when only one mask, (h, w)
                rotated_masks = rotated_masks[:, :, None]  # (h, w, 1)
            rotated_masks = rotated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return self.new(rotated_masks, height=out_shape[0],width=out_shape[1])
    
    def warp_affine(self,M,out_shape,fill_val=0):
        '''
        out_shape:[H,W]
        '''
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = bwmli.im_warp_affine(
                self.masks.transpose((1, 2, 0)),
                M=M,
                out_shape=out_shape[:2][::-1],
                border_value=fill_val)
            if rotated_masks.ndim == 2:
                # case when only one mask, (h, w)
                rotated_masks = rotated_masks[:, :, None]  # (h, w, 1)
            rotated_masks = rotated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return self.new(rotated_masks, height=out_shape[0],width=out_shape[1])

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        """Shear the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            BitmapMasks: The sheared masks.
        """
        if len(self.masks) == 0:
            sheared_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            sheared_masks = bwmli.imshear(
                self.masks.transpose((1, 2, 0)),
                magnitude,
                direction,
                border_value=border_value,
                interpolation=interpolation)
            if sheared_masks.ndim == 2:
                sheared_masks = sheared_masks[:, :, None]
            sheared_masks = sheared_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return self.new(sheared_masks, height=out_shape[0],width=out_shape[1])

    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  fill_val=0,
                  interpolation='bilinear'):
        """Translate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            fill_val (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            BitmapMasks: Translated BitmapMasks.

        Example:
            >>> from mmdet.core.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random(dtype=np.uint8)
            >>> out_shape = (32, 32)
            >>> offset = 4
            >>> direction = 'horizontal'
            >>> fill_val = 0
            >>> interpolation = 'bilinear'
            >>> # Note, There seem to be issues when:
            >>> # * out_shape is different than self's shape
            >>> # * the mask dtype is not supported by cv2.AffineWarp
            >>> new = self.translate(out_shape, offset, direction, fill_val,
            >>>                      interpolation)
            >>> assert len(new) == len(self)
            >>> assert new.height, new.width == out_shape
        """
        if len(self.masks) == 0:
            translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            translated_masks = bwmli.imtranslate(
                self.masks.transpose((1, 2, 0)),
                offset,
                direction,
                border_value=fill_val,
                interpolation=interpolation)
            if translated_masks.ndim == 2:
                translated_masks = translated_masks[:, :, None]
            translated_masks = translated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return self.new(translated_masks, height=out_shape[0],width=out_shape[1])

    @classmethod
    def concatenate(cls,masks):
        masks = np.concatenate([m.masks for m in masks],axis=0)
        return cls(masks)

    @classmethod
    def zeros(cls,*,width=None,height=None,shape=None):
        '''
        shape: [masks_nr,H,W]
        '''
        if shape is not None:
            masks = np.zeros(shape,dtype=np.uint8)
        else:
            masks = np.zeros([1,height,width],dtype=np.uint8)
        return cls(masks)

    def get_bboxes(self):
        masks = masks
        if len(masks) == 0:
            return np.zeros([0,4],dtype=np.float32)

        gtbboxes = []
        for i in range(masks.shape[0]):
            cur_mask = masks[i]
            idx = np.nonzero(cur_mask)
            xs = idx[1]
            ys = idx[0]
            if len(xs)==0:
                gtbboxes.append(np.zeros([4],dtype=np.float32))
            else:
                x0 = np.min(xs)
                y0 = np.min(ys)
                x1 = np.max(xs)
                y1 = np.max(ys)
                gtbboxes.append(np.array([x0,y0,x1,y1],dtype=np.float32))
        
        gtbboxes = np.array(gtbboxes)

        return gtbboxes

    def copy(self):
        return WBitmapMasks(self.masks.copy(),width=self.width,height=self.height)


    def check_consistency(self):
        pass

    def update_shape(self,*,width=None,height=None):
        return
        if height is not None:
            if self.shape[1] != height:
                print(f"Update WBitmapMasks height faild, {self.shape[1]} vs new {height}")
        if width is not None:
            if self.shape[2] != width:
                print(f"Update WBitmapMasks width faild, {self.shape[2]} vs new {width}")
