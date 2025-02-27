import numpy as np
import copy
import cv2
from wml.semantic.mask_utils import get_bboxes_by_contours,npresize_mask
import wml.object_detection2.bboxes as odb
import wml.basic_img_utils as bwmli
import wml.object_detection2.bboxes as odb
from .common import WBaseMaskLike


class WMCKeypointsItem(WBaseMaskLike):
    '''
    每个WMCKeypointsItem共享同一个标签(label)
    '''
    def __init__(self,points,*,width=None,height=None):
        '''
        points:  [points_nr,2]
        '''
        self.points = points.copy()
        self.width = width
        self.height = height

    def copy(self):
        return WMCKeypointsItem(self.points,width=self.width,height=self.height)

    def bitmap(self,width=None,height=None):
        '''
        return: [H,W]
        '''
        raise RuntimeError(f"Not implement bitmap")

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
        ori_type = self.points.dtype
        points = self.points.astype(np.float32)*scale
        points = points.astype(ori_type)
        width = size[0]
        height = size[1]

        return WMCKeypointsItem(points=points,width=width,height=height)

    def flip(self,direction,width=None,height=None):
        if len(self.points)==0:
            return self
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        if direction == WBaseMaskLike.HORIZONTAL:
            self.points[:,0] = width-self.points[:,0]
        elif direction == WBaseMaskLike.VERTICAL:
            self.points[:,1] = height-self.points[:,1]
        elif direction == WBaseMaskLike.DIAGONAL:
            self.points[:,0] = width-self.points[:,0]
            self.points[:,1] = height-self.points[:,1]
        else:
            info = f"unknow flip direction {direction}"
            print(f"ERROR: {info}")
            raise RuntimeError(info)
        
        return self
    
    def crop(self, bbox):
        '''
        bbox: [x0,y0,x1,y1], 包含边界
        offset: [xoffset,yoffset]
        如果offset is not None, 先offset再用bbox crop
        ''' 
        if not isinstance(bbox,np.ndarray):
            bbox = np.array(bbox)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width-1)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height-1)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1+1, 1)
        h = np.maximum(y2 - y1+1, 1)

        if len(self.points)>0:
            keep = odb.is_points_in_bbox(self.points,bbox)
            points = self.points[keep]
            offset = np.reshape(np.array([x1,y1]),[1,2])
            points = points-offset
        else:
            points = np.zeros([0,2],dtype=np.int32)


        cropped_kps = WMCKeypointsItem(points, width=w,height=h)
        return cropped_kps

    def filter_out_of_range(self):
        bbox = [0,0,self.width-1,self.height-1]
        n_kps = self.crop(bbox)
        self.points = n_kps.points
        return self


    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        #out_shape: [h,w]
        """See :func:`BaseInstancepoints.rotate`."""
        if len(self.points) == 0:
            rotated_points = WMCKeypointsItem([], width=out_shape[1],height=out_shape[0])
        else:
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            coords = self.points.copy()
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coords = np.concatenate(
                (coords, np.ones((coords.shape[0], 1), coords.dtype)),
                axis=1)  # [n, 3]
            rotated_coords = np.matmul(
                rotate_matrix[None, :, :],
                coords[:, :, None])[..., 0]  # [n, 2, 1] -> [n, 2]
            rotated_points = WMCKeypointsItem(rotated_coords, width=out_shape[1],height=out_shape[0])
            rotated_points = rotated_points.crop(np.array([0,0,out_shape[1]-1,out_shape[0]-1]))
        return rotated_points

    def warp_affine(self,M,out_shape,fill_val=0):
        #out_shape: [h,w]
        """See :func:`BaseInstancepoints.rotate`."""
        if len(self.points) == 0:
            affined_points = WMCKeypointsItem([], width=out_shape[1],height=out_shape[0])
        else:
            coords = self.points.copy()
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coords = np.concatenate(
                (coords, np.ones((coords.shape[0], 1), coords.dtype)),
                axis=1)  # [n, 3]
            affined_coords = np.matmul(
                M[None, :, :],
                coords[:, :, None])[..., 0]  # [n, 2, 1] -> [n, 2]
            affined_points = WMCKeypointsItem(affined_coords, width=out_shape[1],height=out_shape[0])
            affined_points = affined_points.crop(np.array([0,0,out_shape[1]-1,out_shape[0]-1]))
        return affined_points

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        #out_shape: [h,w]
        if len(self.points) == 0:
            sheared_points = WMCKeypointsItem([], width=out_shape[1],height=out_shape[0])
        else:
            if direction == 'horizontal':
                shear_matrix = np.stack([[1, magnitude],
                                         [0, 1]]).astype(np.float32)
            elif direction == 'vertical':
                shear_matrix = np.stack([[1, 0], [magnitude,
                                                  1]]).astype(np.float32)
            p = self.points.copy()
            new_coords = np.matmul(shear_matrix, p.T)  # [2, n]
            new_coords[0, :] = np.clip(new_coords[0, :], 0,
                                       out_shape[1])
            new_coords[1, :] = np.clip(new_coords[1, :], 0,
                                       out_shape[0])
            sheared_points = new_coords.transpose((1, 0))
            sheared_points = WMCKeypointsItem(sheared_points, width=out_shape[1],height=out_shape[0])
            sheared_points.filter_out_of_range()
        return sheared_points

    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  fill_val=None,
                  interpolation=None):
        """Translate the Polygonpoints.

        Example:
            >>> self = Polygonpoints.random(dtype=np.int32)
            >>> out_shape = (self.height, self.width)
            >>> new = self.translate(out_shape, 4., direction='horizontal')
            >>> assert np.all(new.points[0][0][1::2] == self.points[0][0][1::2])
            >>> assert np.all(new.points[0][0][0::2] == self.points[0][0][0::2] + 4)  # noqa: E501
        """
        if len(self.points) == 0:
            res_points = WMCKeypointsItem([], width=out_shape[1],height=out_shape[0])
        else:
            p = self.points.copy()
            if direction == WBaseMaskLike.HORIZONTAL:
                p[:,0] = np.clip(p[:,0] + offset, 0, out_shape[1])
            elif direction == WBaseMaskLike.VERTICAL:
                p[:,1] = np.clip(p[:,1] + offset, 0, out_shape[0])
            else:
                info = f"error direction {direction}"
                print(f"ERROR: {type(self).__name__} {info}")
                raise RuntimeError(info)
            res_points = WMCKeypointsItem(p, width=out_shape[1],height=out_shape[0])
            res_points.filter_out_of_range()
        
        return res_points

    def offset(self,
               offset):
        '''
        offset: [xoffset,yoffset]
        '''
        w = self.width+offset[0] if self.width is not None else None
        h = self.height+offset[1] if self.height is not None else None
        offset = np.reshape(np.array(offset),[1,2])
        if len(self.points) == 0:
            res_masks = WMCKeypointsItem([], width=w,height=h)
        else:
            p = self.points.copy()+offset
            res_masks = WMCKeypointsItem(p, width=w,height=h)
            res_masks.filter_out_of_range()
        return res_masks

    def valid(self):
        if len(self.points) == 0:
            return False
        bbox = np.array([0,0,self.width,self.height])
        try:
            keep = odb.is_points_in_bbox(self.points,bbox)
        except Exception as e:
            print(e)
            pass
        self.points = self.points[keep]
        return len(self.points)>0

    def split2single_point(self):
        '''
        把多个点组成的Item拆分成由单个点组成的Item
        '''
        if len(self.points) <= 1:
            return [copy.deepcopy(self)]
        else:
            return [WMCKeypointsItem(np.expand_dims(p,axis=0),width=self.width,height=self.height) for p in self.points]

    @property
    def shape(self):
        return [len(self.points),self.height,self.width]
    
    def numel(self):
        return len(self.points)

    def _update_shape(self,*,width=None,height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        



class WMCKeypoints(WBaseMaskLike):
    '''
    每个WMCKeypoints包含多个WMCKeypointsItem, 每个Item与一个label相对应
    WMCKeypoints与多个标签相对应
    '''
    def __init__(self,points,*,width=None,height=None) -> None:
        assert width is not None, f"ERROR: width is None"
        assert height is not None, f"ERROR: height is None"
        super().__init__()
        n_points = []
        for p in points:
            if not isinstance(p,WMCKeypointsItem):
                n_points.append(WMCKeypointsItem(p,width=width,height=height))
            else:
                n_points.append(WMCKeypointsItem(p.points,width=width,height=height))
        self.points = copy.deepcopy(n_points)
        self.width = width
        self.height = height
        if self.width<5 or self.height<5:
            print(f"WARNING: {self.__class__.__name__}: unnormal mask size, width={self.width}, height={self.height}")

    @classmethod
    def zeros(cls,*,width=None,height=None,shape=None):
        '''
        shape: [points_nr,H,W]
        '''
        if shape is not None:
            width = shape[-1]
            height = shape[-2]
        return cls([],width=width,height=height)

    @classmethod
    def from_ndarray(cls,points,*,width=None,height=None):
        raise RuntimeError(f"Unimplement from ndarray")

    def copy(self):
        return WMCKeypoints(self.points,width=self.width,height=self.height)

    def __getitem__(self,idxs):
        if isinstance(idxs,(list,tuple)) and (len(idxs)==2 or len(idxs)==3) and isinstance(idxs[0],slice):
            sx = idxs[-1]
            sy = idxs[-2]
            if self.is_none_slice(sy) and self.is_flip_slice(sx):
                return self.flip(WBaseMaskLike.HORIZONTAL)
            elif self.is_none_slice(sx) and self.is_flip_slice(sy):
                return self.flip(WBaseMaskLike.VERTICAL)
            elif self.is_flip_slice(sx) and self.is_flip_slice(sy):
                return self.flip(WBaseMaskLike.DIAGONAL)
            bbox = self.slice2bbox(sx=sx,sy=sy)
            return self.crop(bbox)
        elif isinstance(idxs,(list,np.ndarray,tuple)):
            idxs = np.array(idxs)
            if idxs.dtype == bool:
                idxs = np.where(idxs)[0]
            try:
                points = [self.points[idx] for idx in idxs]
            except Exception as e:
                print(e)
                pass
            return WMCKeypoints(points,width=self.width,height=self.height)
        else:
            return self.points[idxs]

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
                self.points[i] = value[i]
        else:
            info = f"unknow idxs type {type(idxs)}"
            print(f"ERROR: {type(self).__name__}: {info}")
            raise RuntimeError(info)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_keypoints={len(self.points)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def bitmap(self,exclusion=None):
        raise RuntimeError(f"Not implement bitmap")

    def resize(self,size):
        '''
        size:[w,h]
        '''
        points = [m.resize(size,width=self.width,height=self.height) for m in self.points]
        width = size[0]
        height = size[1]
        return WMCKeypoints(points=points,width=width,height=height)

    def flip(self,direction=WBaseMaskLike.HORIZONTAL):
        [m.flip(direction,width=self.width,height=self.height) for m in self.points]
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
        points = [m.crop(bbox) for m in self.points]
        return WMCKeypoints(points,width=w,height=h)

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

    def copy_from(self,points,dst_bbox=None,src_bbox=None,update_size=False):
        if src_bbox is not None:
            points = points.crop(src_bbox)
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
            self.points = [m.offset(dst_bbox[:2]) for m in points]
        else:
            self.points = points
            if update_size:
                self.width = points.width
                self.height = points.height
        [p._update_shape(width=self.width,height=self.height) for p in self.points] 
        return self


    @classmethod
    def from_bitmap_masks(cls,bitmap_masks):
        raise RuntimeError(f"Not implement from bitmap masks")

    @staticmethod
    def concatenate(points):
        ws = [m.width for m in points]
        hs = [m.height for m in points]
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
        new_points = []
        for m in points:
            new_points.extend(m.points)
        return WMCKeypoints(new_points,width=nw,height=nh)

    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        #out_shape: [h,w]
        points = [m.rotate(out_shape, angle, center, scale, fill_val) for m in self.points]
        width = out_shape[1]
        height = out_shape[0]
        return WMCKeypoints(points,width=width,height=height)
    
    def warp_affine(self,M,out_shape,fill_val=0):
        points = [m.warp_affine(M,out_shape,fill_val) for m in self.points]
        width = out_shape[1]
        height = out_shape[0]
        return WMCKeypoints(points,width=width,height=height)
    

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        #out_shape: [h,w]
        points = [m.shear(out_shape, magnitude, direction, border_value, interpolation) for m in self.points]
        width = out_shape[1]
        height = out_shape[0]
        return WMCKeypoints(points,width=width,height=height)

    def translate(self,
                  out_shape,
                  offset,
                  direction=WBaseMaskLike.HORIZONTAL,
                  fill_val=None,
                  interpolation=None):
        '''
        out_shape: [H,W]
        '''
        points = [m.translate(out_shape, offset, direction, fill_val,interpolation) for m in self.points]
        width = out_shape[1]
        height = out_shape[0]
        return WMCKeypoints(points,width=width,height=height)
    
    def pad(self, out_shape, pad_val=0):
        '''
        out_shape: [H,W]
        '''
        """padding has no effect on polygons`"""
        return WMCKeypoints(self.points, height=out_shape[0],width=out_shape[1])

    @property
    def shape(self):
        return (len(self.points),self.height,self.width)

    def resize_mask_in_bboxes(self,bboxes,size=None,r=None):
        '''
        mask: [N,H,W]
        bboxes: [N,4](x0,y0,x1,y1)
        size: (new_w,new_h)
        '''
        return self.resize(size),None

    def to_ndarray(self):
        """See :func:`BaseInstancepoints.to_ndarray`."""
        return self.bitmap()

    def get_bboxes(self):
        gt_bboxes = [m.get_bbox() for m in self.points]
        if len(gt_bboxes) == 0:
            return np.zeros([0,4],dtype=np.float32)
        gt_bboxes = np.stack(gt_bboxes,axis=0)

        return gt_bboxes

    def valid(self):
        mask = [m.valid() for m in self.points]
        return mask

    @staticmethod
    def split2single_point(kps,labels):
        '''
        让每个WMCKeypointsItem仅包含一个点
        '''
        res_kps = []
        res_labels = []
        for kp,l in zip(kps,labels):
            n_kp = kp.split2single_point()
            l = [l]*len(n_kp)
            res_kps.extend(n_kp)
            res_labels.extend(l)

        if len(res_labels)>0 and isinstance(res_labels[0],(int,float)):
            res_labels = np.array(res_labels,dtype=np.int32)
        
        return WMCKeypoints(res_kps,width=kps.width,height=kps.height),res_labels

    @staticmethod
    def split2single_nppoint(kps,labels):
        points,labels = WMCKeypoints.split2single_point(kps,labels)
        res_points = []
        for p in points:
            res_points.append(p.points)
        if len(res_points)>0:
            res = np.concatenate(res_points,axis=0)
        else:
            res = np.zeros([0,2],dtype=np.float32)
            labels = np.zeros([0],dtype=np.int32)
        return res,labels

    def check_consistency(self):
        for kp in self.points:
            if kp.width != self.width or kp.height != self.height:
                info = f"Unmatch size WMCKeypoints shape {self.shape} vs WMCKeypointsItem shape {kp.shape}"
                print(info)
                raise RuntimeError(info)

    def update_shape(self,*,width=None,height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        for mask in self.points:
            mask._update_shape(width=width,height=height)
