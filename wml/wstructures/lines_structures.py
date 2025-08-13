import numpy as np
import copy
import cv2
import wml.basic_img_utils as bwmli
import wml.object_detection2.bboxes as odb
from .common import WBaseMaskLike
from wml.walgorithm import *
from collections.abc import Iterable


class WMLinesItem(WBaseMaskLike):
    '''
    每个WMLinesItem共享同一个标签(label)
    '''
    def __init__(self,lines,*,width=None,height=None):
        '''
        lines:  [points_nr,4] (x0,y0,x1,y1)
        '''
        self.lines = lines.copy()
        self.width = width
        self.height = height

    def copy(self):
        return WMLinesItem(self.lines,width=self.width,height=self.height)

    def bitmap(self,sigma,width=None,height=None):
        '''
        return: [H,W]
        '''
        '''
        损失为一次方时：
        gt与pred相差sigma/2 loss为最大loss的20%, 相差sigma*0.75为29%, 相差sigma为38%, 相差sigma*2为68%，相差sigma*3为86%
        损失为平方时：
        gt与pred相差sigma/2 loss为最大loss的6%, 相差sigma*0.75为13%, 相差sigma为22%, 相差sigma*2为63%，相差sigma*3为90%
        损失为三次方时：
        gt与pred相差sigma/2 loss为最大loss的1.8%, 相差sigma*0.75为5.7%, 相差sigma为12.5%, 相差sigma*2为57%，相差sigma*3为90%
        损失为四次方时：
        gt与pred相差sigma/2 loss为最大loss的0.5%, 相差sigma*0.75为2.5%, 相差sigma为7%, 相差sigma*2为51%，相差sigma*3为90%
        output_res: [H,W]
        '''
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        x = list(range(width))
        y = list(range(height))
        x,y = np.meshgrid(x,y)

        p = np.stack([x,y],axis=-1)

        res = np.zeros([height,width],dtype=np.float32)

        for l in self.lines:
            d = point2line_distance(p,l[:2],l[2:])
            d2 = np.square(d)
            max_d = (6*sigma + 3)**2
            d2 = np.clip(d2,a_min=0,a_max=max_d)
            g = np.exp(-d2/(2 * sigma ** 2)) #[H,W]
            line_len_s = line_len(l)**2
            dp0 = point2point_distance_square(p,np.reshape(l[:2],[1,1,2]))
            dp1 = point2point_distance_square(p,np.reshape(l[2:],[1,1,2]))
            dp = np.maximum(dp0,dp1)
            mask0 = dp<line_len_s
            mask10 = dp0<sigma
            mask11 = dp1<sigma
            mask1 = np.logical_or(mask10,mask11)
            mask = np.logical_or(mask1,mask0).astype(np.float32)
            g = mask*g
            res = np.maximum(g,res)

        return res


    def resize(self,size,width=None,height=None):
        '''
        size:[w,h]
        '''
        if len(self.lines)==0:
            return self
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        w_scale = size[0]/width
        h_scale = size[1]/height
        scale = np.array([[w_scale,h_scale,w_scale,h_scale]],dtype=np.float32)
        ori_type = self.lines.dtype
        lines = self.lines.astype(np.float32)*scale
        lines = lines.astype(ori_type)
        width = size[0]
        height = size[1]

        return WMLinesItem(lines=lines,width=width,height=height)

    def flip(self,direction,width=None,height=None):
        if len(self.lines)==0:
            return self
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        if direction == WBaseMaskLike.HORIZONTAL:
            self.lines[:,0:4:2] = width-self.lines[:,0:4:2]
        elif direction == WBaseMaskLike.VERTICAL:
            self.lines[:,1:4:2] = height-self.lines[:,1:4:2]
        elif direction == WBaseMaskLike.DIAGONAL:
            self.lines[:,0:4:2] = width-self.lines[:,0:4:2]
            self.lines[:,1:4:2] = height-self.lines[:,1:4:2]
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

        if len(self.lines)>0:
            keep0 = odb.is_points_in_bbox(self.lines[:,:2],bbox)
            keep1 = odb.is_points_in_bbox(self.lines[:,2:],bbox)
            keep_a = np.logical_and(keep0,keep1)
            lines = self.lines
            rlines = []
            for i in range(len(lines)):
                if not keep_a[i]:
                    cur_line = odb.cut_line(bbox,lines[i])
                    if cur_line is not None:
                        rlines.append(cur_line)
                else:
                    rlines.append(lines[i])
            if len(rlines)>0:
                rlines = np.array(rlines)
                offset = np.reshape(np.array([x1,y1,x1,y1]),[1,4])
                rlines = rlines-offset
            else:
                rlines = np.zeros([0,4],dtype=np.int32)
        else:
            rlines = np.zeros([0,4],dtype=np.int32)


        cropped_kps = WMLinesItem(rlines, width=w,height=h)
        return cropped_kps

    def filter_out_of_range(self):
        bbox = [0,0,self.width-1,self.height-1]
        n_kps = self.crop(bbox)
        self.lines = n_kps.lines
        return self


    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        #out_shape: [h,w]
        """See :func:`BaseInstancepoints.rotate`."""
        if len(self.lines) == 0:
            rotated_points = WMLinesItem([], width=out_shape[1],height=out_shape[0])
        else:
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            coords = self.lines.copy()
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coords = np.concatenate([coords[:,:2],coords[:,2:]],axis=0)
            coords = np.concatenate(
                (coords, np.ones((coords.shape[0], 1), coords.dtype)),
                axis=1)  # [n, 3]
            rotated_coords = np.matmul(
                rotate_matrix[None, :, :],
                coords[:, :, None])[..., 0]  # [n, 2, 1] -> [n, 2]
            rotated_coords = np.split(rotated_coords,indices_or_sections=2,axis=0)
            rotated_coords = np.concatenate(rotated_coords,axis=1)
            rotated_points = WMLinesItem(rotated_coords, width=out_shape[1],height=out_shape[0])
            rotated_points = rotated_points.crop(np.array([0,0,out_shape[1]-1,out_shape[0]-1]))
        return rotated_points

    def warp_affine(self,M,out_shape,fill_val=0):
        #out_shape: [h,w]
        """See :func:`BaseInstancepoints.rotate`."""
        if len(self.lines) == 0:
            affined_points = WMLinesItem([], width=out_shape[1],height=out_shape[0])
        else:
            coords = self.lines.copy()
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coords = np.concatenate([coords[:,:2],coords[:,2:]],axis=0)
            coords = np.concatenate(
                (coords, np.ones((coords.shape[0], 1), coords.dtype)),
                axis=1)  # [n, 3]
            affined_coords = np.matmul(
                M[None, :, :],
                coords[:, :, None])[..., 0]  # [n, 2, 1] -> [n, 2]
            affined_coords = np.split(affined_coords,indices_or_sections=2,axis=0)
            affined_coords = np.concatenate(affined_coords,axis=1)
            affined_points = WMLinesItem(affined_coords, width=out_shape[1],height=out_shape[0])
            affined_points = affined_points.crop(np.array([0,0,out_shape[1]-1,out_shape[0]-1]))
        return affined_points

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        #out_shape: [h,w]
        if len(self.lines) == 0:
            sheared_points = WMLinesItem([], width=out_shape[1],height=out_shape[0])
        else:
            if direction == 'horizontal':
                shear_matrix = np.stack([[1, magnitude],
                                         [0, 1]]).astype(np.float32)
            elif direction == 'vertical':
                shear_matrix = np.stack([[1, 0], [magnitude,
                                                  1]]).astype(np.float32)
            coords = self.lines.copy()
            coords = np.concatenate([coords[:,:2],coords[:,2:]],axis=0)
            new_coords = np.matmul(shear_matrix, coords.T)  # [2, n]
            new_coords[0, :] = np.clip(new_coords[0, :], 0,
                                       out_shape[1])
            new_coords[1, :] = np.clip(new_coords[1, :], 0,
                                       out_shape[0])
            sheared_points = new_coords.transpose((1, 0))
            sheared_points = np.split(sheared_points,indices_or_sections=2,axis=0)
            sheared_points = np.concatenate(sheared_points,axis=1)
            sheared_points = WMLinesItem(sheared_points, width=out_shape[1],height=out_shape[0])
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
            >>> assert np.all(new.lines[0][0][1::2] == self.lines[0][0][1::2])
            >>> assert np.all(new.lines[0][0][0::2] == self.lines[0][0][0::2] + 4)  # noqa: E501
        """
        if len(self.lines) == 0:
            res_points = WMLinesItem([], width=out_shape[1],height=out_shape[0])
        else:
            p = self.lines.copy()
            if direction == WBaseMaskLike.HORIZONTAL:
                p[:,0:4:2] = p[:,0:4:2] + offset
            elif direction == WBaseMaskLike.VERTICAL:
                p[:,1:4:2] = p[:,1:4:2] + offset
            else:
                info = f"error direction {direction}"
                print(f"ERROR: {type(self).__name__} {info}")
                raise RuntimeError(info)
            res_points = WMLinesItem(p, width=out_shape[1],height=out_shape[0])
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
        offset = np.concatenate([offset,offset],axis=1)
        if len(self.lines) == 0:
            res_masks = WMLinesItem([], width=w,height=h)
        else:
            p = self.lines.copy()+offset
            res_masks = WMLinesItem(p, width=w,height=h)
            res_masks.filter_out_of_range()
        return res_masks

    def valid(self):
        if len(self.lines) == 0:
            return False
        #bbox = np.array([0,0,self.width,self.height])
        self.filter_out_of_range()
        return len(self.lines)>0

    def split2single_line(self):
        '''
        把多个点组成的Item拆分成由单个线组成的Item
        '''
        if len(self.lines) <= 1:
            return [copy.deepcopy(self)]
        else:
            return [WMLinesItem(np.expand_dims(p,axis=0),width=self.width,height=self.height) for p in self.lines]

    @property
    def shape(self):
        return [len(self.lines),self.height,self.width]
    
    def numel(self):
        return len(self.lines)

    def _update_shape(self,*,width=None,height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

    def __len__(self):
        return len(self.lines)
        



class WMLines(WBaseMaskLike):
    '''
    每个WMLines包含多个WMLinesItem, 每个Item与一个label相对应
    WMLines与多个标签相对应
    '''
    def __init__(self,lines,*,width=None,height=None) -> None:
        assert width is not None, f"ERROR: width is None"
        assert height is not None, f"ERROR: height is None"
        super().__init__()
        n_lines = []
        for p in lines:
            if not isinstance(p,WMLinesItem):
                n_lines.append(WMLinesItem(p,width=width,height=height))
            else:
                n_lines.append(WMLinesItem(p.lines,width=width,height=height))
        self.lines = copy.deepcopy(n_lines)
        self.width = width
        self.height = height
        if self.width<5 or self.height<5:
            print(f"WARNING: {self.__class__.__name__}: unnormal lines size, width={self.width}, height={self.height}")

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
    def from_ndarray(cls,lines,*,width=None,height=None):
        raise RuntimeError(f"Unimplement from ndarray")

    def copy(self):
        return WMLines(self.lines,width=self.width,height=self.height)

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
                lines = [self.lines[idx] for idx in idxs]
            except Exception as e:
                print(e)
                pass
            return WMLines(lines,width=self.width,height=self.height)
        else:
            return self.lines[idxs]

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
                self.lines[i] = value[i]
        else:
            info = f"unknow idxs type {type(idxs)}"
            print(f"ERROR: {type(self).__name__}: {info}")
            raise RuntimeError(info)

    def __len__(self):
        return len(self.lines)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_lines={len(self.lines)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def bitmap(self,num_classes,labels,sigma=1,width=None,height=None,exclusion=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        res = np.zeros([num_classes,height,width])

        for l,line in zip(labels,self.lines):
            g = line.bitmap(sigma=sigma,width=width,height=height)
            res[l] = np.maximum(res[l],g)
        
        return res
        

    def resize(self,size):
        '''
        size:[w,h]
        '''
        lines = [m.resize(size,width=self.width,height=self.height) for m in self.lines]
        width = size[0]
        height = size[1]
        return WMLines(lines=lines,width=width,height=height)

    def flip(self,direction=WBaseMaskLike.HORIZONTAL):
        [m.flip(direction,width=self.width,height=self.height) for m in self.lines]
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
        lines = [m.crop(bbox) for m in self.lines]
        return WMLines(lines,width=w,height=h)

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

    def copy_from(self,lines,dst_bbox=None,src_bbox=None,update_size=False):
        if len(self) != 0:
            print(f"ERROR: MLines copy_from only support copy to empty target.")
        if src_bbox is not None:
            lines = lines.crop(src_bbox)
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
            self.lines = [m.offset(dst_bbox[:2]) for m in lines]
        else:
            self.lines = lines
            if update_size:
                self.width = lines.width
                self.height = lines.height
        [p._update_shape(width=self.width,height=self.height) for p in self.lines] 
        return self


    @classmethod
    def from_bitmap_masks(cls,bitmap_masks):
        raise RuntimeError(f"Not implement from bitmap masks")

    @staticmethod
    def concatenate(lines):
        ws = [m.width for m in lines]
        hs = [m.height for m in lines]
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
        new_lines = []
        for m in lines:
            new_lines.extend(m.lines)
        return WMLines(new_lines,width=nw,height=nh)

    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        #out_shape: [h,w]
        lines = [m.rotate(out_shape, angle, center, scale, fill_val) for m in self.lines]
        width = out_shape[1]
        height = out_shape[0]
        return WMLines(lines,width=width,height=height)
    
    def warp_affine(self,M,out_shape,fill_val=0):
        lines = [m.warp_affine(M,out_shape,fill_val) for m in self.lines]
        width = out_shape[1]
        height = out_shape[0]
        return WMLines(lines,width=width,height=height)
    

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        #out_shape: [h,w]
        lines = [m.shear(out_shape, magnitude, direction, border_value, interpolation) for m in self.lines]
        width = out_shape[1]
        height = out_shape[0]
        return WMLines(lines,width=width,height=height)

    def translate(self,
                  out_shape,
                  offset,
                  direction=WBaseMaskLike.HORIZONTAL,
                  fill_val=None,
                  interpolation=None):
        '''
        out_shape: [H,W]
        '''
        lines = [m.translate(out_shape, offset, direction, fill_val,interpolation) for m in self.lines]
        width = out_shape[1]
        height = out_shape[0]
        return WMLines(lines,width=width,height=height)
    
    def pad(self, out_shape, pad_val=0):
        '''
        out_shape: [H,W]
        '''
        """padding has no effect on polygons`"""
        return WMLines(self.lines, height=out_shape[0],width=out_shape[1])

    @property
    def shape(self):
        return (len(self.lines),self.height,self.width)

    def resize_mask_in_bboxes(self,bboxes,size=None,r=None):
        '''
        mask: [N,H,W]
        bboxes: [N,4](x0,y0,x1,y1)
        size: (new_w,new_h)
        '''
        return self.resize(size),None

    def to_ndarray(self):
        """See :func:`BaseInstancelines.to_ndarray`."""
        return self.bitmap()

    def get_bboxes(self):
        gt_bboxes = [m.get_bbox() for m in self.lines]
        if len(gt_bboxes) == 0:
            return np.zeros([0,4],dtype=np.float32)
        gt_bboxes = np.stack(gt_bboxes,axis=0)

        return gt_bboxes

    def valid(self):
        mask = [m.valid() for m in self.lines]
        return mask

    @staticmethod
    def split2single_line(kps,labels):
        '''
        让每个WMLinesItem仅包含一个点
        '''
        res_kps = []
        res_labels = []
        for kp,l in zip(kps,labels):
            n_kp = kp.split2single_line()
            l = [l]*len(n_kp)
            res_kps.extend(n_kp)
            res_labels.extend(l)

        if len(res_labels)>0 and isinstance(res_labels[0],(int,float)):
            res_labels = np.array(res_labels,dtype=np.int32)
        
        return WMLines(res_kps,width=kps.width,height=kps.height),res_labels

    @staticmethod
    def split2single_nppoint(kps,labels):
        lines,labels = WMLines.split2single_line(kps,labels)
        res_lines = []
        for p in lines:
            res_lines.append(p.lines)
        if len(res_lines)>0:
            res = np.concatenate(res_lines,axis=0)
        else:
            res = np.zeros([0,4],dtype=np.float32)
            labels = np.zeros([0],dtype=np.int32)
        return res,labels

    def check_consistency(self):
        for kp in self.lines:
            if kp.width != self.width or kp.height != self.height:
                info = f"Unmatch size WMLines shape {self.shape} vs WMLinesItem shape {kp.shape}"
                print(info)
                raise RuntimeError(info)

    def update_shape(self,*,width=None,height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        for mask in self.lines:
            mask._update_shape(width=width,height=height)


class MultiClassesLineHeatmapGenerator:
    def __init__(self,num_classes,output_res=None,sigma=1):
        self.set_output_res(output_res)
        self.num_classes = num_classes
        self.sigma = sigma

    def set_output_res(self,output_res):
        if output_res is None:
            self.output_res = [None,None]
            return
        if not isinstance(output_res,Iterable):
            output_res = (output_res,output_res)
        self.output_res = output_res


    def __call__(self, lines,labels,output_res=None):
        if output_res is not None:
            self.set_output_res(output_res)
        
        return lines.bitmap(num_classes=self.num_classes,labels=labels,sigma=self.sigma,height=self.output_res[0],width=self.output_res[1])
    
