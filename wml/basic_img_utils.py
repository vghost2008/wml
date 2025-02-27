import numpy as np
from collections import OrderedDict
from collections.abc import Iterable
import copy
import cv2
import math
import wml.walgorithm as wa

BASE_IMG_SUFFIX=".jpg;;.jpeg;;.bmp;;.png;;.gif;;.tif"


def normal_image(image,min_v=0,max_v=255,dtype=np.uint8):

    if not isinstance(image,np.ndarray):
        image = np.array(image)

    t = image.dtype
    if t!=np.float32:
        image = image.astype(np.float32)

    i_min = np.min(image)
    i_max = np.max(image)
    image = (image-float(i_min))*float(max_v-min_v)/max(float(i_max-i_min),1e-8)+float(min_v)

    if dtype!=np.float32:
        image = image.astype(dtype)

    return image


def _get_translate_matrix(offset, direction='horizontal'):
    """Generate the translate matrix.

    Args:
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either
            "horizontal" or "vertical".

    Returns:
        ndarray: The translate matrix with dtype float32.
    """
    if direction == 'horizontal':
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif direction == 'vertical':
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix

def _get_shear_matrix(magnitude, direction='horizontal'):
    """Generate the shear matrix for transformation.

    Args:
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".

    Returns:
        ndarray: The shear matrix with dtype float32.
    """
    if direction == 'horizontal':
        shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
    elif direction == 'vertical':
        shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
    return shear_matrix

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

'''
box:ymin,xmin,ymax,xmax, absolute corrdinate
'''
def crop_img_absolute(img,box):
    shape = img.shape
    box = np.array(box)
    box[0:4:2] = np.minimum(box[0:4:2],shape[0])
    box[1:4:2] = np.minimum(box[1:4:2],shape[1])
    box = np.maximum(box,0)
    ymin = box[0]
    ymax = box[2]
    xmin = box[1]
    xmax = box[3]
    if len(shape)==2:
        return img[ymin:ymax,xmin:xmax]
    else:
        return img[ymin:ymax,xmin:xmax,:]
'''
box:xmin,ymin,xmax,ymax, absolute corrdinate
img:[H,W,C]
'''
def crop_img_absolute_xy(img,box):
    shape = img.shape
    box = np.array(box)
    box[0:4:2] = np.minimum(box[0:4:2],shape[1])
    box[1:4:2] = np.minimum(box[1:4:2],shape[0])
    box = np.maximum(box,0)
    ymin = box[1]
    ymax = box[3]
    xmin = box[0]
    xmax = box[2]
    return img[ymin:ymax,xmin:xmax]

'''
box:ymin,xmin,ymax,xmax, relative corrdinate
'''
def crop_img(img,box):
    shape = img.shape
    box = np.array(box)
    box = np.minimum(box,1.0)
    box = np.maximum(box,0.0)
    ymin = int((shape[0])*box[0]+0.5)
    ymax = int((shape[0])*box[2]+1+0.5)
    xmin = int((shape[1])*box[1]+0.5)
    xmax = int((shape[1])*box[3]+1+0.5)
    if len(shape)==2:
        return img[ymin:ymax,xmin:xmax]
    else:
        return img[ymin:ymax,xmin:xmax,:]


'''
box:xmin,ymin,xmax,ymax, absolute corrdinate
img: [B,C,H,W]
'''
def crop_batch_img_absolute_xy(img,box):
    shape = img.shape
    box = np.array(box)
    box[0:4:2] = np.minimum(box[0:4:2],shape[-1])
    box[1:4:2] = np.minimum(box[1:4:2],shape[-2])
    box = np.maximum(box,0)
    ymin = box[1]
    ymax = box[3]
    xmin = box[0]
    xmax = box[2]
    return img[:,:,ymin:ymax,xmin:xmax]

def set_subimg(img,sub_img,p0):
    '''
    p0:(x,y)
    '''
    img[p0[1]:p0[1]+sub_img.shape[0],p0[0]:p0[0]+sub_img.shape[1]] = sub_img

    return img

'''
box:xmin,ymin,xmax,ymax, absolute corrdinate
size: (w,h)
'''
def crop_and_pad(img,bbox,size=None,pad_color=127):
    if size is None:
        size = (bbox[2]-bbox[0],bbox[3]-bbox[1])
    img = crop_img_absolute_xy(img,bbox)
    channels = img.shape[-1]
    if img.shape[0]<size[1] or img.shape[1]<size[0]:
        res = np.ones([size[1],size[0],3],dtype=img.dtype)
        if not isinstance(pad_color,Iterable):
            pad_color = (pad_color,)*channels
        pad_color = np.array(list(pad_color),dtype=img.dtype)
        pad_color = pad_color.reshape([1,1,channels])
        res = res*pad_color
        offset_x = 0
        offset_y = 0

        w = img.shape[1]
        h = img.shape[0]
        res[offset_y:offset_y+h,offset_x:offset_x+w,:] = img
        return res
    else:
        return img

def align_pad(img,align=32,value=127):
    size = list(img.shape)
    size[0] = (size[0]+align-1)//align*align
    size[1] = (size[1]+align-1)//align*align

    res = np.ones([size[0],size[1],3],dtype=img.dtype)*value
    w = img.shape[1]
    h = img.shape[0]
    res[:h,:w,:] = img

    return res


'''
box:ymin,xmin,ymax,xmax, absolute corrdinate
mask: [NR,H,W]
'''
def crop_masks_absolute(masks,box):
    shape = masks.shape[1:]
    box = np.array(box)
    box[0:4:2] = np.minimum(box[0:4:2],shape[0])
    box[1:4:2] = np.minimum(box[1:4:2],shape[1])
    box = np.maximum(box,0)
    ymin = box[0]
    ymax = box[2]
    xmin = box[1]
    xmax = box[3]
    return masks[:,ymin:ymax,xmin:xmax]

'''
box:xmin,ymin,xmax,ymax, absolute corrdinate
mask: [NR,H,W]
'''
def crop_masks_absolute_xy(img,box):
    new_box = [box[1],box[0],box[3],box[2]]
    return crop_masks_absolute(img,new_box)

'''
img:[H,W]/[H,W,C]
rect:[ymin,xmin,ymax,xmax] absolute coordinate
与crop_img类似，但如果rect超出img边界会先pad再剪切
'''
def sub_image(img,rect,pad_value=127):
    if rect[0]<0 or rect[1]<0 or rect[2]>img.shape[0] or rect[3]>img.shape[1]:
        py0 = -rect[0] if rect[0]<0 else 0
        py1 = rect[2]-img.shape[0] if rect[2]>img.shape[0] else 0
        px0 = -rect[1] if rect[1] < 0 else 0
        px1 = rect[3] - img.shape[1] if rect[3] > img.shape[1] else 0
        img = np.pad(img,[[py0,py1],[px0,px1],[0,0]],constant_values=pad_value)
        rect[0] += py0
        rect[1] += px0
        rect[2] += py0
        rect[3] += px0

    return copy.deepcopy(img[rect[0]:rect[2],rect[1]:rect[3]])

'''
img:[H,W]/[H,W,C]
rect:[N,4] [ymin,xmin,ymax,xmax] absolute coordinate
'''
def sub_images(img,rects):
    res = []
    for rect in rects:
        res.append(sub_image(img,rect))

    return res
'''
img:[H,W]/[H,W,C]
rect:[xmin,ymin,xmax,ymax] absolute coordinate
'''
def sub_imagev2(img,rect,pad_value=127):
    return sub_image(img,[rect[1],rect[0],rect[3],rect[2]],pad_value=pad_value)

'''
img: [H,W,C]
size: [w,h]
'''
def center_crop(img,size,pad_value=127):
    cx = img.shape[1]//2
    cy = img.shape[0]//2
    x0 = cx-size[0]//2
    y0 = cy-size[1]//2
    x1 = x0+size[0]
    y1 = y0+size[1]
    return sub_image(img,[y0,x0,y1,x1],pad_value=pad_value)

def past_img(dst_img,src_img,pos):
    '''
    dst_img: [H,W,C]
    src_img: [h,w,C]
    pos: [x,y], 粘贴区域的左上角
    '''
    x,y = pos
    dst_img[y:y+src_img.shape[0],x:x+src_img.shape[1]] = src_img
    return dst_img

def crop_and_past_img(dst_img,src_img,src_bbox,pos):
    '''
    src_box:xmin,ymin,xmax,ymax, absolute corrdinate
    pos: [x,y], 粘贴区域的左上角
    '''
    src_img = crop_img_absolute_xy(src_img,src_bbox)
    return past_img(dst_img,src_img,pos)

def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             interpolation='bilinear',
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix, (w, h),
        flags=cv2_interp_codes[interpolation],
        borderValue=border_value)
    return rotated

def imtranslate(img,
                offset,
                direction='horizontal',
                border_value=0,
                interpolation='bilinear'):
    """Translate an image.

    Args:
        img (ndarray): Image to be translated with format
            (h, w) or (h, w, c).
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The translated image.
    """
    assert direction in ['horizontal',
                         'vertical'], f'Invalid direction: {direction}'
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        raise ValueError(
            f'Invalid type {type(border_value)} for `border_value`.')
    translate_matrix = _get_translate_matrix(offset, direction)
    translated = cv2.warpAffine(
        img,
        translate_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. translating masks whose channels
        # large than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation])
    return translated

def imshear(img,
            magnitude,
            direction='horizontal',
            border_value=0,
            interpolation='bilinear'):
    """Shear an image.

    Args:
        img (ndarray): Image to be sheared with format (h, w)
            or (h, w, c).
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The sheared image.
    """
    assert direction in ['horizontal',
                         'vertical'], f'Invalid direction: {direction}'
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        raise ValueError(
            f'Invalid type {type(border_value)} for `border_value`')
    shear_matrix = _get_shear_matrix(magnitude, direction)
    sheared = cv2.warpAffine(
        img,
        shear_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. shearing masks whose channels large
        # than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation])
    return sheared

def im_warp_affine(img,
             M,
             border_value=0,
             interpolation='bilinear',
             out_shape = None,
             ):
    '''
    out_shape:[W,H]
    '''
    if out_shape is None:
        h,w = img.shape[:2]
        out_shape = (w,h)
    rotated = cv2.warpAffine(
        img,
        M, out_shape,
        flags=cv2_interp_codes[interpolation],
        borderValue=border_value)
    return rotated

def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


'''
size:(w,h)
return:
 resized img, resized_img.size <= size
'''
def resize_img(img,size,keep_aspect_ratio=False,interpolation=cv2.INTER_LINEAR,align=None):

    img_shape = img.shape
    if size[0] == img.shape[1] and size[1]==img.shape[0]:
        return img

    if np.any(np.array(img_shape)==0):
        img_shape = list(img_shape)
        img_shape[0] = size[1]
        img_shape[1] = size[0]
        return np.zeros(img_shape,dtype=img.dtype)
    if keep_aspect_ratio:
        if size[1]*img_shape[1] != size[0]*img_shape[0]:
            if size[1]*img_shape[1]>size[0]*img_shape[0]:
                ratio = size[0]/img_shape[1]
            else:
                ratio = size[1]/img_shape[0]
            size = list(copy.deepcopy(size))
            size[0] = int(img_shape[1]*ratio)
            size[1] = int(img_shape[0]*ratio)

            if align:
                size[0] = (size[0]+align-1)//align*align
                size[1] = (size[1] + align - 1) // align * align

    if not isinstance(size,tuple):
        size = tuple(size)
    if size[0]==img_shape[0] and size[1]==img_shape[1]:
        return img

    img = cv2.resize(img,dsize=size,interpolation=interpolation)

    if len(img_shape)==3 and len(img.shape)==2:
        img = np.expand_dims(img,axis=-1)
    
    return img

def resize_imgv2(img,size,interpolation=cv2.INTER_LINEAR,return_scale=False,align=None):
    '''
    size: (w,h)
    '''
    old_shape = img.shape
    img = resize_img(img,size,keep_aspect_ratio=True,interpolation=interpolation)

    if return_scale:
        r = img.shape[0]/max(old_shape[0],1)

    if align is not None:
        img = align_pad(img,align=align)

    if return_scale:
        return img,r
    else:
        return img

def resize_imgv3(img,size,interpolation=cv2.INTER_LINEAR,return_scale=False,align=None,keep_aspect_ratio=True):
    '''
    size: (w,h)
    '''
    old_shape = img.shape
    img = resize_img(img,size,keep_aspect_ratio=keep_aspect_ratio,interpolation=interpolation)

    if return_scale:
        r = (img.shape[1]/max(old_shape[1],1),img.shape[0]/max(old_shape[0],1)) #(w,h) scale

    if align is not None:
        img = align_pad(img,align=align)

    if return_scale:
        return img,r
    else:
        return img

def resize_height(img,h,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_h = h
    new_w = int(shape[1]*new_h/shape[0])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def resize_width(img,w,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    new_w = w
    new_h = int(shape[0]*new_w/shape[1])
    return cv2.resize(img,dsize=(new_w,new_h),interpolation=interpolation)

def resize_short_size(img,size,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    if shape[0]<shape[1]:
        return resize_height(img,size,interpolation)
    else:
        return resize_width(img,size,interpolation)

def resize_long_size(img,size,interpolation=cv2.INTER_LINEAR):
    shape = img.shape
    if shape[0]>shape[1]:
        return resize_height(img,size,interpolation)
    else:
        return resize_width(img,size,interpolation)
'''
size:(w,h)
return:
img,r 
r = new_size/old_size
'''
def resize_and_pad(img,size,interpolation=cv2.INTER_LINEAR,pad_color=(0,0,0),center_pad=True,return_scale=False):
    old_shape = img.shape
    img = resize_img(img,size,keep_aspect_ratio=True,interpolation=interpolation)
    if return_scale:
        r = img.shape[0]/max(old_shape[0],1)
    if img.shape[0] == size[1] and img.shape[1] == size[0]:
        if return_scale:
            return img,r
        return img
    else:
        if len(img.shape)==3:
            channels = img.shape[-1]
            if not isinstance(pad_color,Iterable):
                pad_color = [pad_color]*channels
            res = np.ones([size[1],size[0],channels],dtype=img.dtype)
            pad_color = np.array(list(pad_color),dtype=img.dtype)
            pad_color = pad_color.reshape([1,1,channels])
        else:
            if not isinstance(pad_color,Iterable):
                pad_color = [pad_color]
            res = np.ones([size[1],size[0]],dtype=img.dtype)
            pad_color = np.array(list(pad_color),dtype=img.dtype)
            pad_color = pad_color.reshape([1,1])
        res = res*pad_color
        if center_pad:
            offset_x = (size[0]-img.shape[1])//2
            offset_y = (size[1]-img.shape[0])//2
        else:
            offset_x = 0
            offset_y = 0

        w = img.shape[1]
        h = img.shape[0]
        res[offset_y:offset_y+h,offset_x:offset_x+w] = img
        if return_scale:
            return res,r
        else:
            return res

def rotate_img(img,angle,scale=1.0,border_value=0,dsize=None,center=None,interpolation=cv2.INTER_LINEAR):
    if center is None:
        center = (img.shape[1]//2,img.shape[0]//2)
    if dsize is None:
        dsize=(img.shape[1],img.shape[0])
        M = cv2.getRotationMatrix2D(center,angle,scale)
    else:
        M = wa.getRotationMatrix2D(center,angle,scale,out_offset=(dsize[0]//2,dsize[1]//2))
    img = cv2.warpAffine(img,M,dsize,borderValue=border_value,flags=interpolation)
    return img

def rotate_img_file(filepath,angle,scale=1.0):
    img = cv2.imread(filepath)
    center = (img.shape[1]//2,img.shape[0]//2)
    M = cv2.getRotationMatrix2D(center,angle,scale)
    img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    cv2.imwrite(filepath,img)

'''
box:[ymin,xmin,ymax,xmax], relative coordinate
crop_size:[heigh,width] absolute pixel size.
'''
def crop_and_resize(img,box,crop_size):
    img = crop_img(img,box)
    return resize_img(img,crop_size)

'''
img:[H,W]/[H,W,C]
box:[N,4] ymin,xmin,ymax,xmax, relative corrdinate
从同一个图上切图
'''
def crop_and_resize_imgs(img,boxes,crop_size):
    res_imgs = []
    for box in boxes:
        sub_img = crop_and_resize(img,box,crop_size)
        res_imgs.append(sub_img)

    return np.stack(res_imgs,axis=0)
'''
img:[N,H,W]/[N,H,W,C]
box:[N,4] ymin,xmin,ymax,xmax, relative corrdinate
box 与 img一对一的进行切图
return:
[N]+crop_size
'''
def one_to_one_crop_and_resize_imgs(imgs,boxes,crop_size):
    res_imgs = []
    for i,box in enumerate(boxes):
        sub_img = crop_and_resize(imgs[i],box,crop_size)
        res_imgs.append(sub_img)

    return np.stack(res_imgs,axis=0)





'''
img:[H,W,C]
size:(w,h)
'''
CENTER_PAD=0
RANDOM_PAD=1
TOPLEFT_PAD=2
def pad_img(img,size,pad_value=127,pad_type=CENTER_PAD,return_pad_value=False):
    '''
    pad_type: 0, center pad
    pad_type: 1, random pad
    pad_type: 2, topleft_pad

    '''
    if pad_type==0:
        if img.shape[0]<size[1]:
            py0 = (size[1]-img.shape[0])//2
            py1 = size[1]-img.shape[0]-py0
        else:
            py0 = 0
            py1 = 0
        if img.shape[1]<size[0]:
            px0 = (size[0] - img.shape[1]) // 2
            px1 = size[0] - img.shape[1] - px0
        else:
            px0 = 0
            px1 = 0
    elif pad_type==1:
        if img.shape[0]<size[1]:
            py0 = random.randint(0,size[1]-img.shape[0])
            py1 = size[1]-img.shape[0]-py0
        else:
            py0 = 0
            py1 = 0
        if img.shape[1]<size[0]:
            px0 = random.randint(0,size[0]-img.shape[1])
            px1 = size[0] - img.shape[1] - px0
        else:
            px0 = 0
            px1 = 0
    elif pad_type==2:
        if img.shape[0]<size[1]:
            py0 = 0
            py1 = size[1]-img.shape[0]-py0
        else:
            py0 = 0
            py1 = 0
        if img.shape[1]<size[0]:
            px0 = 0
            px1 = size[0] - img.shape[1] - px0
        else:
            px0 = 0
            px1 = 0
    if len(img.shape)==3:
        img = np.pad(img, [[py0, py1], [px0, px1], [0, 0]], constant_values=pad_value)
    else:
        img = np.pad(img, [[py0, py1], [px0, px1]], constant_values=pad_value)
    
    if return_pad_value:
        return img,px0,px1,py0,py1
    return img

'''
img:[H,W,C]
size:(w,h)
'''
def pad_imgv2(img,size,pad_color=(0,0,0),center_pad=False):
    if img.shape[0] == size[1] and img.shape[1] == size[0]:
        return img
    else:
        res = np.ones([size[1],size[0],3],dtype=img.dtype)
        pad_color = np.array(list(pad_color),dtype=img.dtype)
        pad_color = pad_color.reshape([1,1,3])
        res = res*pad_color
        if center_pad:
            offset_x = (size[0]-img.shape[1])//2
            offset_y = (size[1]-img.shape[0])//2
        else:
            offset_x = 0
            offset_y = 0

        w = img.shape[1]
        h = img.shape[0]
        res[offset_y:offset_y+h,offset_x:offset_x+w,:] = img
        return res

def pad_imgv2(img,px0,px1,py0,py1,pad_value=127):
    if len(img.shape)==3:
        img = np.pad(img, [[py0, py1], [px0, px1], [0, 0]], constant_values=pad_value)
    else:
        img = np.pad(img, [[py0, py1], [px0, px1]], constant_values=pad_value)
    
    return img

'''
img:[H,W]/[H,W,C]
rect:[N,4] [xmin,ymin,xmax,ymax] absolute coordinate
'''
def sub_imagesv2(img,rects):
    res = []
    for rect in rects:
        res.append(sub_imagev2(img,rect))

    return res

def __get_discrete_palette(palette=[(0,(0,0,255)),(0.5,(255,255,255)),(1.0,(255,0,0))],nr=1000):
    res = np.zeros([nr,3],dtype=np.float32)
    pre_p = palette[0]
    for cur_p in palette[1:]:
        end_idx = min(math.ceil(cur_p[0]*nr),nr)
        beg_idx = min(max(math.floor(pre_p[0]*nr),0),end_idx)
        color0 = np.array(pre_p[1],dtype=np.float32)
        color1 = np.array(cur_p[1],dtype=np.float32)
        for i in range(beg_idx,end_idx):
            cur_color = (i-beg_idx)*(color1-color0)/(end_idx-beg_idx)+color0
            res[i] = cur_color
        pre_p = cur_p

    
    res = np.clip(res,0,255)
    res = res.astype(np.uint8)

    return res

def __get_discrete_img(img,nr=1000):
    img = img.astype(np.float32)*(nr-1)
    img = np.clip(img,0,nr-1)
    img = img.astype(np.int32)
    return img


def pseudocolor_img(img,palette=[(0,(0,0,255)),(0.5,(255,255,255)),(1.0,(255,0,0))],auto_norm=True):
    '''
    img: (H,W) #float, value in [0,1] if auto_norm is not True
    '''
    if auto_norm:
        img = normal_image(img,0.0,1.0,dtype=np.float32)
    color_nr = 256
    img = __get_discrete_img(img,nr=color_nr)
    palette = __get_discrete_palette(palette,nr=color_nr)
    H,W = img.shape
    img = np.reshape(img,[-1])
    new_img = palette[img]
    new_img = np.reshape(new_img,[H,W,3])

    return new_img

