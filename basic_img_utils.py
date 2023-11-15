import numpy as np
from collections import OrderedDict, Iterable
import copy
import cv2


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
    pos: [x,y]
    '''
    x,y = pos
    dst_img[y:y+src_img.shape[0],x:x+src_img.shape[1]] = src_img
    return dst_img

def crop_and_past_img(dst_img,src_img,src_bbox,pos):
    '''
    src_box:xmin,ymin,xmax,ymax, absolute corrdinate
    pos: [x,y]
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