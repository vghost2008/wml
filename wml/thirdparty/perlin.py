import imgaug.augmenters as iaa
import numpy as np
import torch
import math
from collections.abc import Iterable
from wml.semantic.basic_toolkit import npresize_mask

def align(v,a):
    return int(math.ceil(v/a)*a)

def generate_thr(img_shape, min=0, max=4):
    min_perlin_scale = min
    max_perlin_scale = max
    perlin_scalex = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scaley = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_noise_np = rand_perlin_2d_np((img_shape[-2], img_shape[-1]), (perlin_scalex, perlin_scaley))
    threshold = 0.5
    perlin_noise_np = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise_np)
    perlin_thr = np.where(perlin_noise_np > threshold, np.ones_like(perlin_noise_np), np.zeros_like(perlin_noise_np))
    return perlin_thr

def force_get_mask(img_shape,mask_fg):
    mask = np.zeros(img_shape)
    if mask_fg is None or len(mask_fg) < 4 or torch.max(mask_fg)<0.5:
        y = mask.shape[0]//2
        x = mask.shape[1]//2
        mask[y,x] = 1
        mask[y+1,x] = 1
        mask[y,x+1] = 1
        mask[y+1,x+1] = 1
        return mask
    else:
        mask_fg = mask_fg.cpu().numpy()
        fg = npresize_mask(np.expand_dims(mask_fg,axis=0),(img_shape[1],img_shape[0]))[0]
        ys,xs = np.where(fg)
        ys = ys[:4]
        xs = xs[:4]
        for y,x in zip(ys,xs):
            mask[y,x] = 1
        return mask
        


def perlin_mask(img_shape, min, max, mask_fg, flag=0):
    '''
    img_shape: [H,W]
    return: value is 0 or 1
    '''
    mask_ = np.zeros(img_shape)
    max_try_nr = 10
    while np.max(mask_) == 0 and max_try_nr>0:
        perlin_thr_1 = generate_thr(img_shape, min, max)
        temp = torch.rand(1).numpy()[0]
        if temp > 2 / 3:
            perlin_thr_2 = generate_thr(img_shape, min, max)
            perlin_thr = perlin_thr_1 + perlin_thr_2
            perlin_thr = np.where(perlin_thr > 0, np.ones_like(perlin_thr), np.zeros_like(perlin_thr))
        elif temp > 1 / 3:
            perlin_thr_2 = generate_thr(img_shape, min, max)
            perlin_thr = perlin_thr_1 * perlin_thr_2
        else:
            perlin_thr = perlin_thr_1
        if mask_fg is not None:
            perlin_thr = perlin_thr * mask_fg
        mask_ = perlin_thr
        max_try_nr -= 1
    if np.max(mask_) == 0 and max_try_nr<=0:
        print(f"Get perline mask timeout.")
        try:
            return force_get_mask(img_shape[-2:],mask_fg)
        except Exception as e:
            print(f"ERROR: force get mask faild, {e}")
    
    return mask_


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    '''
    fade:插值系数计算
    value = (1-fade(t))*start_point+fade(t)*end_point
    '''
    if shape[0]%res[0]!= 0 or shape[1]%res[1]!= 0:
        align_shape = [align(shape[0],res[0]),align(shape[1],res[1])]
        perlin_noise_np = _rand_perlin_2d_np(align_shape,res,fade) 
        perlin_noise_np = perlin_noise_np[:shape[0],:shape[1]]
        return perlin_noise_np
    return _rand_perlin_2d_np(shape,res,fade) 


def _rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1],
                                                  axis=1)
    dot = lambda grad, shift: (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])
