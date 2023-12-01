#coding=utf-8
import numpy as np
from PIL import Image

def convert_semantic_to_rgb(semantic,color_map,return_nparray=False):
    '''
    convert semantic label map to rgb PIL image or a np.ndarray
    Args:
        semantic: [H,W] label value
        color_map: list[int], [r0,g0,b0,r1,g1,b1,....]
    Returns:
        image: [H,W,3]
    '''
    new_mask = Image.fromarray(semantic.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_map)
    if return_nparray:
        return np.array(new_mask.convert('RGB'))
    return new_mask

def draw_semantic_on_image(image,semantic,color_map,alpha=0.4,ignored_label=0):
    '''
    draw semantic on image
    Args:
        image:
        semantic: [H,W] label value
        color_map: list[int], [r0,g0,b0,r1,g1,b1,....]
        alpha: mask percent
        ignored_label:
    Returns:
        return image*(1-alpha)+semantic+alpha
    '''
    mask = convert_semantic_to_rgb(semantic,color_map=color_map,return_nparray=True)
    new_img = image.astype(np.float32)*(1-alpha)+mask.astype(np.float32)*alpha
    new_img = np.clip(new_img,0,255).astype(np.uint8)
    pred = np.expand_dims(semantic!=ignored_label,axis=-1)
    new_img = np.where(pred,new_img,image)
    return new_img