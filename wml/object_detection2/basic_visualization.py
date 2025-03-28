#coding=utf-8
import numpy as np
from PIL import Image
import cv2

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

def draw_text_on_image(img,text,font_scale=1.2,color=(0.,255.,0.),pos=None,thickness=1):
    
    if len(img.shape)>=3 and img.shape[2]>3: #MCI
        if len(img.shape)==3:
            img = np.expand_dims(img,axis=-1)
            img = np.tile(img,[1,1,1,3])
        res = []
        for i in range(img.shape[2]):
            cur_img = np.ascontiguousarray(img[:,:,i])
            cur_img = draw_text_on_image(cur_img,
                              text=text,
                              font_scale=font_scale,
                              color=color,
                              pos=pos,
                              thickness=thickness)
            res.append(cur_img)
        res = np.stack(res,axis=2)
        return res

    if isinstance(text,bytes):
        text = str(text,encoding="utf-8")
    if not isinstance(text,str):
        text = str(text)
    thickness = 2
    size = cv2.getTextSize(text,fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=font_scale,thickness=thickness)
    if pos is None:
        pos = (0,(img.shape[0]+size[0][1])//2)
    elif isinstance(pos,str) and pos.lower() == "tl":
        text_size,_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_DUPLEX,fontScale=font_scale,thickness=thickness)
        tw,th = text_size
        pos = (0,th+5)
    elif isinstance(pos,str) and pos.lower() == "bl":
        text_size,_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_DUPLEX,fontScale=font_scale,thickness=thickness)
        tw,th = text_size
        pos = (0,img.shape[0]-th-5)
        
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, color=color, thickness=thickness)
    return img