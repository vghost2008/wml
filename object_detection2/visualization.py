#coding=utf-8
import cv2
import random
import numpy as np
import semantic.visualization_utils as smv
from PIL import Image
from iotoolkit.coco_data_fwd import JOINTS_PAIR as COCO_JOINTS_PAIR
from .basic_datadef import colors_tableau ,colors_tableau_large, PSEUDOCOLOR
from .basic_datadef import DEFAULT_COLOR_MAP as _DEFAULT_COLOR_MAP
import object_detection2.bboxes as odb

DEFAULT_COLOR_MAP = _DEFAULT_COLOR_MAP

def draw_text_on_image(img,text,font_scale=1.2,color=(0.,255.,0.),pos=None):
    if isinstance(text,bytes):
        text = str(text,encoding="utf-8")
    if not isinstance(text,str):
        text = str(text)
    thickness = 2
    size = cv2.getTextSize(text,fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=font_scale,thickness=thickness)
    if pos is None:
        pos = (0,(img.shape[0]+size[0][1])//2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, color=color, thickness=thickness)
    return img

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

def draw_bbox(img, bbox, shape=None, label=None, color=[255, 0, 0], thickness=2,is_relative_bbox=False,xy_order=True):
    if is_relative_bbox:
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    else:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
    if xy_order:
        p1 = p1[::-1]
        p2 = p2[::-1]
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    if label is not None:
        cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    return img

'''
pmin: (y0,x0)
pmax: (y1,x1)
'''
def get_text_pos_fn(pmin,pmax,bbox,label,text_size):
    text_width,text_height = text_size
    return (pmin[0]+text_height+5,pmin[1]+5)
    '''if bbox[0]<text_height:
        p1 = (pmax[0],pmin[1])
    else:
        p1 = pmin
    return (p1[0]-5,p1[1])'''

def get_text_pos_tr(pmin,pmax,bbox,label):
    p1 = (pmax[0],pmin[1])
    return (p1[0]-5,p1[1])

def get_text_pos_tm(pmin,pmax,bbox,label):
    p1 = ((pmin[0]+pmax[0])//2,pmin[1])
    return p1
def get_text_pos_br(pmin,pmax,bbox,label):
    p1 = (pmax[0],pmax[1])
    return (p1[0]-5,p1[1])

def random_color_fn(label):
    del label
    nr = len(colors_tableau)
    return colors_tableau[random.randint(0,nr-1)]

def fixed_color_fn(label):
    color_nr = len(colors_tableau)
    return colors_tableau[label%color_nr]

def fixed_color_large_fn(label,probs):
    if isinstance(label,(str,bytes)):
        return colors_tableau_large[len(label)]
    color_nr = len(colors_tableau_large)
    return colors_tableau_large[label%color_nr]

def pesudo_color_fn(label,probs):
    color_nr = len(PSEUDOCOLOR)
    idx = int(probs*color_nr)
    return PSEUDOCOLOR[idx%color_nr]

def red_color_fn(label):
    del label
    return (255,0,0)

def blue_color_fn(label):
    del label
    return (0,0,255)

def green_color_fn(label):
    del label
    return (0,255,0)

def default_text_fn(label,score):
    return str(label)

'''
bboxes: [N,4] (y0,x0,y1,x1)
color_fn: tuple(3) (*f)(label)
text_fn: str (*f)(label,score)
get_text_pos_fn: tuple(2) (*f)(lt_corner,br_corner,bboxes,label)
'''
def draw_bboxes(img, classes=None, scores=None, bboxes=None,
                        color_fn=fixed_color_large_fn,
                        text_fn=default_text_fn,
                        get_text_pos_fn=get_text_pos_fn,
                        thickness=2,show_text=True,font_scale=1.2,text_color=(0.,255.,0.),
                        is_relative_coordinate=True,
                        is_show_text=None,
                        fill_bboxes=False):
    if bboxes is None:
        return img

    bboxes = np.array(bboxes)
    if len(bboxes) == 0:
        return img
    if classes is None:
        classes = np.zeros([bboxes.shape[0]],dtype=np.int32)
    if is_relative_coordinate and np.any(bboxes>1.1):
        print(f"Use relative coordinate and max bboxes value is {np.max(bboxes)}")
    elif not is_relative_coordinate and np.all(bboxes<1.1):
        print(f"Use absolute coordinate and max bboxes value is {np.max(bboxes)}")

    bboxes_thickness = thickness if not fill_bboxes else -1
    if is_relative_coordinate:
        shape = img.shape
    else:
        shape = [1.0,1.0]
    if len(img.shape)<2:
        print(f"Error img size {img.shape}.")
        return img
    img = np.array(img)
    if scores is None:
        scores = np.ones([len(classes)],dtype=np.float32)
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    for i in range(bboxes.shape[0]):
        try:
            bbox = bboxes[i]
            if color_fn is not None:
                color = color_fn(classes[i],scores[i])
            else:
                color = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
            p10 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            cv2.rectangle(img, p10[::-1], p2[::-1], color, bboxes_thickness)
            if show_text and text_fn is not None:
                f_show_text = True
                if is_show_text is not None:
                    f_show_text = is_show_text(p10,p2)

                if f_show_text:
                    text_thickness = 1
                    s = text_fn(classes[i], scores[i])
                    text_size,_ = cv2.getTextSize(s,cv2.FONT_HERSHEY_DUPLEX,fontScale=font_scale,thickness=text_thickness)
                    p = get_text_pos_fn(p10,p2,bbox,classes[i],text_size)
                    cv2.putText(img, s, p[::-1], cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=font_scale,
                                color=text_color,
                                thickness=text_thickness)
        except Exception as e:
            bbox = bboxes[i]
            p10 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            if color_fn is not None:
                color = color_fn(classes[i])
            else:
                color = (random.random()*255, random.random()*255, random.random()*255)
            print("Error:",img.shape,shape,bboxes[i],classes[i],p10,p2,color,thickness,e)
            

    return img

def draw_bboxes_xy(img, classes=None, scores=None, bboxes=None,
                color_fn=fixed_color_large_fn,
                text_fn=default_text_fn,
                get_text_pos_fn=get_text_pos_fn,
                thickness=2,show_text=True,font_scale=1.2,text_color=(0.,255.,0.),
                is_relative_coordinate=False,
                is_show_text=None,
                fill_bboxes=False):
    if bboxes is not None:
        bboxes = odb.npchangexyorder(bboxes)
    return draw_bboxes(img,classes,scores=scores,bboxes=bboxes,color_fn=color_fn,
                       text_fn=text_fn,get_text_pos_fn=get_text_pos_fn,thickness=thickness,
                       show_text=show_text,font_scale=font_scale,text_color=text_color,
                       is_relative_coordinate=is_relative_coordinate,
                       is_show_text=is_show_text,
                       fill_bboxes=fill_bboxes)

def draw_legend(labels,text_fn,img_size,color_fn,thickness=4,font_scale=1.2,text_color=(0.,255.,0.),fill_bboxes=True):
    '''
    Generate a legend image
    Args:
        labels: list[int] labels
        text_fn: str fn(label) trans label to text
        img_size: (H,W) the legend image size, the legend is drawed in veritical direction
        color_fn: tuple(3) fn(label): trans label to RGB color
        thickness: text thickness
        font_scale: font size
        text_color: text color
    Returns:

    '''
    boxes_width = max(img_size[1]//3,20)
    boxes_height = img_size[0]/(2*len(labels))
    def lget_text_pos_fn(pmin, pmax, bbox, label):
        p1 = (pmax[0]+5, pmax[1]+5)
        return p1

    bboxes = []
    for i,l in enumerate(labels):
        xmin = 5
        xmax = xmin+boxes_width
        ymin = int((2*i+0.5)*boxes_height)
        ymax = ymin + boxes_height
        bboxes.append([ymin,xmin,ymax,xmax])
    img = np.ones([img_size[0],img_size[1],3],dtype=np.uint8)
    def _text_fn(x,_):
        return text_fn(x)
    return draw_bboxes(img,labels,bboxes=bboxes,color_fn=color_fn,text_fn=_text_fn,
                get_text_pos_fn=lget_text_pos_fn,
                thickness=thickness,
                show_text=True,
                font_scale=font_scale,
                text_color=text_color,
                is_relative_coordinate=False,
                fill_bboxes=fill_bboxes)



'''
img: [H,W,C]
mask only include the area within bbox
bboxes: [N,4](y0,x0,y1,x1)
mask: [N,h,w]
'''
def draw_mask(img,classes,bboxes,masks,
              color_fn=fixed_color_large_fn,
              is_relative_coordinate=True):
    masks = masks.astype(np.uint8)
    if is_relative_coordinate:
        scales = np.array([[img.shape[1],img.shape[0],img.shape[1],img.shape[0]]],dtype=np.float32)
        bboxes = bboxes*scales
    for i,bbox in enumerate(bboxes):
        if color_fn is not None:
            color = list(color_fn(classes[i]))
        else:
            color = [random.random()*255, random.random()*255, random.random()*255]
        color = np.reshape(np.array(color,dtype=np.float32),[1,1,-1])
        x = int(bbox[1])
        y = int(bbox[0])
        w = int((bbox[3]-bbox[1]))
        h = int((bbox[2]-bbox[0]))
        if w<=0 or h<=0:
            continue
        mask = masks[i]
        mask = cv2.resize(mask,(w,h))
        mask = np.expand_dims(mask,axis=-1)
        try:
            img[y:y+h,x:x+w,:] = (img[y:y+h,x:x+w,:]*(np.array([[[1]]],dtype=np.float32)-mask*0.4)).astype(np.uint8)+(mask*color*0.4).astype(np.uint8)
        except:
            pass

    return img

'''
mask only include the area within bbox
'''
def draw_mask_xy(img,classes,bboxes,masks,
              color_fn=fixed_color_large_fn,
              is_relative_coordinate=False):
    bboxes = odb.npchangexyorder(bboxes)
    img = draw_mask(img=img,
                    classes=classes,bboxes=bboxes,
                    masks=masks,color_fn=color_fn,
                    is_relative_coordinate=is_relative_coordinate)
    return img
'''
mask only include the area within bbox
'''
def draw_bboxes_and_mask(img,classes,scores,bboxes,masks,
                         color_fn=fixed_color_large_fn,
                         text_fn=default_text_fn,
                         thickness=4,
                         show_text=False,
                         font_scale=0.8,
                         is_relative_coordinate=False):
    masks = masks.astype(np.uint8)
    img = draw_mask(img=img,
                    classes=classes,bboxes=bboxes,
                    masks=masks,color_fn=color_fn,
                    is_relative_coordinate=is_relative_coordinate)
    img = draw_bboxes(img,classes,scores,bboxes,
                               color_fn=color_fn,
                               text_fn=text_fn,
                               thickness=thickness,
                               show_text=show_text,
                               fontScale=font_scale)
    return img

'''
img: [H,W,C]
mask: [N,H,W], include the area of whole image
bboxes: [N,4], [y0,x0,y1,x1]
'''
def draw_maskv2(img,classes,bboxes=None,masks=None,
                           color_fn=fixed_color_large_fn,
                           is_relative_coordinate=True,
                           ):
    if not isinstance(masks,np.ndarray):
        masks = np.array(masks)
    if is_relative_coordinate:
        scales = np.array([[img.shape[1],img.shape[0],img.shape[1],img.shape[0]]],dtype=np.float32)
        bboxes = bboxes*scales
    masks = masks.astype(np.uint8)
    if masks.shape[1] < img.shape[0] or masks.shape[2]<img.shape[1]:
        masks = np.pad(masks,[[0,0],[0,img.shape[0]-masks.shape[1]],[0,img.shape[1]-masks.shape[2]]])
    for i,bbox in enumerate(bboxes):
        if color_fn is not None:
            color = list(color_fn(classes[i]))
        else:
            color = [random.random()*255, random.random()*255, random.random()*255]
        x = int(bbox[1]*img.shape[1])
        y = int(bbox[0]*img.shape[0])
        w = int((bbox[3]-bbox[1])*img.shape[1])
        h = int((bbox[2]-bbox[0])*img.shape[0])
        if w<=0 or h<=0:
            continue
        mask = masks[i]
        img = smv.draw_mask_on_image_array(img,mask,color=color,alpha=0.4)
    
    return img

'''
mask include the area of whole image
'''
def draw_maskv2_xy(img,classes,bboxes=None,masks=None,
                           color_fn=fixed_color_large_fn,
                           is_relative_coordinate=False,
                           ):
    bboxes = odb.npchangexyorder(bboxes)
    img = draw_maskv2(img=img,
                    classes=classes,bboxes=bboxes,
                    masks=masks,color_fn=color_fn,
                    is_relative_coordinate=is_relative_coordinate)
    return img
'''
mask include the area of whole image
'''
def draw_bboxes_and_maskv2(img,classes,scores=None,bboxes=None,masks=None,
                           color_fn=fixed_color_large_fn,
                           text_fn=default_text_fn,
                           thickness=4,
                           show_text=False,
                           is_relative_coordinate=True,
                           font_scale=0.8):
    if not isinstance(masks,np.ndarray):
        masks = np.array(masks)
    masks = masks.astype(np.uint8)
    img = draw_maskv2(img=img,
                    classes=classes,bboxes=bboxes,
                    masks=masks,color_fn=color_fn,
                    is_relative_coordinate=is_relative_coordinate)

    img = draw_bboxes(img,classes,scores,bboxes,
                               color_fn=color_fn,
                               text_fn=text_fn,
                               thickness=thickness,
                               show_text=show_text,
                               is_relative_coordinate=is_relative_coordinate,
                               font_scale=font_scale)
    return img

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

def draw_heatmap_on_image(image,scores,color_pos=(255,0,0),color_neg=(0,0,255),alpha=0.4):
    '''
    draw semantic on image
    Args:
        image:
        scores: [H,W] scores value
        color_map: list[int], [r,g,b]
        alpha: mask percent
        ignored_label:
    Returns:
        return image*(1-alpha)+semantic+alpha
    '''

    color_pos = np.reshape(np.array(color_pos),[1,1,3])
    color_neg = np.reshape(np.array(color_neg),[1,1,3])
    color_pos = color_pos*np.ones_like(image).astype(np.float32)
    color_neg = color_neg*np.ones_like(image).astype(np.float32)
    scores = np.expand_dims(scores,axis=-1)*alpha
    color = color_pos*scores+color_neg*(1-scores)
    new_img = image.astype(np.float32)*(1-alpha)+color*alpha
    new_img = np.clip(new_img,0,255).astype(np.uint8)
    return new_img

def add_jointsv1(image, joints, color, r=5,line_thickness=2,no_line=False,joints_pair=None,left_node=None):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, line_thickness )

    # add link
    if not no_line and joints_pair is not None:
        for pair in joints_pair:
            link(pair[0], pair[1], color)

    # add joints
    node_color = None
    for i, joint in enumerate(joints):
        if left_node is None:
            node_color = colors_tableau[i]
        elif i in left_node:
            node_color = (0,255,0)
        else:
            node_color = (0,0,255)
        cv2.circle(image, (int(joint[0]), int(joint[1])), r, node_color, -1)

    return image

def add_jointsv2(image, joints, color, r=5,line_thickness=2,no_line=False,joints_pair=None,left_node=None):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        if jointa[2] > 0.01 and jointb[2] > 0.01:
            cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, line_thickness )

    # add link
    if not no_line and joints_pair is not None:
        for pair in joints_pair:
            link(pair[0], pair[1], color)

    # add joints
    for i, joint in enumerate(joints):
        if joint[2] > 0.05 and joint[0] > 1 and joint[1] > 1:
            if left_node is None:
                node_color = colors_tableau[i]
            elif i in left_node:
                node_color = (0,255,0)
            else:
                node_color = (0,0,255)
            cv2.circle(image, (int(joint[0]), int(joint[1])), r, node_color, -1)

    return image

def draw_keypoints(image, joints, color=[0,255,0],no_line=False,joints_pair=COCO_JOINTS_PAIR,left_node=list(range(1,17,2)),r=5,line_thickness=2):
    '''

    Args:
        image: [H,W,3]
        joints: [N,kps_nr,2] or [kps_nr,2]
        color:
        no_line:
        joints_pair: [[first idx,second idx],...]
    Returns:

    '''
    image = np.ascontiguousarray(image)
    joints = np.array(joints)
    if color is None:
        use_random_color=True
    else:
        use_random_color = False
    if len(joints.shape)==2:
        joints = [joints]
    else:
        assert len(joints.shape)==3,"keypoints need to be 3-dimensional."

    for person in joints:
        if use_random_color:
            color = np.random.randint(0, 255, size=3)
            color = [int(i) for i in color]

        if person.shape[-1] == 3:
            add_jointsv2(image, person, color=color,no_line=no_line,joints_pair=joints_pair,left_node=left_node,r=r,
                         line_thickness=line_thickness)
        else:
            add_jointsv1(image, person, color=color,no_line=no_line,joints_pair=joints_pair,left_node=left_node,r=r,
                         line_thickness=line_thickness)

    return image


def draw_keypoints_diff(image, joints0, joints1,color=[0,255,0]):
    image = np.ascontiguousarray(image)
    joints0 = np.array(joints0)
    joints1 = np.array(joints1)
    if color is None:
        use_random_color=True
    else:
        use_random_color = False
    if len(joints0.shape)==2:
        points_nr = joints0.shape[0]
        joints0 = [joints0]
        joints1 = [joints1]
    else:
        points_nr = joints0.shape[1]
        assert len(joints0.shape)==3,"keypoints need to be 3-dimensional."

    for person0,person1 in zip(joints0,joints1):
        if use_random_color:
            color = np.random.randint(0, 255, size=3)
            color = [int(i) for i in color]
        for i in range(points_nr):
            jointa = person0[i]
            jointb = person1[i]
            if person0.shape[-1] == 3:
                if person0[i][-1]>0.015 and person1[i][-1]>0.015:
                    cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                     color, 2 )
            else:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                     color, 2 )

    return image


def draw_points(img,points,classes=None,show_text=False,r=2,
                color_fn=fixed_color_large_fn,
                text_fn=default_text_fn,
                font_scale=0.8,
                thickness=2):
    '''
    img: [H,W,3]
    points: [N,2]/[N,3]
    classes:[N]/None
    color_fn: tuple(3) (*f)(label)
    text_fn: str (*f)(label,score)
    '''
    img = np.ascontiguousarray(img)
    nr = points.shape[0]
    if classes is None:
        classes = np.ones([nr],dtype=np.int32)
    if points.shape[1]>=3:
        scores = points[:,-1]
    else:
        scores = np.ones([nr],dtype=np.float32)
    for i,joint in enumerate(points):
        color = color_fn(classes[i])
        pos = (int(joint[0]), int(joint[1]))
        cv2.circle(img, pos, r, color, -1)
        if show_text:
            text = text_fn(classes[i],scores[i])
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, color=color, thickness=thickness)
    
    return img

'''
bboxes:[(ymin,xmin,ymax,xmax),....] value in range[0,1]
mask:[X,h,w]
size:[H,W]
'''
def get_fullsize_mask(boxes,masks,size,mask_bg_value=0):
    dtype = masks.dtype

    res_masks = []
    boxes = np.clip(boxes,0.0,1.0)
    for i,bbox in enumerate(boxes):
        x = int(bbox[1]*size[1])
        y = int(bbox[0]*size[0])
        w = int((bbox[3]-bbox[1])*size[1])
        h = int((bbox[2]-bbox[0])*size[0])
        res_mask = np.ones(size,dtype=dtype)*mask_bg_value
        if w>1 and h>1:
            mask = masks[i]
            mask = cv2.resize(mask,(w,h))
            sys.stdout.flush()
            res_mask[y:y+h,x:x+w] = mask
        res_masks.append(res_mask)

    if len(res_masks)==0:
        return np.zeros([0,size[0],size[1]],dtype=dtype)
    return np.stack(res_masks,axis=0)

def generate_mask_by_boxes(boxes,masks,mask_value=1):
    '''
    boxes:[N,4],[x0,y0,x1,y1]
    masks:[N,H,W]/[H,W]
    '''
    if len(masks.shape)==3:
        shape = masks.shape[1:]
    else:
        shape = masks.shape

    boxes[:,0:4:2] = np.clip(boxes[:,0:4:2],0.0,shape[1])
    boxes[:,1:4:2] = np.clip(boxes[:,1:4:2],0.0,shape[0])
    boxes = boxes.astype(np.int32)
    for i,bbox in enumerate(boxes):
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        if len(masks.shape)==3:
            masks[i,y0:y1,x0:x1] = mask_value
        else:
            masks[y0:y1,x0:x1] = mask_value

    return masks

def draw_polygon(img,polygon,color=(255,255,255),is_line=True,isClosed=True):
    if is_line:
        return cv2.polylines(img, [polygon], color=color,isClosed=isClosed)
    else:
        return cv2.fillPoly(img,[polygon],color=color)
