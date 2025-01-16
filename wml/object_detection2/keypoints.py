import cv2
import numpy as np
import math

def npget_bbox(keypoints,threshold=0.02):
    '''

    Args:
        keypoints: [N,3] or [N,2] ,[x,y,visible]
        threshold:

    Returns:
        [xmin,ymin,xmax,ymax]

    '''
    assert len(keypoints.shape)==2,f"ERROR kps shape {keypoints.shape}"

    if keypoints.shape[-1]>=3:
        mask = keypoints[:,2]>threshold
        if np.any(mask):
            keypoints = keypoints[mask]
        else:
            return None
    xmin = np.min(keypoints[:,0])
    xmax = np.max(keypoints[:,0])
    ymin = np.min(keypoints[:,1])
    ymax = np.max(keypoints[:,1])
    return np.array([xmin,ymin,xmax,ymax],dtype=np.float32)

def expand_bbox_by_kps(bbox,keypoints,threshold=0.02):
    '''

    Args:
        bbox: [xmin,ymin,xmax,ymax]
        keypoints: [[x,y,score],...]
        threshold:

    Returns:
        [xmin,ymin,xmax,ymax]

    '''
    kp_bbox = npget_bbox(keypoints,threshold=threshold)
    if kp_bbox is None:
        return bbox
    xmin,ymin,xmax,ymax = bbox
    xmin = np.minimum(xmin,kp_bbox[0])
    ymin = np.minimum(ymin,kp_bbox[1])
    xmax = np.maximum(xmax,kp_bbox[2])
    ymax = np.maximum(ymax,kp_bbox[3])

    return np.array([xmin,ymin,xmax,ymax],dtype=np.float32)

def expand_yxyx_bbox_by_kps(bbox,keypoints,threshold=0.02):
    '''

    Args:
        bbox: [ymin,xmin,ymax,xmax]
        keypoints: [[x,y,score],...]
        threshold:

    Returns:
        [ymin,xmin,ymax,xmax]

    '''
    kp_bbox = npget_bbox(keypoints, threshold=threshold)
    if kp_bbox is None:
        return bbox
    ymin, xmin, ymax, xmax = bbox
    xmin = np.minimum(xmin, kp_bbox[0])
    ymin = np.minimum(ymin, kp_bbox[1])
    xmax = np.maximum(xmax, kp_bbox[2])
    ymax = np.maximum(ymax, kp_bbox[3])

    return np.array([ymin, xmin, ymax, xmax], dtype=np.float32)


def npbatchget_bboxes(keypoints,threshold=0.02):
    if not isinstance(keypoints,np.ndarray):
        keypoints = np.array(keypoints)

    if len(keypoints.shape)==2:
        return npget_bbox(keypoints,threshold)

    bboxes = []
    for kps in keypoints:
        bboxes.append(npget_bbox(kps,threshold))
    return np.array(bboxes)

def keypoint_distance(kp0,kp1,use_score=True,score_threshold=0.1,max_dis=1e8):
    '''

    Args:
        kp0: [NP_NR,2/3] (x,y) or (x,y,score)
        kp1: [NP_NR,2/3] (x,y) or (x,y,score)
        use_score:
        score_threshold:
        max_dis:

    Returns:
    '''
    KP_NR = kp0.shape[0]
    NR_THRESHOLD = KP_NR/4
    def point_dis(p0,p1):
        dx = p0[0]-p1[0]
        dy = p0[1]-p1[1]
        return math.sqrt(dx*dx+dy*dy)

    if use_score:
        count_nr = 0
        sum_dis = 0.0
        for i in range(KP_NR):
            if kp0[i][2]>score_threshold and kp1[i][2]>score_threshold:
                count_nr += 1
                sum_dis += point_dis(kp0[i],kp1[i])
        if count_nr<=NR_THRESHOLD:
            return max_dis
        else:
            return sum_dis/count_nr
    else:
        sum_dis = 0.0
        for i in range(KP_NR):
            sum_dis += point_dis(kp0[i],kp1[i])

        return sum_dis/KP_NR

def keypoint_distancev2(kp0,kp1,bbox0,bbox1,use_score=True,score_threshold=0.1,max_dis=1e8):
    dis = keypoint_distance(kp0,kp1,use_score,score_threshold,max_dis)
    bboxes = np.stack([bbox0,bbox1],axis=0)
    hw = bboxes[...,2:]-bboxes[...,:2]
    size = np.maximum(1e-8,hw[...,0]*hw[...,1])
    size = np.sqrt(size)
    size = np.mean(size)
    return dis/size


def keypoints_distance(kps,use_score=True,score_threshold=0.1,max_dis=1e8):
    '''

    Args:
        kps: [N,KP_NR,3/2] (x,y) or (x,y,score)
        use_score:
        score_threshold:

    Returns:

    '''

    N = kps.shape[0]
    res = np.zeros([N],dtype=np.float32)

    for i in range(1,N):
        res[i] = keypoint_distance(kps[i-1],kps[i],use_score,score_threshold,max_dis)

    return res

def keypoints_distancev2(kps,bboxes,use_score=True,score_threshold=0.1,max_dis=1e8):
    '''

    Args:
        kps: [N,KP_NR,2/3]
        bboxes: [N,4] [ymin,xmin,ymax,xmax]
        use_score:
        score_threshold:
        max_dis:

    Returns:

    '''
    dis = keypoints_distance(kps,use_score,score_threshold,max_dis)

    hw = bboxes[...,2:]-bboxes[...,:2]
    size = np.maximum(1e-8,hw[...,0]*hw[...,1])
    size = np.sqrt(size)
    dis = dis/size
    return dis

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def rotate(angle,img,kps,bbox,scale=1.0):
    '''

    Args:
        angle: [-360,360]
        img: [RGB]
        kps: [N,2]/[N,3]
        bbox: [xmin,ymin,xmax,ymax]

    Returns:

    '''
    cx = (bbox[0]+bbox[2])/2
    cy = (bbox[1] + bbox[3]) / 2
    matrix = cv2.getRotationMatrix2D([cx,cy],angle,scale)
    img = cv2.warpAffine(img,matrix,dsize=(img.shape[1],img.shape[0]),
                         flags=cv2.INTER_LINEAR)
    num_joints = kps.shape[0]

    def is_good_point(p):
        return p[0]>=0 and p[1]>=0 and p[0]<img.shape[1] and p[1]<img.shape[0]

    for i in range(num_joints):
        if kps[i, 2] > 0.0:
            kps[i, 0:2] = affine_transform(kps[i, 0:2], matrix)
            if not is_good_point(kps[i]):
                kps[i,:] = 0
    points = [bbox[:2],[bbox[0],bbox[3]],bbox[2:],[bbox[2],bbox[1]]]
    res_points = []
    for p in points:
        p = affine_transform(np.array(p),matrix)
        res_points.append(p)
    res_points = np.array(res_points)
    x0y0 = np.maximum(np.min(res_points,axis=0),0)
    x1y1 = np.minimum(np.max(res_points,axis=0),[img.shape[1]-1,img.shape[0]-1])
    bbox = np.concatenate([x0y0,x1y1],axis=0)

    return img,kps,bbox

def fliplr_joints(joints, width, matched_parts,joints_vis):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        if joints_vis is not None:
            joints_vis[pair[0], :], joints_vis[pair[1], :] = \
                joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints,joints_vis

def flip(img,kps,joints_vis,matched_parts,bbox=None):
    '''

    Args:
        img:
        kps: [N,3] x,y,vis
        joints_vis: [N,3]
        matched_parts: [[id0_0,id0_1],...]
        bboxes: [xmin,ymin,xmax,ymax]

    Returns:

    '''
    img = img[:, ::-1, :]
    joints, joints_vis = fliplr_joints(
                    kps,img.shape[1], matched_parts,joints_vis)
    if bbox is not None:
        xmax = img.shape[1] - bbox[0] - 1
        bbox[0] = img.shape[1] - bbox[2] - 1
        bbox[2] = xmax

    return img,joints,joints_vis,bbox


def cut2size(kps,bbox,dst_size):
    '''
    bbox:[xmin,ymin,xmax,ymax]
    dst_size:[W,H]
    kps:[N,3]/[N,2]
    '''
    kps[...,:2] = kps[...,:2]-np.array([[bbox[0],bbox[1]]],dtype=np.float32)
    bbox_w = bbox[2]-bbox[0]
    bbox_h = bbox[3]-bbox[1]
    if bbox_w<1e-8 or bbox_h<1e-8:
        print(f"object_detection2.keypoints: ERROR bbox in cut2size.")
        bbox_w = max(1.0,bbox_w)
        bbox_h = max(1.0,bbox_h)
    kps[...,:2] = kps[...,:2]*np.array([[dst_size[0]/bbox_w,dst_size[1]/bbox_h]],dtype=np.float32)
    return kps


def mckps_distance_matrix(kps0,kps1):
    '''
    kps0: [N,2]
    kps1: [M,2]
    return:
    [N,M]
    '''
    kps0 = np.expand_dims(kps0,axis=1)
    kps1 = np.expand_dims(kps1,axis=0)
    delta = kps0-kps1
    pdelta = np.square(delta)
    ssquare = np.sum(pdelta,axis=-1,keepdims=False)
    dis = np.sqrt(ssquare)
    return dis