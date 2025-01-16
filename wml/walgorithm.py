#coding=utf-8
from multiprocessing import Pool
import numpy as np
import math
import cv2

def _edit_distance(v0,v1):
    if v0 == v1:
        return 0
    if (len(v0)==0) or (len(v1)==0):
        return max(len(v0),len(v1))
    c0 = _edit_distance(v0[:-1],v1)+1
    c1 = _edit_distance(v0,v1[:-1])+1
    cr = 0
    if v0[-1] != v1[-1]:
        cr = 1
    c2 = _edit_distance(v0[:-1],v1[:-1])+cr
    return min(min(c0,c1),c2)

def mt_edit_distance(v0,v1,pool):
    if v0 == v1:
        return 0
    if (len(v0)==0) or (len(v1)==0):
        return max(len(v0),len(v1))
    c0 = edit_distance(v0[:-1],v1)+1
    c1 = edit_distance(v0,v1[:-1])+1
    cr = 0
    if v0[-1] != v1[-1]:
        cr = 1
    c2 = edit_distance(v0[:-1],v1[:-1])+cr
    return min(min(c0,c1),c2)

def edit_distance(sm, sn):
    m, n = len(sm) + 1, len(sn) + 1

    matrix = np.ndarray(shape=[m,n],dtype=np.int32)

    matrix[0][0] = 0
    for i in range(1, m):
        matrix[i][0] = matrix[i - 1][0] + 1

    for j in range(1, n):
        matrix[0][j] = matrix[0][j - 1] + 1

    for i in range(1, m):
        for j in range(1, n):
            if sm[i - 1] == sn[j - 1]:
                cost = 0
            else:
                cost = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

    return matrix[m - 1][n - 1]


def pearsonr(x,y):
    #Pearson_correlation coefficient [-1,1]
    if not isinstance(x,np.ndarray):
        x = np.array(x)

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_ba = x-x_mean
    y_ba = y-y_mean
    v = np.sum(x_ba*y_ba)
    dx = np.sum((x-x_mean)**2)
    dy = np.sum((y-y_mean)**2)
    sv = np.sqrt(dx*dy)+1e-8

    return v/sv

def points_to_polygon(points):
    '''

    Args:
        points: [N,2],(x,y)

    Returns:
        idxs,[N],sorted points[N,2]
    '''

    points = np.array(points)
    base_point = 0
    if points.shape[0]<=3:
        return list(range(points.shape[0])),points
    for i in range(points.shape[0]):
        if points[i,1]<points[base_point,1]:
            base_point = i
        elif points[i, 1] == points[base_point, 1] and points[i,0]<points[base_point,0]:
            base_point = i

    angles = np.zeros([points.shape[0]],dtype=np.float32)

    for i in range(points.shape[0]):
        y = points[i,1]-points[base_point,1]
        x = points[i,0]-points[base_point,0]
        angles[i] = math.atan2(y,x)
        if angles[i]<0:
            angles[i] += math.pi
    angles[base_point] = -1e-8
    idxs = np.argsort(angles)
    return idxs,points[idxs]

def left_shift_array(array,size=1):
    '''

    Args:
        array: [N]
        size: 1->N-1
    example:
        array = [1,2,3,4]
        size=1
        return:
        [2,3,4,1]
    Returns:
        [N]
    '''
    first_part = array[size:]
    second_part = array[:size]
    return np.concatenate([first_part,second_part],axis=0)

def right_shift_array(array, size=1):
    '''

    Args:
        array: [N]
        size: 1->N-1
    example:
        array = [1,2,3,4]
        size=1
        return:
        [4,1,2,3,]
    Returns:
        [N]
    '''
    first_part = array[-size:]
    second_part = array[:-size]
    return np.concatenate([first_part, second_part], axis=0)


def sign_point_line(point,line):
    '''

    Args:
        point: [2] x,y
        line: np.array([2,2]) [(x0,y0),(x1,y1)]

    Returns:
        True or False
    '''
    line = np.array(line)
    p0 = line[0]
    vec0 = line[1]-p0
    vec1 = point-p0
    return vec0[0]*vec1[1]-vec0[1]*vec1[0]<0

def in_range(v,*kargs):
    if len(kargs)==1:
        min_v = kargs[0][0]
        max_v = kargs[0][1]
    elif len(kargs)==2:
        min_v = kargs[0]
        max_v = kargs[1]
    else:
        raise RuntimeError(f"in_range: ERROR args {kargs}")

    return v>=min_v and v<=max_v

def points_on_circle(center=None,r=None,points_nr=100):
    '''
    将圆离散化为散点
    '''
    points = []
    for i in range(points_nr):
        angle = math.pi*2*i/points_nr
        x = math.cos(angle)
        y = math.sin(angle)
        points.append([x,y])
    if r is not None:
        points = np.array(points)*r
    if center is not None:
        center = np.reshape(np.array(center),[1,2])
        points = points + center
    
    return points

def getRotationMatrix2D(center, angle, scale,out_offset=None):
    if out_offset is None:
        '''
        cv2为先平移-center,scale,rotate,再平移center
        M(center)*M(rotate)*M(scale)*M(-center)*X
        '''
        return cv2.getRotationMatrix2D(center=center,angle=angle,scale=scale)
    offset_in = np.array([[1,0,-center[0]],[0,1,-center[1]]],dtype=np.float32)
    rotate_m = cv2.getRotationMatrix2D(center=[0,0],angle=angle,scale=scale)
    offset_out = np.array([[1,0,out_offset[0]],[0,1,out_offset[1]]],dtype=np.float32)
    line3 = np.array([[0,0,1]],dtype=np.float32)
    offset_in = np.concatenate([offset_in,line3],axis=0)
    rotate_m = np.concatenate([rotate_m,line3],axis=0)
    offset_out = np.concatenate([offset_out,line3],axis=0)
    m = np.dot(rotate_m,offset_in)
    m = np.dot(offset_out,m)
    return m[:2]

def lower_bound(datas, target):
    """
    对于升序数组，找到第一个大于等于（或不小于）给定值的目标元素的位置
    """
    if datas[-1]<target:
        return -1
    if datas[0]>=target:
        return 0
    left, right = 0, len(datas) - 1  # 闭区间[left, right]
    while left <= right:  # 区间不为空
        mid = (left + right) // 2
        if datas[mid] < target:
            left = mid + 1  # [mid + 1, right]
        else:
            right = mid - 1  # [left, mid - 1]
    return left


def remove_non_ascii(s):
    return ''.join(filter(str.isascii, s))