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

def align_to(input,align_v):
    return ((input+align_v-1)//align_v)*align_v

def point2point_distance_square(p0,p1):
    '''
    p0: [...,2] (x,y)
    p1: [...,2] (x,y)
    '''

    delta = p1-p0
    return np.square(delta[...,0])+np.square(delta[...,1])

def point2point_distance(p0,p1):
    '''
    p0: [...,2] (x,y)
    p1: [...,2] (x,y)
    '''
    return np.sqrt(point2point_distance_square(p0,p1))



def point2line_distance(p,line_p0,line_p1,abs=True):
    '''
    p: [...,2] (x,y)
    '''
    x0,y0 = line_p0
    x1,y1 = line_p1
    #转换为ax+by+c=0的形式
    a = y1-y0
    b = x0-x1
    c = y0*x1-x0*y1
    inv_s = math.sqrt(a*a+b*b)

    if inv_s<1e-9:
        raise RuntimeError(f"ERROR: line {line_p0}, {line_p1}")
    
    x,y = np.split(p,2,axis=-1)
    x = np.squeeze(x,axis=-1)
    y = np.squeeze(y,axis=-1)
    numerator = a*x+b*y+c
    if abs:
        if len(numerator)>1:
            numerator = np.abs(numerator)
        else:
            numerator = math.abs(numerator)
    return numerator/inv_s

def trans_line2abc(line):
    x0,y0,x1,y1 = line
    a = y1-y0
    b = x0-x1
    c = y0*x1-x0*y1
    return a,b,c

def line_cross_point(a1,b1,c1,a2,b2,c2):
    d = a1*b2-a2*b1
    if math.fabs(d)<1e-9:
        return None
    
    x = (b1*c2-b2*c1)/d
    y = (a2*c1-a1*c2)/d

    return (x,y)

def line_cross_point_p(line0,line1):
    '''
    line0: [4](x0,y0,x1,y1)
    line1: [4](x0,y0,x1,y1)
    '''
    a1,b1,c1 = trans_line2abc(line0) 
    a2,b2,c2 = trans_line2abc(line1)
    return line_cross_point(a1,b1,c1,a2,b2,c2)

def __norm_point(p):
    inv_s = math.sqrt(p[0]*p[0]+p[1]*p[1])
    if inv_s<1e-9:
        return p
    return p/inv_s

def __point_len(p):
    return math.sqrt(p[0]*p[0]+p[1]*p[1])

def line_len(line):
    v = line[2:]-line[:2]
    return __point_len(v)

def is_on_line_segment(line,p,eps=1e-1):
    '''
    测试点p是否在线段line上
    '''
    p0 = line[:2]
    p1 = line[2:]
    v0 = p1-p0
    v1 = p-p0

    v1_len = __point_len(v1)

    if v1_len < eps:
        return True
    
    if __point_len(v1)>__point_len(v0):
        return False

    v0 = __norm_point(v0)
    v1 = __norm_point(v1)

    d = v0[0]*v1[0]+v0[1]*v1[1]

    delta = eps*1e-2

    if math.fabs(d-1)<delta:
        return True
    return False


def line_segment_cross_point_p(line0,line1,eps=1e-1):
    '''
    line0: [4](x0,y0,x1,y1)
    line1: [4](x0,y0,x1,y1)
    '''
    p = line_cross_point_p(line0,line1)

    if is_on_line_segment(line0,p,eps=eps) and is_on_line_segment(line1,p,eps=eps):
        return p
    
    return None
