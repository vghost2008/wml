import numpy as np
from typing import Iterable
from .kps_structures import WMCKeypoints,WMCKeypointsItem

class HeatmapGenerator:
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


    def __call__(self, joints):
        '''
        joints: [max_num_people,num_joints,2] (x,y)
        '''
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

class MultiClassesHeatmapGenerator:
    def __init__(self,  num_classes, output_res=None,sigma=1):
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
        self.set_output_res(output_res)
        self.num_classes = num_classes
        if sigma < 0 and self.output_res is not None:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def set_output_res(self,output_res):
        if output_res is None:
            self.output_res = None
            return
        if not isinstance(output_res,Iterable):
            output_res = (output_res,output_res)
        self.output_res = output_res

    def __call__(self, joints,labels,output_res=None):
        '''
        joints: [points_nr,N,2] (x,y) or WMCKeypoints
        labels: [points_nr]
        return:
        [num_classes,H,W]
        '''
        if output_res is not None:
            self.set_output_res(output_res)
        hms = np.zeros((self.num_classes, self.output_res[0], self.output_res[1]),
                       dtype=np.float32)
        sigma = self.sigma
        for idx,pts in zip(labels,joints):
            if isinstance(pts,WMCKeypointsItem):
                pts = pts.points
            for pt in pts:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or \
                   x >= self.output_res[1] or y >= self.output_res[0]:
                    continue
    
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
    
                c, d = max(0, -ul[0]), min(br[0], self.output_res[1]) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res[0]) - ul[1]
    
                cc, dd = max(0, ul[0]), min(br[0], self.output_res[1])
                aa, bb = max(0, ul[1]), min(br[1], self.output_res[0])
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms



class ScaleAwareHeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_gaussian_kernel(self, sigma):
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        for p in joints:
            sigma = p[0, 3]
            g = self.get_gaussian_kernel(sigma)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], g[a:b, c:d])
        return hms