import numpy as np
import wml.basic_img_utils as bwmli
import pickle
import colorama
import copy
import os
import os.path as osp
import cv2
import tifffile
import json

class MCITIFF:
    '''
    Multi channel image
    后辍名 .mci
    '''
    def __init__(self,data,metadata={}):
        self.data = np.array(data,dtype=np.uint8)
        '''
        metadata:
        annotations: 原始标注文件的通道号
        annotation_files: 原始标注文件名
        '''
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    @property
    def shape(self):
        return self.data.shape

    @property
    def annotations(self):
        return self.metadata.get('annotations',None)
    
    @classmethod
    def zeros(cls,shape):
        '''
        shape: [H,W,C]
        '''
        return cls(np.zeros(shape))

    
    @classmethod
    def from_files(cls,files,annotations=None,annotation_files=None):
        imgs = []
        shape = None
        for f in files:
            if f is not None:
                img = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
                shape = img.shape
            else:
                img = None
            imgs.append(img)
    
        imgs = [x if x is not None else np.zeros(shape,dtype=np.uint8) for x in imgs]

        imgs = np.stack(imgs,axis=2)
        metadata=dict(files=files,annotations=annotations,annotation_files=annotation_files)

        return cls(imgs,metadata=metadata)

    @staticmethod
    def read(file_path):
        with tifffile.TiffFile(file_path) as tif:
            image_data = tif.asarray()
            shape = image_data.shape
            image_data = np.reshape(image_data,[shape[2],shape[0],shape[1]])
            image_data = np.transpose(image_data,[1,2,0])
            
            # 读取元数据
            metadata = {}
            if tif.pages[0].description:
                try:
                    metadata = json.loads(tif.pages[0].description)
                except json.JSONDecodeError:
                    metadata['raw_description'] = tif.pages[0].description
            
            return MCITIFF(image_data,metadata)

    
    @staticmethod
    def write(file_path,data,metadata=None,*args,**kwargs):
        '''
        data: [H,W,C]
        fmt: raw/jpg
        '''


        dir_path = osp.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        if metadata is None:
            metadata = {}
        metadata['shape'] = data.shape

        data = np.transpose(data,[2,0,1])
        if metadata is not None:
            metadata_json = json.dumps(metadata, indent=2)
        else:
            metadata_json = None

        data = np.ascontiguousarray(data)
    
        tifffile.imwrite(
            file_path, 
            data, 
            shape=data.shape,
            description=metadata_json,
            compression='lzw',
            software='WML MCITIFF Writer'
        )
    


    def save(self,file_path,fmt="jpg"):
        '''
        fmt: jpg/raw
        '''
        MCITIFF.write(file_path,self.data,metadata=self.metadata,fmt=fmt)
    
    @staticmethod
    def get_img_size(file_path):
        with open(file_path,"rb") as f:
            data = pickle.load(f)
            shape = data['shape']
            return shape

    def sub_image(self,rect,pad_value=127):
        cur_img = self.data
    
        if rect[0]<0 or rect[1]<0 or rect[2]>cur_img.shape[0] or rect[3]>cur_img.shape[1]:
            py0 = -rect[0] if rect[0]<0 else 0
            py1 = rect[2]-cur_img.shape[0] if rect[2]>cur_img.shape[0] else 0
            px0 = -rect[1] if rect[1] < 0 else 0
            px1 = rect[3] - cur_img.shape[1] if rect[3] > cur_img.shape[1] else 0
            cur_img = np.pad(cur_img,[[py0,py1],[px0,px1],[0,0]],constant_values=pad_value)
            rect[0] += py0
            rect[1] += px0
            rect[2] += py0
            rect[3] += px0
    
        cur_img = copy.deepcopy(cur_img[rect[0]:rect[2],rect[1]:rect[3]])

        return MCITIFF(cur_img,self.metadata)




