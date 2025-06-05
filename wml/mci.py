import numpy as np
import wml.basic_img_utils as bwmli
import pickle
import colorama
import copy
import os
import os.path as osp
import cv2

class MCI:
    '''
    Multi channel image
    后辍名 .mci
    '''
    def __init__(self,data,metadata=None):
        self.data = np.array(data,dtype=np.uint8)
        self.metadata = metadata

    @property
    def shape(self):
        return self.data.shape
    
    @classmethod
    def zeros(cls,shape):
        '''
        shape: [H,W,C]
        '''
        return cls(np.zeros(shape))

    
    @classmethod
    def from_files(cls,files):
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
        metadata=dict(files=files)

        return cls(imgs,metadata=metadata)

    @staticmethod
    def read(file_path):
        with open(file_path,"rb") as f:
            data = pickle.load(f)
            shape = data['shape']
            datas = []

            if 'metadata' in data:
                metadata = data['metadata']
                fmt = metadata.get('fmt',"jpg")
            else:
                fmt = 'jpg'

            if fmt.lower() == "raw":
                datas = data['imgs']
            else:
                for d in data['imgs']:
                    d = bwmli.decode_img(d,fmt="grayscale")
                    datas.append(d)
                datas = np.stack(datas,axis=-1)

            return datas

    
    @staticmethod
    def write(file_path,data,metadata=None,fmt='jpg'):
        '''
        data: [H,W,C]
        fmt: raw/jpg
        '''

        if fmt not in ['raw','jpg']:
            print(colorama.Fore.YELLOW+f"WARNING: MCI: unknow save fmt {fmt}, use default jpg.")
            fmt = 'jpg'

        if fmt == 'raw':
            imgs = data
        else:
            imgs = []
            for i in range(data.shape[2]):
                img = data[:,:,i]
                img = bwmli.encode_img(img)
                imgs.append(img)
        datas = {'shape':data.shape,'imgs':imgs}
        if metadata is not None:
            metadata = dict(metadata)
            metadata['fmt'] = fmt
        else:
            metadata = {'fmt':fmt}

        datas['metadata'] = metadata
        
        dir_path = osp.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as f:
            pickle.dump(datas,f)

    def save(self,file_path,fmt="jpg"):
        '''
        fmt: jpg/raw
        '''
        MCI.write(file_path,self.data,metadata=self.metadata,fmt=fmt)
    
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

        return MCI(cur_img,self.metadata)




