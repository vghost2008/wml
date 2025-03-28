import numpy as np
import wml.basic_img_utils as bwmli
import pickle
import cv2

class MCI:
    '''
    Multi channel image
    后辍名 .mci
    '''
    def __init__(self,data):
        self.data = np.array(data,dtype=np.uint8)

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

        return cls(imgs)

    @staticmethod
    def read(file_path):
        with open(file_path,"rb") as f:
            data = pickle.load(f)
            shape = data['shape']
            datas = []
            for d in data['imgs']:
                d = bwmli.decode_img(d,fmt="grayscale")
                datas.append(d)
            datas = np.stack(datas,axis=-1)
            return datas

    
    @staticmethod
    def write(file_path,data):
        imgs = []
        for i in range(data.shape[2]):
            img = data[:,:,i]
            img = bwmli.encode_img(img)
            imgs.append(img)
        datas = {'shape':data.shape,'imgs':imgs}

        with open(file_path,"wb") as f:
            pickle.dump(datas,f)

    def save(self,file_path):
        MCI.write(file_path,self.data)
    
    @staticmethod
    def get_img_size(file_path):
        with open(file_path,"rb") as f:
            data = pickle.load(f)
            shape = data['shape']
            return shape


