import os.path as osp
import wml.wml_utils as wmlu
import copy
import random
from typing import Type
from itertools import chain
from collections import namedtuple

VideoInfo = namedtuple('VideoInfo',["dir_path","frames","label"])

class RawFrameDataset:
    get_item_in_namedtuple = False
    def __init__(self,file_path=None) -> None:
        self.data = []
        self.dirname = None
        if file_path is not None:
            self.read(file_path)

    @classmethod
    def create(cls,other, data=None):
        ds = cls()
        if data is not None:
            ds.data = data
        else:
            ds.data = copy.deepcopy(other.data)

        ds.dirname = other.dirname

        return ds

    def read(self,file_path):
        self.data = []
        self.dirname = osp.dirname(file_path)
        with open(file_path,"r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if "#" in line:
                datas = line.split("#")
            else:
                datas = line.split(" ")
            self.data.append([datas[0],int(datas[1]),int(datas[2])]) #sub dir name, total frames, type

    def write(self,file_name,spliter="#"):
        save_path = osp.join(self.dirname,file_name)
        print(f"Save path {save_path}")
        with open(save_path,"w") as f:
            for dirname,frames_nr,f_type in self.data:
                f.write(f"{dirname}{spliter}{frames_nr}{spliter}{f_type}\n")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        data = self.data[idx]
        if self.get_item_in_namedtuple:
            return VideoInfo(osp.join(self.dirname,data[0]),data[1],data[2])
        else:
            return osp.join(self.dirname,data[0]),data[1],data[2]

    def show_statistics_info(self):
        print(f"Total {len(self.data)} files.")
        counter = wmlu.MDict(dvalue=0)
        for dirname,frames_nr,f_type in self.data:
            counter[f_type] += 1
        
        datas = list(zip(counter.keys(),counter.values()))
        datas.sort(key=lambda x:x[0])
        for f_type,nr in datas:
            print(f"Type {f_type:5d}: {nr}")

    def split_data(self,percent=[0.5,-1], shuffle=True,per_classes=True):
        '''
        按percent分割数据，-1表示所有剩余的数据
        '''
        res = []
        if not per_classes:
            datas = RawFrameDataset.simple_split_data(self.data,percent,shuffle)
            for data in datas:
                res.append(RawFrameDataset.create(self,data))
            return res
        else:
            type_dict = self.get_type_dict()

            all_res = []
            for k,v in type_dict.items():
                datas = RawFrameDataset.simple_split_data(v,percent,shuffle)
                all_res.append(datas)

            datas = list(zip(*all_res))
            for data in datas:
                data = list(chain(*data))
                if shuffle:
                    random.shuffle(data)
                res.append(RawFrameDataset.create(self,data))

            return res
        
    def get_type_dict(self):
        res = wmlu.MDict(dtype=list)
        for data in self.data:
            res[data[-1]].append(data)
        
        return res

    @staticmethod
    def simple_split_data(data,percent=[0.5,-1], shuffle=True):
        data = copy.deepcopy(data)
        total_nr = len(data)
        if shuffle:
            random.shuffle(data)
        res = []
        for x in percent:
            if x>=0:
                nr = int(total_nr*x+0.5)
                res.append(data[:nr])
                data = data[nr:]
            else:
                res.append(data)
                break
        return res




            

            
