import random
import os
import wml_utils as wmlu
from .common import resample,ignore_case_dict_label_text2id
from abc import ABCMeta, abstractmethod
from functools import partial



class BaseDataset(metaclass=ABCMeta):
    def __init__(self,label_text2id=None,
                      filter_empty_files=False,filter_error=False,resample_parameters=None,shuffle=True,silent=False,keep_no_ann_imgs=False,absolute_coord=True,):
        '''
        label_text2id: func(name)->int
        '''
        self.files = None
        if isinstance(label_text2id,dict):
            self.label_text2id = partial(ignore_case_dict_label_text2id,
                    dict_data=wmlu.trans_dict_key2lower(label_text2id))
        else:
            self.label_text2id = label_text2id
        self.filter_empty_files = filter_empty_files
        self.filter_error = filter_error

        if resample_parameters is not None:
            self.resample_parameters = {}
            for k,v in resample_parameters.items():
                if isinstance(k,(str,bytes)) and self.label_text2id is not None:
                    k = self.label_text2id(k)
                self.resample_parameters[k] = v
            print("resample parameters")
            wmlu.show_dict(self.resample_parameters)
        else:
            self.resample_parameters = None

        self.shuffle = shuffle
        self.silent = silent
        self.keep_no_ann_imgs = keep_no_ann_imgs
        self.absolute_coord = absolute_coord
        pass

    def find_files(self,dir_path,img_suffix):
        if isinstance(dir_path,str):
            #从单一目录读取数据
            print(f"Read {dir_path}")
            if not os.path.exists(dir_path):
                print(f"Data path {dir_path} not exists.")
                return False
            files = self.find_files_in_dir(dir_path)
        elif isinstance(dir_path,(list,tuple)):
            if isinstance(dir_path[0],(str,bytes)) and os.path.isdir(dir_path[0]):
                '''
                从多个目录读取数据
                Example:
                [/dataset/dir0,/dataset/dir1,/dataset/dir2]
                '''
                files = self.find_files_in_dirs(dir_path, img_suffix=img_suffix)
            elif isinstance(dir_path[0],(list,tuple)) and not isinstance(dir_path[0][1],(str,bytes)):
                '''
                用于对不同目录进行不同的重采样
                Example:
                [(/dataset/dir0,3),(/dataset/dir1,1),(/dataset/dir2,100)]
                '''
                files = self.find_files_in_dirs_with_repeat(dir_path,img_suffix=img_suffix)
            else:
                #文件列表
                files = dir_path
        
        return files

    @abstractmethod
    def find_files_in_dir(self,dir_path,img_suffix):
        pass

    def find_files_in_dirs(self,dirs,img_suffix=".jpg"):
        all_files = []
        for dir_path in dirs:
            files = self.find_files_in_dir(dir_path,img_suffix)
            print(f"Find {len(files)} in {dir_path}")
            all_files.extend(files)

        return all_files

    def find_files_in_dirs_with_repeat(self,dirs,img_suffix=".jpg"):
        all_files = []
        raw_nr = 0
        for dir_path,repeat_nr in dirs:
            files = self.find_files_in_dir(dir_path,img_suffix)
            files_nr = len(files)
            if repeat_nr>1:
                files = list(files)*int(repeat_nr)

            print(f"Find {files_nr} in {dir_path}, repeat to {len(files)} files")
            raw_nr += files_nr
            all_files.extend(files)

        print(f"Total find {raw_nr} files, repeat to {len(all_files)} files.")
        return all_files

    def read_data(self,dir_path,img_suffix=".jpg;;.bmp;;.png;;.jpeg"):
        _files = self.find_files(dir_path,img_suffix=img_suffix)
        if self.filter_empty_files and self.label_text2id:
            _files = self.apply_filter_empty_files(_files)
        elif self.filter_error:
            _files = self.apply_filter_error_files(_files)
        if self.resample_parameters is not None and self.label_text2id:
            _files = self.resample(_files)
        
        self.files = _files

        if isinstance(dir_path,(str,bytes)):
            print(f"Total find {len(self.files)} in {dir_path}")
        else:
            print(f"Total find {len(self.files)}")
        if self.shuffle:
            random.shuffle(self.files)
        print("Files")
        wmlu.show_list(self.files[:100])
        if len(self.files)>100:
            print("...")

    def apply_filter_empty_files(self,files):
        new_files = []
        for fs in files:
            try:
                labels,labels_names = self.get_labels(fs)
                is_none = [x is None for x in labels]
                if not all(is_none):
                    new_files.append(fs)
                else:
                    print(f"File {fs[1]} is empty, remove from dataset, labels names {labels_names}, labels {labels}")
            except Exception as e:
                print(f"Read {fs[1]} faild, info: {e}.")
                pass

        return new_files

    def apply_filter_error_files(self,files):
        new_files = []
        for fs in files:
            try:
                labels,labels_names = self.get_labels(fs)
                new_files.append(fs)
            except Exception as e:
                print(f"Read {fs[1]} faild, info: {e}.")
                pass

        return new_files

    @abstractmethod
    def get_labels(self,fs):
        #return labels,labels_names
        pass
    
    def resample(self,files):
        all_labels = []
        for fs in files:
            try:
                labels,_ = self.get_labels(fs)
                all_labels.append(labels)
            except Exception as e:
                print(f"Labelme resample error: {e}")

        return resample(files,all_labels,self.resample_parameters)

    def __len__(self):
        return len(self.files)

    
    

