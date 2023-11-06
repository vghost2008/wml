import wml_utils as wmlu
import os
import json
import numpy as np
import cv2 as cv
import copy
import img_utils as wmli
import random
import matplotlib.pyplot as plt
import sys
import cv2
from object_detection2.standard_names import *
import object_detection2.bboxes as odb
from functools import partial
from .common import *
from .labelme_toolkit_fwd import *
import glob

class LabelMeData(object):
    def __init__(self,label_text2id=None,shuffle=False,absolute_coord=True,
                 filter_empty_files=False,
                 resample_parameters=None):
        '''
        label_text2id: func(name)->int
        '''
        self.files = None
        if isinstance(label_text2id,dict):
            self.label_text2id = partial(dict_label_text2id,dict_data=label_text2id)
        else:
            self.label_text2id = label_text2id
        self.shuffle = shuffle
        self.absolute_coord = absolute_coord
        self.filter_empty_files = filter_empty_files
        if resample_parameters is not None:
            self.resample_parameters = {}
            for k,v in resample_parameters.items():
                if isinstance(k,(str,bytes)):
                    k = self.label_text2id(k)
                self.resample_parameters[k] = v
            print("resample parameters")
            wmlu.show_dict(self.resample_parameters)
        else:
            self.resample_parameters = None
        
    def read_data(self,dir_path,img_suffix=".jpg;;.bmp;;.png;;.jpeg"):
        _files = get_files(dir_path,img_suffix=img_suffix)
        if self.filter_empty_files and self.label_text2id:
            _files = self.apply_filter_empty_files(_files)
        if self.resample_parameters is not None and self.label_text2id:
            _files = self.resample(_files)
        
        self.files = _files

        print(f"Total find {len(self.files)} in {dir_path}")
        if self.shuffle:
            random.shuffle(self.files)
        print("Files")
        wmlu.show_list(self.files[:100])
        if len(self.files)>100:
            print("...")

    def apply_filter_empty_files(self,files):
        new_files = []
        for fs in files:
            img_file,json_file = fs
            image, annotations_list = read_labelme_data(json_file, None,use_semantic=True,mask_on=False)
            labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
            labels = [self.label_text2id(x) for x in labels_names]
            is_none = [x is None for x in labels]
            if not all(is_none):
                new_files.append(fs)
            else:
                print(f"File {json_file} is empty, labels names {labels_names}, labels {labels}")

        return new_files

    def resample(self,files):
        all_labels = []
        for fs in files:
            img_file,json_file = fs
            image, annotations_list = read_labelme_data(json_file, None,use_semantic=True,mask_on=False)
            labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
            labels = [self.label_text2id(x) for x in labels_names]
            all_labels.append(labels)

        return resample(files,all_labels,self.resample_parameters)

    def __len__(self):
        return len(self.files)

    def get_ann_info(self,idx):
        img_file,json_file = self.files[idx]
        with open(json_file,"r",encoding="gb18030") as f:
            data_str = f.read()
            image = {}
            try:
                json_data = json.loads(data_str)
                img_width = int(json_data["imageWidth"])
                img_height = int(json_data["imageHeight"])
                image["height"] = int(img_height)
                image["width"] = int(img_width)
                image["file_name"] = wmlu.base_name(json_file)
            except:
                pass
        
        return image
    
    def get_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        '''
        for i in range(len(self.files)):
            sys.stdout.write('\r>> read data %d/%d' % (i + 1, len(self.files)))
            sys.stdout.flush()
            yield self.__getitem__(i)

    def __getitem__(self,idx):
        '''
        :return: 
        full_path,img_size,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped
        binary_masks:[N,H,W]
        bboxes:[N,4] (ymin,xmin,ymax,xmax)
        '''
        img_file, json_file = self.files[idx]
        image, annotations_list = read_labelme_data(json_file, None,use_semantic=True)
        labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
        masks = [ann["segmentation"] for ann in annotations_list]
        if len(masks)>0:
            try:
                masks = np.stack(masks,axis=0)
            except:
                img_height = image['height']
                img_width = image['width']
                masks = np.zeros(shape=[0,img_height,img_width],dtype=np.uint8)
        else:
            img_height = image['height']
            img_width = image['width']
            masks = np.zeros(shape=[0,img_height,img_width],dtype=np.uint8)

        
        if self.label_text2id is not None:
            labels = [self.label_text2id(x) for x in labels_names]
            keep = [x is not None for x in labels]
            labels = [x if x is not None else -1 for x in labels]
            labels = np.array(labels,dtype=np.int32)
            labels = labels[keep]
            bboxes = bboxes[keep]
            masks = masks[keep]
            labels_names = np.array(labels_names)[keep]
        else:
            labels = None
            
        return img_file, [image['height'],image['width']],labels, labels_names, bboxes, masks, None, None,None 
    
    def get_boxes_items(self):
        '''
        :return: 
        full_path,img_size,category_ids,boxes,is_crowd
        '''
        for i,(img_file, json_file) in enumerate(self.files):
            sys.stdout.write('\r>> read data %d/%d' % (i + 1, len(self.files)))
            sys.stdout.flush()
            image, annotations_list = read_labelme_data(json_file, None)
            labels_names,bboxes = get_labels_and_bboxes(image,annotations_list,is_relative_coordinate=not self.absolute_coord)
            labels = [self.label_text2id(x) for x in labels_names]
            yield img_file,[image['height'],image['width']],labels, bboxes,  None
            
if __name__ == "__main__":
    #data_statistics("/home/vghost/ai/mldata/qualitycontrol/rdatasv3")
    import img_utils as wmli
    import object_detection2.visualization as odv
    import matplotlib.pyplot as plt
    ID_TO_TEXT = {1:{"id":1,"name":"a"},2:{"id":2,"name":"b"},3:{"id":3,"name":"c"}}
    NAME_TO_ID = {}
    for k,v in ID_TO_TEXT.items():
        NAME_TO_ID[v["name"]] = v["id"]
    def name_to_id(name):
        return NAME_TO_ID[name]

    data = LabelMeData(label_text2id=name_to_id,shuffle=True)
    #data.read_data("/data/mldata/qualitycontrol/rdatasv5_splited/rdatasv5")
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatav10_preproc")
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10_neg_preproc")
    data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10_1x_neg_preproc")
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/x")
    for x in data.get_items():
        full_path, img_info,category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)


        def text_fn(classes, scores):
            return ID_TO_TEXT[classes]['name']

        odv.draw_bboxes_and_maskv2(
            img=img, classes=category_ids, scores=None, bboxes=boxes, masks=binary_mask, color_fn=None,
            text_fn=text_fn, thickness=4,
            show_text=True,
            fontScale=0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()
