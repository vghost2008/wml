import json
import os.path as osp
import wml.object_detection2.bboxes as odb
import numpy as np
import wml.object_detection2.visualization as odv
import wml.wml_utils as wmlu
import PIL.Image as Image
import wml.img_utils as wmli
from .common import *


class Object365:
    def __init__(self,absolute_coord=True):
        self.json_path = None
        self.img_dir = None
        self.data = None
        self.absolute_coord = absolute_coord
        self.images_data = {}
        self.annotation_data = {}
        self.ids = []
        self.id2name = {}

    def read_data(self,json_path,img_dir):
        self.json_path = json_path
        self.img_dir = img_dir
        with open(self.json_path,"r") as f:
            self.data = json.load(f)
        self.images_data = {}
        self.ids = []
        self.annotation_data = {}

        for d in self.data['images']:
            id = d['id']
            self.ids.append(id)
            self.images_data[id] = d
        for d in self.data['annotations']:
            id = d['image_id']
            self.add_anno_data(self.annotation_data,id,d)
        name_dict = {}
        for x in self.data['categories']:
            name_dict[x['id']] = x['name']
        self.id2name = name_dict
        info = list(name_dict.items())
        info.sort(key=lambda x:x[0])
        print(f"Category")
        wmlu.show_list(info)

    @staticmethod
    def add_anno_data(dict_data,id,item_data):
        if id in dict_data:
            dict_data[id].append(item_data)
        else:
            dict_data[id] = [item_data]

    def __len__(self):
        return len(self.ids)

    def getitem_by_id(self,id):
        idx = self.ids.index(id)
        return self.__getitem__(idx)

    def __getitem__(self, item):
        try:
            id = self.ids[item]
            item_data = self.annotation_data[id]
            is_crowd = []
            bboxes = []
            labels = []
            labels_names = []
            for data in item_data:
                is_crowd.append(data['iscrowd'])
                bboxes.append(data['bbox'])
                label = data['category_id']
                label_text = self.id2name[label]
                labels.append(label)
                labels_names.append(label_text)
            bboxes = np.array(bboxes,dtype=np.float32)
            bboxes[...,2:] = bboxes[...,:2]+bboxes[...,2:]
            bboxes = odb.npchangexyorder(bboxes)
            is_crowd = np.array(is_crowd,dtype=np.float32)
            image_data = self.images_data[id]
            shape = [image_data['height'],image_data['width']]
            if not self.absolute_coord:
                bboxes = odb.absolutely_boxes_to_relative_boxes(bboxes,width=shape[1],height=shape[0])
            img_name = image_data['file_name']
            img_file = osp.join(self.img_dir,img_name)
            '''
            bboxes: [N,4] (y0,x0,y1,x1)
            '''
            return DetData(img_file, shape, labels, labels_names, bboxes, None, None, is_crowd, None)
        except Exception as e:
            print(e)
            return None

    def get_items(self):
        for i in range(len(self.ids)):
            res = self.__getitem__(i)
            if res is None:
                continue
            yield res

class TorchObject365(Object365):
    def __init__(self, img_dir,anno_path,absolute_coord=True):
        super().__init__(absolute_coord)
        super().read_data(anno_path,img_dir)

    def __getitem__(self, item):
        x = super().__getitem__(item)
        full_path, shape,category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)
        img = Image.fromarray(img)
        res = []
        nr = len(category_ids)
        boxes = odb.npchangexyorder(boxes)
        boxes[...,2:] = boxes[...,2:] - boxes[...,:2]
        for i in range(nr):
            item = {"bbox":boxes[i],"category_id":category_ids[i],"iscrowd":is_crowd[i]}
            res.append(item)
        
        return img,res

if __name__ == "__main__":
    import wml.img_utils as wmli
    import random
    import matplotlib.pyplot as plt
    import time

    random.seed(time.time())

    save_dir = "/home/wj/ai/mldata1/Objects365/tmp"

    wmlu.create_empty_dir_remove_if(save_dir)
    data = Object365(absolute_coord=False)
    data.read_data("/home/wj/ai/mldata1/Objects365/Annotations/train/train.json","/home/wj/ai/mldata1/Objects365/Images/train/train")
    idxs = list(range(len(data)))
    random.shuffle(idxs)
    max_nr = 100
    idxs = idxs[:max_nr]
    for idx in idxs:
        x = data[idx]
        full_path, shape,category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)

        def text_fn(classes, scores):
            return data.id2name[classes]

        img = odv.draw_bboxes(
            img=img, classes=category_ids, scores=None, bboxes=boxes, color_fn=None,
            text_fn=text_fn, thickness=2,
            show_text=True,
            font_scale=0.8)
        save_path = osp.join(save_dir,osp.basename(full_path))
        wmli.imwrite(save_path,img)
        '''plt.figure()
        plt.imshow(img)
        plt.show()'''
