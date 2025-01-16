from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import numpy as np
from pycocotools import mask
import wml.wml_utils as wmlu
import wml.img_utils as wmli
import wml.object_detection2.bboxes as odb
import sys
from PIL import Image
from wml.iotoolkit.coco_data_fwd import *
import copy
import os.path as osp
from wml.iotoolkit.pascal_voc_toolkit import read_voc_xml
from collections import defaultdict
from .common import *
from functools import partial



def create_category_index(categories):
  """Creates dictionary of COCO compatible categories keyed by category id.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
  """
  category_index = {}
  for cat in categories:
    category_index[cat['id']] = cat
  return category_index

class COCOData:
    load_patch = False
    def __init__(self,trans_label=None,include_masks=True,is_relative_coordinate=False,remove_crowd=True,*args,**kwargs):
        '''

        Args:
            trans_label: label fn(label) : return transed label if label is useful else return None
            include_masks:
        '''
        self.images = None
        self.annotations_index = None
        self.image_dir = None
        self.include_masks = include_masks
        self.category_index = None
        self.id2name = None #same as category_index
        if isinstance(trans_label,dict):
            self.trans_label= partial(dict_label_text2id,dict_data=trans_label)
        else:
            self.trans_label = trans_label
        self.filename2image = None
        self.is_relative_coordinate = is_relative_coordinate
        self.ids = []
        self.trans_file_name = None  # fn(filename,image_dir)->filename
        self.update_id2name = True
        self.remove_crowd = remove_crowd
        self.patchs_index = []

    def get_image_full_path(self,image):
        filename = image['file_name']
        return os.path.join(self.image_dir, filename)

    @staticmethod
    def get_image_dir_by_annotations_file(path):
        dir_name = wmlu.parent_dir_path_of_file(path)
        basename = wmlu.base_name(path).split("_")[-1]
        #return osp.join(dir_name,"images",basename)
        return osp.join(dir_name,basename)

    def read_data(self,annotations_file,image_dir=None,*args,**kwargs):
        if image_dir is None:
            image_dir = self.get_image_dir_by_annotations_file(annotations_file)
            print(f"image dir {image_dir}")
        if self.trans_label is not None:
            print(f"Trans label is not None")
        sys.stdout.flush()
        image_id2shape = {}
        with open(annotations_file, 'r') as fid:
            groundtruth_data = json.load(fid)
            _images = groundtruth_data['images']
            category_index = create_category_index(
                groundtruth_data['categories'])
            for image in groundtruth_data['images']:
                image_id2shape[image['id']] = (image['height'],image['width'])

            annotations_index = {}
            if 'annotations' in groundtruth_data:
                print(
                    'Found groundtruth annotations. Building annotations index.')
                for annotation in groundtruth_data['annotations']:
                    if self.trans_label is not None:
                        category_id = annotation['category_id']
                        new_category_id = self.trans_label(category_id)
                        if new_category_id is None:
                            continue
                        else:
                            annotation['category_id'] = new_category_id
                    if self.remove_crowd and annotation['iscrowd']:
                        continue
                    image_id = annotation['image_id']
                    bbox = annotation['bbox']
                    bbox = self.check_bbox(bbox,image_id2shape[image_id])
                    if bbox is None:
                        continue
                    annotation['bbox'] = bbox
                    annotation['name'] = category_index[category_id]['name']
                    if image_id not in annotations_index:
                        annotations_index[image_id] = []
                    annotations_index[image_id].append(copy.deepcopy(annotation))
            missing_annotation_count = 0
            images = []
            for image in _images:
                #image["file_name"] = osp.basename(image['file_name'])
                image_id = image['id']
                if image_id not in annotations_index:
                    missing_annotation_count += 1
                    annotations_index[image_id] = []
                else:
                    images.append(image)
            print(f'{missing_annotation_count} images are missing annotations.')

        self.image_dir = image_dir
        if self.trans_file_name is not None:
            _images = images
            images = []
            for image in _images:
                image["file_name"] = self.trans_file_name(image["file_name"],self.image_dir)
                images.append(image)
        self.images = images
        self.annotations_index = annotations_index
        self.category_index = category_index
        self.ids = [image["id"] for image in images]
        if self.trans_label is None:
            self.id2name = {}
            for id,info in self.category_index.items():
                self.id2name[id] = info['name']
            wmlu.show_dict(self.id2name)
        '''if COCOData.load_patch:
            self._load_patch()'''


    def _load_patch(self):
        if not COCOData.load_patch:
            return

        patchs_index = []

        for idx,image in enumerate(self.images):
            image_id = image['id']
            annotations_list = self.annotations_index[image_id]
            full_path = self.get_image_full_path(image)
            patch_path = wmlu.change_suffix(full_path,"xml")
            if not osp.exists(patch_path):
                continue
            shape, bboxes, labels_text, difficult, truncated, probs = read_voc_xml(patch_path,absolute_coord=True)
            if bboxes.shape[0] == 0:
                continue

            bboxes = odb.npchangexyorder(bboxes)
            labels = [int(x) for x in labels_text]

            old_len = len(annotations_list)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                bbox[2:] = bbox[2:]-bbox[:2]
                area = np.prod(bbox[2:])
                annotation = {"bbox":bbox,"category_id":labels[i],"iscrowd":False,"area":area}
                annotations_list.append(annotation)
            
            print(f"Load patch {patch_path}, old len {old_len}, new len {len(self.annotations_index[image_id])}.")

            patchs_index.append(idx)

        self.patchs_index = set(patchs_index)


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        '''
        binary_masks:[N,H,W]
        '''
        image = self.images[item]
        res = self.get_image_annotation(image)
        return res

    def filter(self,filter_func):
        '''
        filter_func(labels,bboxes,is_crowd)->bool
        '''
        old_images = self.images
        new_images = []
        for image in old_images:
            x = self.get_image_annotation(image)
            full_path,img_shape,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped = x
            if filter_func(labels=category_ids,bboxes=boxes,is_crowd=is_crowd):
                new_images.append(image)
        self.images = new_images
        self.ids = [image["id"] for image in new_images]
        print(f"Old len {len(old_images)}, new len {len(new_images)}")

    def get_image_id(self,item):
        return self.images[item]["id"]

    def get_image_annotation_by_image_name(self,file_name):
        if self.filename2image is None:
            self.filename2image = {}
            for image in self.images:
                tfile_name = image["file_name"]
                self.filename2image[tfile_name] = image

        image = self.filename2image[file_name]

        return self.get_image_annotation(image)
    
    def check_bbox(self,bbox,img_shape):
        image_height,image_width = img_shape
        (x, y, width, height) = bbox
        if width <= 0 or height <= 0:
            return None
        if x<0:
            x = 0
        if y<0:
            y=0
        if x>=image_width:
            x = image_width-1
        if  y>=image_height:
            y = image_height-1
        if x + width > image_width:
            width = image_width-x
        if y + height > image_height:
            height = image_height-y
        if width <= 0 or height <= 0:
            return None
        return (x,y,width,height)

    def get_image_annotation(self,image):
        image_height = image['height']
        image_width = image['width']
        image_id = image['id']

        full_path = self.get_image_full_path(image)

        xmin = []
        xmax = []
        ymin = []
        ymax = []
        is_crowd = []
        category_names = []
        category_ids = []
        area = []
        num_annotations_skipped = 0
        annotations_list = self.annotations_index[image_id]
        binary_masks = []
        for object_annotations in annotations_list:
            (x, y, width, height) = tuple(object_annotations['bbox'])
            category_id = int(object_annotations['category_id'])
            org_category_id = category_id

            if self.is_relative_coordinate:
                xmin.append(float(x) / image_width)
                xmax.append(float(x + width) / image_width)
                ymin.append(float(y) / image_height)
                ymax.append(float(y + height) / image_height)
            else:
                xmin.append(float(x))
                xmax.append(float(x + width))
                ymin.append(float(y))
                ymax.append(float(y + height))

            is_crowd.append(object_annotations['iscrowd'])
            category_ids.append(category_id)
            category_names.append(object_annotations['name'])
            area.append(object_annotations['area'])

            if self.include_masks:
                run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                    image_height, image_width)
                binary_mask = mask.decode(run_len_encoding)
                if not object_annotations['iscrowd']:
                    binary_mask = np.amax(binary_mask, axis=2)
                binary_masks.append(binary_mask)

        boxes = np.array(list(zip(ymin,xmin,ymax,xmax)),dtype=np.float32)

        if len(binary_masks)>0:
            binary_masks = np.stack(binary_masks,axis=0)
        else:
            binary_masks = None

        img_shape = [image_height,image_width]

        if len(category_ids)==0:
            print("No annotation: ", full_path)
            sys.stdout.flush()
            return DetData(full_path,img_shape,[],[],np.zeros([0,4],dtype=np.float32),None,[],[],num_annotations_skipped)

        category_ids = np.array(category_ids,dtype=np.int32)
        return DetData(full_path,img_shape,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped)

    def get_items(self):
        for image in self.images:
            res = self.get_image_annotation(image)
            if res[0] is not None:
                yield res

    def get_boxes_items(self):
        for image in self.images:
            full_path,img_size,category_ids,category_names,boxes,binary_mask,area,is_crowd,num_annotations_skipped = \
            self.get_image_annotation(image)
            if full_path is not None:
                yield DetBboxesData(full_path,img_size,category_ids,boxes,is_crowd)


class TorchCOCOData(COCOData):
    def __init__(self, img_dir, anno_path,trans_label=None):
        super().__init__(is_relative_coordinate=False,trans_label=trans_label)
        super().read_data(anno_path, img_dir)

    def __getitem__(self, item):
        x = super().__getitem__(item)
        full_path, shape, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        try:
            img = wmli.imread(full_path)
        except Exception as e:
            print(f"Read {full_path} faild, error:{e}")
            img = np.zeros([shape[0],shape[1],3],dtype=np.uint8)
        img = Image.fromarray(img)
        res = []
        nr = len(category_ids)
        boxes = odb.npchangexyorder(boxes)
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]
        #new bboxes is [N,4], [x0,y0,w,h]
        for i in range(nr):
            item = {"bbox": boxes[i], "category_id": category_ids[i], "iscrowd": is_crowd[i],"area":area[i]}
            res.append(item)

        return img, res

def load_coco_results(json_path):
    with open(json_path,"r") as f:
        data = json.load(f)
    res = wmlu.MDict(dvalue=wmlu.MDict(dtype=list))
    r_res = wmlu.MDict(dvalue=wmlu.MDict(dtype=list))
    if not isinstance(data,list) and 'annotations' in data:
        data = data['annotations']
    """
          {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    """
    for re in data:
        image_id = re["image_id"]
        category_id = re["category_id"]
        bbox = re["bbox"]
        score = re.get("score",1.0)
        res[image_id]["bbox"].append(bbox)
        res[image_id]["category_id"].append(category_id)
        res[image_id]["score"].append(score)
    
    for k,v in res.items():
        for k0,v0 in v.items():
            r_res[k][k0] = np.array(v0)
    
    return r_res

if __name__ == "__main__":
    import wml.img_utils as wmli
    import wml.object_detection2.visualization as odv
    import matplotlib.pyplot as plt
    data = COCOData()
    data.read_data("/data/mldata/coco/annotations/instances_train2014.json",image_dir="/data/mldata/coco/train2014")
    for x in data.get_items():
        full_path, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)
        def text_fn(classes,scores):
            return ID_TO_TEXT[classes]['name']
        odv.draw_bboxes_and_maskv2(
        img=img, classes=category_ids, scores=None, bboxes=boxes, masks=binary_mask, color_fn = None, text_fn = text_fn, thickness = 4,
        show_text = True,
        fontScale = 0.8)
        plt.figure()
        plt.imshow(img)
        plt.show()
