from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import numpy as np
from pycocotools import mask
import wml_utils as wmlu
import img_utils as wmli
import object_detection2.bboxes as odb
import sys
from PIL import Image
from iotoolkit.coco_data_fwd import *



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
    trans_file_name = None  # fn(filename,image_dir)->filename
    def __init__(self,trans_label=None,include_masks=False,is_relative_coordinate=False):
        '''

        Args:
            trans_label: label fn(label) : return transed label is label is useful else return None
            include_masks:
        '''
        self.images = None
        self.annotations_index = None
        self.image_dir = None
        self.include_masks = include_masks
        self.category_index = None
        self.trans_label = trans_label
        self.filename2image = None
        self.is_relative_coordinate = is_relative_coordinate
        self.ids = []

    def get_image_full_path(self,image):
        filename = image['file_name']
        return os.path.join(self.image_dir, filename)

    def read_data(self,annotations_file,image_dir):
        with open(annotations_file, 'r') as fid:
            groundtruth_data = json.load(fid)
            images = groundtruth_data['images']
            category_index = create_category_index(
                groundtruth_data['categories'])

            annotations_index = {}
            if 'annotations' in groundtruth_data:
                print(
                    'Found groundtruth annotations. Building annotations index.')
                for annotation in groundtruth_data['annotations']:
                    image_id = annotation['image_id']
                    if image_id not in annotations_index:
                        annotations_index[image_id] = []
                    annotations_index[image_id].append(annotation)
            missing_annotation_count = 0
            for image in images:
                image_id = image['id']
                if image_id not in annotations_index:
                    missing_annotation_count += 1
                    annotations_index[image_id] = []
            print(f'{missing_annotation_count} images are missing annotations.')

        self.image_dir = image_dir
        if COCOData.trans_file_name is not None:
            _images = images
            images = []
            for image in _images:
                image["file_name"] = COCOData.trans_file_name(image["file_name"],self.image_dir)
                images.append(image)
        self.images = images
        self.annotations_index = annotations_index
        self.category_index = category_index
        self.ids = [image["id"] for image in images]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        image = self.images[item]
        res = self.get_image_annotation(image)
        return res

    def get_image_annotation_by_image_name(self,file_name):
        if self.filename2image is None:
            self.filename2image = {}
            for image in self.images:
                tfile_name = image["file_name"]
                self.filename2image[tfile_name] = image

        image = self.filename2image[file_name]

        return self.get_image_annotation(image)

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
            if width <= 0 or height <= 0:
                num_annotations_skipped += 1
                continue
            if x<0 or x>=image_width  or y<0 or y>=image_height:
                num_annotations_skipped += 1
                continue
            if x + width > image_width:
                width = image_width-x
            if y + height > image_height:
                height = image_height-y

            category_id = int(object_annotations['category_id'])
            org_category_id = category_id
            if self.trans_label is not None:
                category_id = self.trans_label(category_id)
                if category_id is None:
                    continue

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
            category_names.append(str(self.category_index[org_category_id]['name'].encode('utf8'),encoding='utf-8'))
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

        if len(category_ids)==0:
            print("No annotation: ", full_path)
            sys.stdout.flush()
            return None,None,None,None,None,None,None,None,None
        img_shape = [image_height,image_width]
        category_ids = np.array(category_ids,dtype=np.int32)
        return full_path,img_shape,category_ids,category_names,boxes,binary_masks,area,is_crowd,num_annotations_skipped

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
                yield full_path,img_size,category_ids,boxes,is_crowd


class TorchCOCOData(COCOData):
    def __init__(self, img_dir, anno_path, absolute_coord=True):
        super().__init__(is_relative_coordinate=not absolute_coord)
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

if __name__ == "__main__":
    import img_utils as wmli
    import object_detection_tools.visualization as odv
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
