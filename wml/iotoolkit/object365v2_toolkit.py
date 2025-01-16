from wml.iotoolkit.coco_toolkit import *
import os.path as osp
from .o365_to_coco import *
from .coco_data_fwd import ID_TO_TEXT
from functools import partial


def trans_file_name(filename,image_dir):
    '''
    object365文件名使用了类似于:'images/v1/patch8/objects365_v1_00420917.jpg'格式，而实际存的为
    data_root/train/pathch8/objects365_v1_00420917.jpg这种格式，这里仅保留最后的目录名及文件名，也就是输出
    pathch8/objects365_v1_00420917.jpg
    注: read_data的image_dir需要设置为data_root/train/
    '''
    names = filename.split("/")[-2:]
    return osp.join(*names)

def trans_label2coco(id,fn=None):
    if id in o365_id_to_coco_id:
        id = o365_id_to_coco_id[id]
        if fn is not None:
            return fn(id)
        else:
            return id
    else:
        return None

class Object365V2(COCOData):
    def __init__(self,is_relative_coordinate=False,trans2coco=False,remove_crowd=True,trans_label=None):
        super().__init__(is_relative_coordinate=is_relative_coordinate,remove_crowd=remove_crowd,trans_label=trans_label,include_masks=False)
        if trans2coco and trans_label is not None:
            self.trans_label = partial(trans_label2coco,fn=trans_label)
        elif trans2coco:
            self.trans_label = trans_label2coco
        elif trans_label is not None:
            self.trans_label = trans_label
        self.trans_file_name = trans_file_name
        if trans2coco:
            self.id2name = {}
            for k,info in ID_TO_TEXT.items():
                self.id2name[k] = info['name']

    def read_data(self,annotations_file,image_dir=None):
        if image_dir is None:
            image_dir = osp.dirname(osp.abspath(annotations_file))
        return super().read_data(annotations_file,image_dir)

class TorchObject365V2(Object365V2):
    def __init__(self, img_dir, anno_path,trans2coco=False,trans_label=None):
        super().__init__(is_relative_coordinate=False,trans2coco=trans2coco,trans_label=trans_label)
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
