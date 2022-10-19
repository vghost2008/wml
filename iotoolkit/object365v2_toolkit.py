from iotoolkit.coco_toolkit import *
import os.path as osp

class Object365V2TransFile:

    @staticmethod
    def apply(filename,image_dir):
        names = filename.split("/")[-2:]
        return osp.join(*names)

class Object365V2(COCOData):
    def __init__(self,is_relative_coordinate=False):
        super().__init__(is_relative_coordinate=is_relative_coordinate)
        self.trans_file_name = Object365V2TransFile()

class TorchObject365V2(Object365V2):
    def __init__(self, img_dir, anno_path):
        super().__init__(is_relative_coordinate=False)
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
