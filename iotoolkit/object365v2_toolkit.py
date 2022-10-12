from iotoolkit.coco_toolkit import *
import os.path as osp

def trans_file_name(filename,image_dir):
    names = filename.split("/")[-2:]
    return osp.join(*names)

class TorchObject365V2(TorchCOCOData):
    def __init__(self, img_dir, anno_path):
        COCOData.trans_file_name = trans_file_name
        super().__init__(img_dir, anno_path)
        COCOData.trans_file_name = None
