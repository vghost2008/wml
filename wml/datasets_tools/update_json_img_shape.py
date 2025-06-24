import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
import matplotlib.pyplot as plt
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData
from wml.iotoolkit.fast_labelme import FastLabelMeData
import argparse
import os.path as osp
import os
import wml.wml_utils as wmlu
import wml.wtorch.utils as wtu
from wml.iotoolkit import get_auto_dataset_type
from wml.iotoolkit.labelme_toolkit_fwd import get_files
import numpy as np
import shutil
import json

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, nargs="+",help='source video directory')
    args = parser.parse_args()

    return args

def main(args):
    dir = args.src_dir
    files = get_files(dir)
    for img_file,json_file in files:
        h,w = wmli.get_img_size(img_file)
        with open(json_file,"r") as f:
            json_data = json.load(f)
            img_width = int(json_data["imageWidth"])
            img_height = int(json_data["imageHeight"])
            if img_width == w and img_height == h:
                continue
        print(f"Update {json_file} img size from {(img_height,img_width)} to {(h,w)}")
        json_data["imageWidth"] = w
        json_data["imageHeight"] = h
        with open(json_file,"w") as f:
            json.dump(json_data,f)
            print(f"Save {json_file}")
            pass


if __name__ == "__main__":
    args = parse_args()
    main(args)

