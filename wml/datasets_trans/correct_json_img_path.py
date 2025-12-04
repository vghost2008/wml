import argparse
from wml.iotoolkit.labelme_toolkit_fwd import get_files
import json
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="correct json path")
    parser.add_argument("ann_dir",type=str,help="ann path")
    args = parser.parse_args()
    return args


def trans_data(ann_dir):
    files = get_files(ann_dir)
    print(f"Find {len(files)} in {ann_dir}")
    for img_path,json_path in files:
        try:
            with open(json_path,"r") as f:
                data = json.load(f)
            json_img_path = data['imagePath']
            if json_img_path  == osp.basename(img_path):
                print(f"OK: {json_path} {data['imagePath']}")
                continue
            data['imagePath'] = osp.basename(img_path)
            print(f"{json_path} {data['imagePath']}")
            with open(json_path,"w") as f:
                json.dump(data,f)
        except Exception as e:
            print(f"ERROR: {e}")
            pass


if __name__ == "__main__":
    args = parse_args()
    ann_path = args.ann_dir
    trans_data(ann_path)

