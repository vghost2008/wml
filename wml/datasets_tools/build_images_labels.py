import os
import os.path as osp
import wml.wml_utils as wmlu
import argparse
import shutil
from wml.iotoolkit.labelme_toolkit_fwd import get_files,get_labels_set,set_reader
from wml.iotoolkit.reader import DecryptReader

'''
将数据集按images/labels分别存放,可指定labels, 如果没有指定自动获取所有labels
'''


def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="out dir")
    parser.add_argument("--labels",type=str,nargs="+",default=None,help="labels")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    try:
        set_reader(DecryptReader())
    except Exception as e:
        print(f"Set reader error: {e}")
    args = parse_args()
    wmlu.create_empty_dir(args.out_dir,remove_if_exists=False)
    files = get_files(args.src_dir)

    print(f"Total find {len(files)} files.")

    img_save_dir = osp.join(args.out_dir,"images")
    json_save_dir = osp.join(args.out_dir,"labels")
    classes_save_path = osp.join(args.out_dir,"classes.txt")
    print(f"Save dir:",img_save_dir,json_save_dir)

    for imgf,annf in files:
        bn = wmlu.get_relative_path(imgf,args.src_dir)
        print(bn,imgf,args.src_dir)
        save_path = osp.join(img_save_dir,bn)
        dir_path = osp.dirname(save_path)
        os.makedirs(dir_path,exist_ok=True)
        shutil.copy(imgf,save_path)

        bn = wmlu.get_relative_path(annf,args.src_dir)
        save_path = osp.join(json_save_dir,bn)
        dir_path = osp.dirname(save_path)
        os.makedirs(dir_path,exist_ok=True)
        shutil.copy(annf,save_path)

    labels = args.labels
    if labels is None or len(labels) == 0:
        labels = get_labels_set(files)
    if len(labels)>0:
        info = ",".join(labels)
        print(f"Classes: {info}")
        with open(classes_save_path,"w") as f:
            f.write(info)
    print(f"Save dir {args.out_dir}")


