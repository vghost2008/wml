import argparse
import wml.wml_utils as wmlu
import os.path as osp
import os
import math
import random
import time
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('dst_dir', type=str, help='dit dir directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--suffix',
        type=str,
        default='xml',
        choices=['json', 'xml', 'txt'],
        help='annotation suffix')
    parser.add_argument(
        '--img-suffix',
        type=str,
        default=".jpg",
        help='img suffix')
    args = parser.parse_args()

    return args

def get_outdir_info(dir_path,img_suffix=".jpg"):
    files = wmlu.get_files(dir_path,suffix=img_suffix)
    datas = wmlu.MDict(dtype=list) 
    for file in files:
        name = wmlu.base_name(file)
        idx = name.rfind("_")
        kname = name[:idx]
        datas[kname].append(file)

    diff_datas = {}

    for k,v in dict(datas).items():
        if len(v)==1:
            continue
        z = set([wmlu.file_md5(f) for f in v])
        if len(z)>1:
            print(f"{v} md5 different")
            diff_datas[k] = v
            datas.pop(k)
        else:
            print(f"{v} md5 same")

    return datas,diff_datas

def process(src_dir,dst_dir,out_dir,suffix,img_suffix=".jpg"):
    ann_files = wmlu.get_files(src_dir,suffix=suffix)
    datas,diff_datas = get_outdir_info(dst_dir,img_suffix=img_suffix)
    os.makedirs(out_dir,exist_ok=True)
    nr = 0
    for afile in ann_files:
        name = wmlu.base_name(afile)
        if name not in datas and name not in diff_datas:
            print(f"Find {name} faild.")
            continue
        if name in datas:
            img_file = datas[name][0]
            ann_file = wmlu.change_suffix(img_file,suffix)
            if wmlu.file_md5(afile) == wmlu.file_md5(ann_file):
                continue
            '''print(f"{afile} changed")
            save_ann = osp.join(out_dir,name+"."+suffix)
            save_img = osp.join(out_dir,name+img_suffix)
            shutil.copy(afile,save_ann)
            wmlu.try_link(img_file,save_img)
            save_ann = osp.join(out_dir,name+"_old."+suffix)
            save_img = osp.join(out_dir,name+"_old"+img_suffix)
            shutil.copy(ann_file,save_ann)
            wmlu.try_link(img_file,save_img)'''
            for img_file in datas[name]:
                ann_file = wmlu.change_suffix(img_file,suffix)
                os.remove(ann_file)
                print(f"{afile} -> {ann_file}")
                shutil.copy(afile,ann_file)
            nr += 1
        elif name in diff_datas:
            print(f"{afile} is ambiguous with {diff_datas[name]} ")
    
    print(f"Total change {nr} files.")


if __name__ == "__main__":
    args = parse_args()
    process(args.src_dir,args.dst_dir,args.out_dir,args.suffix,args.img_suffix)

