import argparse
import glob
import os.path as osp
import wml.wml_utils as wmlu
from wml.iotoolkit import get_auto_dataset_suffix
import wml.img_utils as wmli
import shutil

'''
指定标注文件目录，从图像文件目标中拷贝base name相同的图像文件到标注文件目录
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='arguments')
    parser.add_argument('ann_dir', default=None,type=str,help='ann_dir')
    parser.add_argument('img_dir', default=None,type=str,help='img_dir')
    parser.add_argument('--suffix', default=wmli.BASE_IMG_SUFFIX,type=str,help='img suffix')
    parser.add_argument('--type', default="auto",type=str,help='img_dir')
    parser.add_argument('-l','--level', default=0,type=int,help='test parent level number')
    args = parser.parse_args()
    return args

def get_name_key(path,level=0):
    path = osp.abspath(path)
    bn = wmlu.base_name(path)
    bn = bn.replace("\\","")
    bn = wmlu.remove_non_ascii(bn)
    if 0 == level:
        return bn
    names = path.split(osp.sep)[:-1]
    names = names[-level:]
    names = names+[bn]
    dir_name = str(osp.sep).join(names)
    return dir_name

def get_all_imgs(img_dir,level=0,img_suffix=".jpg;;.jpeg;;.png;;.bmp"):
    files = wmlu.get_files(img_dir,suffix=img_suffix)
    res = {}
    for f in files:
        basename = get_name_key(f,level)
        if basename in res:
            d = res[basename]
            if isinstance(d,str):
                if wmlu.file_md5(d) == wmlu.file_md5(f):
                    print(f"{d} and {f} is the same file.")
                    continue
                d = [d]
            res[basename] = d+[f]
        else:
            res[basename] = f
    return res

def copy_imgfiles(ann_dir,img_dir,level=0,img_suffix=wmli.BASE_IMG_SUFFIX,ann_type=".xml"):
    if ann_type == "auto":
        ann_type = "."+get_auto_dataset_suffix(ann_dir)
    all_img_files = get_all_imgs(img_dir,level=level,img_suffix=img_suffix)
    imgs_for_ann = get_all_imgs(ann_dir,img_suffix=img_suffix)
    imgs_for_ann = [osp.abspath(f) for f in imgs_for_ann]
    imgs_for_ann = [wmlu.change_suffix(f,ann_type[1:]) for f in imgs_for_ann]
    _xml_files = wmlu.get_files(ann_dir,suffix=ann_type)
    _xml_files = [osp.abspath(f) for f in _xml_files]
    xml_files = []
    for xf in _xml_files:
        if xf in imgs_for_ann:
            print(f"Img for {xf} exists, skip.")
            continue
        xml_files.append(xf)
    copy_nr = 0
    error_nr = 0
    not_found_nr = 0
    #wmlu.show_list(list(all_img_files.keys()))
    for xf in xml_files:
        base_name = get_name_key(xf,level)
        print(base_name)
        if base_name in all_img_files:
            files = all_img_files[base_name]
            if not isinstance(files,list):
                cur_dir = osp.dirname(xf)
                print(f"{files} --> {cur_dir}")
                save_path = osp.join(cur_dir,wmlu.base_name(xf)+osp.splitext(files)[-1])
                shutil.copy(files,save_path)
                copy_nr += 1
            else:
                print(f"ERROR: Find multi img files for {xf}, img files {files}, key={base_name}")
                error_nr += 1
        else:
            print(f"ERROR: Find img file for {xf} faild, key={base_name}")
            not_found_nr += 1

    print(f"total copy {copy_nr} files, {error_nr} multi files, {not_found_nr} not found files.")

if __name__ == "__main__":
    args = parse_args()
    copy_imgfiles(args.ann_dir,args.img_dir,level=args.level,ann_type=args.type,img_suffix=args.suffix)
