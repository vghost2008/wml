import argparse
import glob
import os.path as osp
import wml_utils as wmlu

'''
指定标注文件目录，从图像文件目标中拷贝base name相同的图像文件到标注文件目录
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='arguments')
    parser.add_argument('ann_dir', default=None,type=str,help='ann_dir')
    parser.add_argument('img_dir', default=None,type=str,help='img_dir')
    parser.add_argument('--type', default=".xml",type=str,help='img_dir')
    args = parser.parse_args()
    return args

def get_all_imgs(img_dir,img_suffix=".jpg;;.jpeg;;.png;;.bmp"):
    files = wmlu.get_files(img_dir,suffix=img_suffix)
    res = {}
    for f in files:
        basename = wmlu.base_name(f)
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

def copy_imgfiles(xml_dir,img_dir,img_suffix=".jpg",ann_type=".xml"):
    xml_files = glob.glob(osp.join(xml_dir,"*"+ann_type))
    all_img_files = get_all_imgs(img_dir)
    copy_nr = 0
    error_nr = 0
    not_found_nr = 0
    for xf in xml_files:
        base_name = wmlu.base_name(xf)
        if base_name in all_img_files:
            files = all_img_files[base_name]
            if not isinstance(files,list):
                wmlu.try_link(files,xml_dir)
                copy_nr += 1
            else:
                print(f"ERROR: Find multi img files for {xf}, img files {files}")
                error_nr += 1
        else:
            print(f"ERROR: Find img file for {xf} faild.")
            not_found_nr += 1

    print(f"total copy {copy_nr} files, {error_nr} multi files, {not_found_nr} not found files.")

if __name__ == "__main__":
    args = parse_args()
    copy_imgfiles(args.ann_dir,args.img_dir,ann_type=args.type)
