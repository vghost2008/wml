import argparse
import glob
import os.path as osp
import wml_utils as wmlu

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
            res[basename] = res[basename]+[f]
        else:
            res[basename] = f
    return res

def copy_imgfiles(xml_dir,img_dir,img_suffix=".jpg",ann_type=".xml"):
    xml_files = glob.glob(osp.join(xml_dir,"*"+ann_type))
    all_img_files = get_all_imgs(img_dir)
    for xf in xml_files:
        base_name = wmlu.base_name(xf)
        img_name = base_name+img_suffix
        img_path = osp.join(img_dir,img_name)
        dst_img_path = osp.join(xml_dir,img_name)
        if osp.exists(img_path):
            wmlu.try_link(img_path,dst_img_path)
        elif base_name in all_img_files:
            files = all_img_files[base_name]
            if not isinstance(files,list):
                wmlu.try_link(files,xml_dir)
            else:
                print(f"ERROR: Find multi img files for {xf}, img files {files}")
        else:
            print(f"ERROR: Find img file for {xf} faild.")

if __name__ == "__main__":
    args = parse_args()
    copy_imgfiles(args.ann_dir,args.img_dir,ann_type=args.type)