import argparse
import os.path as osp
import os
import wml.wml_utils as wmlu
from wml.iotoolkit import get_auto_dataset_suffix
import shutil
'''
如果没有设置save_dir:将标注文件及图像从src_dir拷贝到ref_dir
如果设置了save_dir: 对比src_dir与ref_dir的图像存在与更新时间,将数据拷贝到save_dir
'''

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('ref_dir', type=str, help='output rawframe directory')
    parser.add_argument('--save-dir', '-sd', type=str, help='output rawframe directory')
    parser.add_argument(
        '--base-name',
        '-bn',
        action='store_true',
        help='save file with base name.')
    parser.add_argument(
        '--suffix',
        '-s',
        type=str,
        default="auto",
        help='no label name.')
    parser.add_argument(
        '--update',
        '-u',
        action='store_true',
        help='update.')
    parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        help='force use base name.')
    args = parser.parse_args()

    return args


def get_files(data_dir,args):
    data_dir = osp.abspath(osp.expanduser(data_dir))
    files = wmlu.get_img_files(data_dir)
    files = [(x,wmlu.change_suffix(x,args.suffix)) for x in files]
    files = list(filter(lambda x:osp.exists(x[1]),files))

    return files

def newer(lhfile,rhfile):
    return (osp.getmtime(lhfile)>osp.getmtime(rhfile)) or (osp.getsize(lhfile) != osp.getsize(rhfile))

if __name__ == "__main__":

    args = parse_args()
    if args.suffix == "auto":
        args.suffix = get_auto_dataset_suffix(args.src_dir)
    
    files = get_files(args.src_dir,args)

    if args.save_dir is None or len(args.save_dir)==0:
        args.save_dir = args.ref_dir

    dfiles = get_files(args.ref_dir,args)
    dst_data = {}

    for img_f,ann_f in dfiles:
        bn = osp.basename(img_f)
        if bn in dst_data:
            wmlu.print_error("{bn} is already exists in ref dir, {img_f},{dst_data[bn][0]}")
            if not args.force:
                exit(-1)
        else:
            dst_data[bn] = (img_f,ann_f)

    files2copy = []
    for img_f,ann_f in files:
        rpath = wmlu.get_relative_path(ann_f,args.src_dir)
        dann_f = osp.join(args.ref_dir,rpath)
        if osp.exists(dann_f):
            if not newer(ann_f,dann_f):
                continue
            else:
                print(f"{ann_f} have been modified, need to copy")
        elif args.base_name:
            bn = osp.basename(img_f)
            if bn in dst_data:
                dann_f = dst_data[bn][1]
                if not newer(ann_f,dann_f):
                    continue
                else:
                    print(f"{ann_f} have been modified, need to copy")
            else:
                print(f"{ann_f} not exists in {args.ref_dir}, need to copy")
        else:
            print(f"{ann_f} not exists in {args.ref_dir}, need to copy")
        files2copy.append([img_f,ann_f])

    print(f"Total find {len(files)} in src dir, find {len(dfiles)} in ref dir")
    print(f"{len(files2copy)} need copy:")

    if len(files2copy) == 0:
        exit(0)

    wmlu.show_list(files2copy[:20])
    if len(files2copy)>20:
        print("...")

    ans = input(f"Copy {len(files2copy)} files to {args.save_dir} [Y/N]?\n")
    if ans.lower().strip() == "n":
        print(f"Cancel copy files.")
        exit(0)

    for img_f,ann_f in files2copy:
        dimg_f = wmlu.get_relative_path(img_f,args.src_dir)
        dann_f = wmlu.change_suffix(dimg_f,args.suffix)

        save_img_f = osp.join(args.save_dir,dimg_f)
        save_ann_f = osp.join(args.save_dir,dann_f)

        save_dir = osp.dirname(save_img_f)
        os.makedirs(save_dir,exist_ok=True)

        print(f"{img_f} --> {save_img_f}")
        shutil.copy2(img_f,save_img_f)
        print(f"{ann_f} --> {save_ann_f}")
        shutil.copy2(ann_f,save_ann_f)




