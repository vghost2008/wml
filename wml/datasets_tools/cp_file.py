import argparse
import os.path as osp
import os
import wml.wml_utils as wmlu
from wml.iotoolkit import get_auto_dataset_suffix
import shutil
'''
如果没有设置save_dir:将指定文件src_dir拷贝到ref_dir
如果设置了save_dir: 对比src_dir与ref_dir的图像存在与更新时间,将数据拷贝到save_dir
'''

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('save_dir',type=str, help='output rawframe directory')
    parser.add_argument(
        '--base-name',
        '-bn',
        action='store_true',
        help='save file with base name.')
    parser.add_argument(
        '--suffix',
        '-s',
        type=str,
        default="json",
        help='file suffix.')
    args = parser.parse_args()

    return args


def cp_file(data_dir,save_dir,suffix,args):
    files = wmlu.get_files(data_dir,suffix=suffix)
    for f in files:
        if args.base_name:
            save_name = osp.basename(f)
        else:
            save_name = wmlu.get_relative_path(f,data_dir)
        save_path = osp.join(save_dir,save_name)
        wmlu.make_dir_for_file(save_path)
        print(f"{f} --> {save_path}")
        shutil.copy2(f,save_path)


if __name__ == "__main__":

    args = parse_args()
    cp_file(args.src_dir, args.save_dir, args.suffix, args)
