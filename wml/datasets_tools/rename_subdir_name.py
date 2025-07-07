import argparse
import wml.wml_utils as wmlu
import os.path as osp
import shutil

'''
把数据集中的文件重命名为相应的标签
'''

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '-ad',
        '--all-dir',
        action='store_true',
        help='rename all dir') #默认只重命名叶子节点的目录名,通过这个选项可以重命名所有的目录名

    args = parser.parse_args()

    return args

def simple_key(k):
    k = k.lower()
    k = k.replace("-","")
    k = k.replace("#","")
    k = k.replace(" ","")
    return k

def main(args,trans_dict):
    new_tran_dict = {}
    for k,v in trans_dict.items():
        k = simple_key(k)
        new_tran_dict[k] = v
    trans_dict = new_tran_dict

    if args.all_dir:
        dirs = wmlu.recurse_get_subdir_in_dir(args.src_dir)
    else:
        dirs = wmlu.get_leaf_dirs(args.src_dir)

    for dir in dirs:
        bn = wmlu.base_name(dir)
        bn = simple_key(bn)

        if bn not in trans_dict:
            wmlu.print_warning(f"Skip {dir}")
            continue

        '''
        rpath = wmlu.get_relative_path(dir,args.src_dir)
        dirname = osp.dirname(rpath)
        save_path = osp.join(args.out_dir,dirname,trans_dict[bn])
        '''
        save_path = osp.join(args.out_dir,trans_dict[bn])
        print(f"Copy dir {dir} --> {save_path}")
        wmlu.copy_files2dir(dir,save_path)

if __name__ == "__main__":
    args = parse_args()
    trans_dict = {
    "0-False":"OK",
    "1-粗糙":"CC",
    "2-划伤":"HS",
    "3-镍颗粒":"NKL",
    "4-漏镀":"LD",
    "5-异物":"YW",
    "6-阻镀":"ZD",
    "7-边缘阻镀":"BYZD",
    "10-焊盘缺陷":"HPQX",
    "11-水渍":"SZ",         
    "12-镍金线":"NJX",
    "13-Bonding异常":"BondingYC",
    "13-bonding不良":"BondingYC",
     }

    
    main(args,trans_dict)
    
    