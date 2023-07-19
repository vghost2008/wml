import os.path as osp
import os
import random
import wml_utils as wmlu
import shutil
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, default="/home/wj/ai/mldata1/B7mura/datas/try_s0",help='source video directory')
    parser.add_argument('save_dir', type=str, help='save dir')
    parser.add_argument("--sample-nr",type=int,help="sample nr")
    parser.add_argument("--suffix",type=str,default="xml", help="annotation suffix")
    parser.add_argument('--sample-in-sub-dirs', action='store_true',help='whether to sample data in sub dirs.')
    args = parser.parse_args()
    return args

def sample_in_one_dir(dir_path,nr):
    print(f"Sample in {dir_path}")
    files = wmlu.recurse_get_filepath_in_dir(dir_path,suffix=".jpg;;.jpeg;;.bmp;;.png")
    random.shuffle(files)
    return files[:nr]

def append_to_dict(dict,key,data):
    if key in dict:
        dict[key].extend(data)
    else:
        dict[key] = data

def sample_in_dir(dir_path,nr,split_nr=None,sample_in_sub_dirs=True):
    '''
    sample data in dir_path's sub dirs
    sample nr images in each sub dir, if split_nr is not None, sampled nr images will be split 
    to split_nr part and saved in different dir
    '''
    res = {}
    if sample_in_sub_dirs:
        dirs = wmlu.get_subdir_in_dir(dir_path,absolute_path=True)
    else:
        dirs = [dir_path]
    print(f"Find dirs in {dir_path}")
    wmlu.show_list(dirs)

    for dir in dirs:
        data = sample_in_one_dir(dir,nr)
        if split_nr is None:
            #append_to_dict(res,0,data)
            append_to_dict(res,wmlu.base_name(dir),data)
        else:
            data = wmlu.list_to_2dlistv2(data,split_nr)
            for i,d in enumerate(data):
                append_to_dict(res,i,d)

    return res

def save_data(data,save_dir,suffix=None):
    for k,v in data.items():
        tsd = osp.join(save_dir,str(k))
        #tsd = save_dir
        wmlu.create_empty_dir(tsd,False)
        for f in v:
            '''dir_name = wmlu.base_name(osp.dirname(f))
            if dir_name == "":
                print(f"Get dir name faild {f}.")
            name = dir_name+"_"+osp.join(osp.basename(f))
            os.link(f,osp.join(tsd,name))'''
            wmlu.try_link(f,tsd)
            ann_path = wmlu.change_suffix(f,suffix)
            if osp.exists(ann_path):
                shutil.copy(ann_path,tsd)
            #shutil.copy(f,osp.join(tsd,name))

if __name__ == "__main__":
    args = parse_args()
    data_dir = args.src_dir
    save_dir = args.save_dir
    wmlu.create_empty_dir(save_dir,False)
    data = sample_in_dir(data_dir,args.sample_nr,sample_in_sub_dirs=args.sample_in_sub_dirs)
    print(f"Save_path {save_dir}")
    save_data(data,save_dir,suffix=args.suffix)