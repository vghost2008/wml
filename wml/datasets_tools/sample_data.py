import os.path as osp
import os
import random
import wml.wml_utils as wmlu
import shutil
from argparse import ArgumentParser
from wml.iotoolkit import get_auto_dataset_suffix
from split_train_val import get_labels
import time

'''
从指定目录下采样sample-nr个文件，或者(sub-dir==True)从指定目标的每一个子目录中采样sample-nr个文件，并把采样的文件分子目录存在
save_dir中
如果指定--by-labels那么从每个类采样nr个数据
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str, default="/home/wj/ai/mldata1/B7mura/datas/try_s0",help='source video directory')
    parser.add_argument('save_dir', type=str, help='save dir')
    parser.add_argument("-nr","--sample-nr",type=int,help="sample nr")
    parser.add_argument("--suffix",type=str,default="auto", help="annotation suffix")
    parser.add_argument('--sub-dir', action='store_true',help='whether to sample data in sub dirs.')
    parser.add_argument(
        '-bl',
        '--by-labels',
        action='store_true',
        help='split by labels xmls') #每个类别采样nr个文件
    parser.add_argument(
        '-kk',
        '--keep-key',
        action='store_true',
        help='keep key in save dir')
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
    split_nr当前没有使用
    return:
      在split_nr==None的情况下：
      dict key=sub_dir_name,vlaue=采集的文件
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

def save_data(data,save_dir,suffix=None,keep_key=False):
    for k,v in data.items():
        if keep_key:
            tsd = osp.join(save_dir,str(k))  #保存目录中包含key(子目录中采样就为子目录的名字)
        else:
            tsd = save_dir #保存目录不包含key
        wmlu.create_empty_dir(tsd,False)
        for f in v:
            '''dir_name = wmlu.base_name(osp.dirname(f))
            if dir_name == "":
                print(f"Get dir name faild {f}.")
            name = dir_name+"_"+osp.join(osp.basename(f))
            os.link(f,osp.join(tsd,name))'''
            save_path = osp.join(tsd,osp.basename(f))
            shutil.copy(f,save_path)
            ann_path = wmlu.change_suffix(f,suffix)
            if osp.exists(ann_path):
                save_path = wmlu.change_suffix(save_path,suffix)
                shutil.copy(ann_path,save_path)

def sample_by_labels(args):
    src_dir = args.src_dir
    img_files = wmlu.get_img_files(src_dir)
    ann_files = [wmlu.change_suffix(x,args.suffix) for x in img_files]

    all_files = list(zip(img_files,ann_files))
    wmlu.show_list(all_files[:100])
    if len(all_files)>100:
        print("...")

    allow_empty = False
    if args.suffix == "none":
        allow_empty = True

    if not allow_empty:
        all_files = list(filter(lambda x:osp.exists(x[1]),all_files))

    label2files = wmlu.MDict(dtype=list)
    for i,(img_f,ann_f) in enumerate(all_files):
        labels = get_labels(ann_f,args.suffix)
        for l in set(labels):
            label2files[l].append(img_f)
        if len(labels)==0:
            label2files['NONE'].append(img_f)
    res = {}
    for k,v in label2files.items():
        if len(v)>args.sample_nr:
            random.shuffle(v)
            v = v[:args.sample_nr]
        res[k] =  v
    
    return res


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.src_dir
    save_dir = args.save_dir
    wmlu.create_empty_dir(save_dir,False)

    random.seed(time.time())

    if args.suffix == "auto":
        args.suffix = get_auto_dataset_suffix(data_dir)
    if args.by_labels:
        data = sample_by_labels(args)
    else:
        data = sample_in_dir(data_dir,args.sample_nr,sample_in_sub_dirs=args.sub_dir)

    save_data(data,save_dir,suffix=args.suffix,keep_key=args.keep_key)

    print(f"Save_path {save_dir}")
