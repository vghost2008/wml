import os.path as osp
import os
import wml.wml_utils as wmlu
import math
import random
import time
import argparse
from wml.iotoolkit.pascal_voc_toolkit import read_voc_xml
from wml.iotoolkit.labelme_toolkit import read_labelme_data
from wml.iotoolkit import get_auto_dataset_suffix,check_dataset_dir
from wml.iotoolkit import ImageFolder
import shutil
import copy

'''
将指定数据集按每一个样本中包含的标签名拷贝到相应的子目录中
如果一个样本有多个不同类型的标签，那么会拷贝到多个相应的子目录中
'''

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, nargs="+",help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--splits',
        type=float,
        nargs="+",
        default=[0.1,0.9], #val, train
        help='split percent')
    parser.add_argument(
        '--max-nr',
        type=int,  #指定val的最大值
        help='split set max nr')
    parser.add_argument(
        '--suffix',
        type=str,
        default='auto',
        choices=['json', 'xml', 'txt'],
        help='annotation suffix')
    parser.add_argument(
        '--img-suffix',
        type=str,
        default=".jpg;;.jpeg;;.bmp;;.png",
        help='img suffix')
    parser.add_argument(
        '-ae',
        '--allow-empty',
        action='store_true',
        help='include img files without annotation')
    parser.add_argument(
        '--no-imgs',
        action='store_true',
        help='only split xmls')
    parser.add_argument(
        '-bl',
        '--by-labels',
        action='store_true',
        help='split by labels xmls')
    parser.add_argument(
        '--silent',
        action='store_true',
        help='silent')
    parser.add_argument(
        '-bn',
        '--basename',
        action='store_true',
        help='use basename, instead of releative path.')
    parser.add_argument(
        '-txt',
        '--txt',
        action='store_true',
        help='save txt files, instead copy files.')
    args = parser.parse_args()

    args = parser.parse_args()

    return args


def copy_files(files,save_dir,add_nr,src_dir,use_basename=False):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    for i,(imgf,annf) in enumerate(files):
        if osp.splitext(annf)[-1] == ".none":
            label = ImageFolder.get_label(imgf)
            basename = osp.join(label,wmlu.base_name(imgf))
        elif use_basename:
            basename = wmlu.base_name(imgf)
        else:
            basename = wmlu.get_relative_path(imgf,src_dir)
            basename = osp.splitext(basename)[0]

        if add_nr:
            basename = basename+f"_{i}"
        suffix = osp.splitext(imgf)[1]
        save_path = osp.join(save_dir,basename+suffix)
        cur_save_dir = osp.dirname(save_path)
        os.makedirs(cur_save_dir,exist_ok=True)

        print(imgf,"--->",save_path)
        shutil.copy(imgf,save_path)

        if not osp.exists(annf):
            continue

        suffix = osp.splitext(annf)[1]
        print(annf,"--->",osp.join(save_dir,basename+suffix))
        shutil.copy(annf,osp.join(save_dir,basename+suffix))

def write_txt(files,save_dir,name):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir,name+".dl")
    with open(save_path,"a") as f:
        for i,(imgf,annf) in enumerate(files):
            f.write(imgf+","+annf+"\n")

def get_labels(ann_file,suffix):
    if suffix == "xml":
        return read_voc_xml(ann_file)[2]
    elif suffix == "json":
        image,annotation_list = read_labelme_data(ann_file,label_text_to_id=None,mask_on=False)
        labels = [x['category_id'] for x in annotation_list]
        return labels
    elif suffix == "none":
        return [ImageFolder.get_label(ann_file)]


def split_one_set(src_files,src_dir,save_dir,splits,args,copyed_files=None,use_basename=False):
    
    splits = copy.deepcopy(splits)
    max_nr = args.max_nr


    img_files = [x[0] for x in src_files]
    ann_files = [x[1] for x in src_files]
    basenames = [wmlu.base_name(x) for x in img_files]
    if len(basenames) == len(set(basenames)):
        add_nr = False
    else:
        add_nr = True
        print(f"Need to add nr name")
    all_files = src_files
    if not args.allow_empty:
        all_files = list(filter(lambda x:osp.exists(x[1]),all_files))
    os.makedirs(save_dir,exist_ok=True)
    random.seed(int(time.time()))
    random.shuffle(all_files)

    use_percent = True
    total_nr = 0
    for v in splits:
        if v<0 or v>=1:
            use_percent = False
            if v>0:
                total_nr += v
    old_splits = copy.deepcopy(splits)
    for i,v in enumerate(splits):
        if v<0:
            splits[i] = max(len(all_files)-total_nr,0)
            print(f"Update splits from {old_splits} to {splits}")
            break
    
    names = ['val','train']

    for i,v in enumerate(splits):
        if i<=1:
            cur_name = names[i]
        else:
            cur_name = "data_"+str(v)
        t_save_dir = osp.join(save_dir,cur_name)
        if i<len(splits)-1:
            if use_percent:
                t_nr = int(v*len(all_files)+0.5)
            else:
                t_nr = int(v)
            if i==0 and max_nr is not None and max_nr>0: #仅对第一个，也就是val使用max_nr
                t_nr = min(t_nr,max_nr)

            tmp_files = all_files[:t_nr]
            all_files = all_files[t_nr:]
        else: #最后一个分组包含剩余的所有文件
            tmp_files = all_files
            t_nr = len(tmp_files)
        print(f"split {v} as {t_nr} files")
        wmlu.show_list(tmp_files)
        if copyed_files is not None:
            _tmp_files = list(tmp_files)
            tmp_files = []
            for img_file,ann_file in _tmp_files:
                if img_file not in copyed_files:
                    tmp_files.append((img_file,ann_file))
                    copyed_files.add(img_file)

        if args.txt:
            write_txt(tmp_files,save_dir,cur_name)
        else:
            copy_files(tmp_files,t_save_dir,add_nr,src_dir=src_dir,use_basename=use_basename)

    return copyed_files

if __name__ == "__main__":
    args = parse_args()

    if isinstance(args.src_dir,(list,tuple)) and len(args.src_dir)==1:
        args.src_dir = args.src_dir[0]

    src_dir = check_dataset_dir(args.src_dir)
    print(f"src dir: {src_dir}")
    args.suffix = get_auto_dataset_suffix(src_dir,args.suffix)
    if not args.no_imgs:
        img_files = wmlu.get_files(src_dir,suffix=args.img_suffix)
        ann_files = [wmlu.change_suffix(x,args.suffix) for x in img_files]
    else:
        ann_files = wmlu.get_files(src_dir,suffix=args.suffix)
        img_files = [wmlu.change_suffix(x,"jpg") for x in ann_files]
    basenames = [wmlu.base_name(x) for x in img_files]
    if len(basenames) == len(set(basenames)):
        add_nr = False
    else:
        add_nr = True
    all_files = list(zip(img_files,ann_files))
    wmlu.show_list(all_files[:100])
    if len(all_files)>100:
        print("...")

    if args.suffix == "none":
        args.allow_empty = True

    if not args.allow_empty:
        all_files = list(filter(lambda x:osp.exists(x[1]),all_files))
    wmlu.create_empty_dir_remove_if(args.out_dir)
    save_dir = args.out_dir
    random.seed(int(time.time()))
    random.shuffle(all_files)
    print(f"Find {len(all_files)} files")

    if args.by_labels:
        label2files = wmlu.MDict(dtype=list)
        for i,(img_f,ann_f) in enumerate(all_files):
            labels = get_labels(ann_f,args.suffix)
            for l in set(labels):
                label2files[l].append((img_f,ann_f))
            if len(labels)==0:
                label2files['NONE'].append((img_f,ann_f))
        copyed_files = set()
        for k,v in label2files.items():
            copyed_files = split_one_set(v,src_dir,save_dir,args.splits,args,copyed_files=copyed_files,use_basename=args.basename)
    else:
        split_one_set(all_files,src_dir,save_dir,args.splits,args,use_basename=args.basename)


