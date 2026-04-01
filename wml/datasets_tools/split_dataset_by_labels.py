import argparse
import wml.wml_utils as wmlu
import os.path as osp
import os
import math
import random
import time
from wml.iotoolkit.pascal_voc_toolkit import read_voc_xml
from wml.iotoolkit.labelme_toolkit import read_labelme_data
from wml.iotoolkit import get_auto_dataset_suffix, get_ann_file_path, get_img_file_path
from wml.iotoolkit.image_folder import ImageFolder
import wml.img_utils as wmli
import shutil
import re

'''
将指定数据集按每一个样本中包含的标签名拷贝到相应的子目录中
如果一个样本有多个不同类型的标签，那么会拷贝到多个相应的子目录中
如果有重复的文件名,文件名后会加上序号
'''

def parse_args():
    parser = argparse.ArgumentParser(description='split dataset')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--suffix',
        type=str,
        default='auto',
        choices=['json', 'xml', 'txt','auto'],
        help='annotation suffix')
    parser.add_argument(
        '--img-suffix',
        type=str,
        default=wmli.BASE_IMG_SUFFIX,
        help='img suffix')
    parser.add_argument(
        '--allow-empty',
        action='store_true',
        help='include img files without annotation')
    parser.add_argument(
        '--no-imgs',
        action='store_true',
        help='only split xmls')
    parser.add_argument(
        '-bn',
        '--base-name',
        action='store_true',
        help='only split xmls')
    parser.add_argument(
        '--silent',
        action='store_true',
        help='silent')
    args = parser.parse_args()

    args = parser.parse_args()

    return args

def copy_files(imgf,annf,save_dir,add_nr,silent=False,allow_empty=False,args=None):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if args is not None:
        if args.base_name:
            basename = wmlu.base_name(imgf)
        else:
            basename = wmlu.get_relative_path(imgf,args.src_dir)
            basename = osp.splitext(basename)[0]
    else:
        basename = wmlu.base_name(imgf)
    suffix = osp.splitext(imgf)[1]
    save_path = osp.join(save_dir,basename+suffix)
    if add_nr and osp.exists(save_path):
        basename = basename+f"_{i}"
        save_path = osp.join(save_dir,basename+suffix)
    t_dir = osp.dirname(save_path)
    if not osp.exists(t_dir):
        os.makedirs(t_dir)
    if osp.exists(imgf):
        if not silent:
            print(imgf,"--->",save_path)
        shutil.copy(imgf,save_path)

    if allow_empty and not osp.exists(annf):
        return

    suffix = osp.splitext(annf)[1]
    if not silent:
        print(annf,"--->",osp.join(save_dir,basename+suffix))
    shutil.copy(annf,osp.join(save_dir,basename+suffix))

def get_labels(ann_file,suffix):
    if suffix == "xml":
        labels = read_voc_xml(ann_file)[2]
    elif suffix == "json":
        image,annotation_list = read_labelme_data(ann_file,label_text_to_id=None,mask_on=False)
        labels = [x['category_id'] for x in annotation_list]
    elif suffix == "none":
        labels = [ImageFolder.get_label(ann_file)]
    if len(labels)==0:
        labels = ['NONE']

    return labels

_label_text2id = {
        "B1.F": 0,
        "D1.S": 1,
        "D1.U": 2,
        "E3.W": 3,
        "G1.U": 4,
        "SD1L1.R": 5,
        "ACTL1.U": 6,
        "GA1L1.R": 7,
        "GA2L1.R": 8,
        "SD2L1.R": 9,
        "BSML1.U": 10,
        "GA1L2.U": 11,
        "GA2L3.U": 12,
        "GA1L3.U": 13,
        "SD1L3.U": 14,
        "SD1L3.C": 15,
        "ACTL3.U": 16,
        "SD1L4.U": 17,
        "ACTL4.U": 18,
        "GA1L4.U": 19,
        "GA2L4.U": 20,
        "O1.F": 21,
        "P1.C": 22,
        "P1.U": 23,
        "P3.U": 24,
        "P5.U": 25,
        "P6.C": 26,
        "BSMR1.S": 27,
        "ACTR1.C": 28,
        "SD1R1.S": 29,
        "GA1R1.S": 30,
        "GA2R1.S": 31,
        "SD2R1.S": 32,
        "GA1R2.D": 33,
        "SD2R2.D": 34,
        "ACTR2.D": 35,
        "GA2R2.D": 36,
        "BSMR2.D": 37,
        "SD1R2.D": 38,
        "SD1R3.T": 39,
        "BSMR3.T": 40,
        "ACTR3.T": 41,
        "GA1R3.T": 42,
        "BSMR3.A": 43,
        "SD2R3.T": 44,
        "GA2R3.T": 45,
        "ACTR5.U": 46,
        "GA2R5.U": 47,
        "BSMR5.U": 48,
        "GA1R5.U": 49,
        "SD2R6.T": 50,
        "GA2R6.T": 51,
        "SD1R6.T": 52,
        "ACTR6.T": 53,
        "GA1R6.T": 54,
        "ACTR7.W": 55,
        "GA2R7.W": 56,
        "GA1R7.W": 57,
        "GA1R8.T": 58,
        "ACTR8.T": 59,
        "S3.U": 60,
        "T1.F": 61,
        "T2.C": 62,
        "W1.S": 63,
        "W2.U": 64,
        "Y1.F": 65,
        "other": 66,
        "NG": -1,
        "...5A.U": -1,
        "...B1.F": 0,
        "...B2.F": 66,
        "...B3.F": 66,
        "...B4.F": 66,
        "...D1.S": 1,
        "...D1.U": 2,
        "...E3.W": 3,
        "...G1.U": 4,
        "....P3.U": 24,
        "....P6.C": 26,
        "....P1.U": 23,
        "....P5.U": 25,
        "....W1.S": 63,
        "...M1.U": 66,
        "...O1.F": 21,
        "...P1.U": 23,
        "...P1.C": 22,
        "...P3.U": 24,
        "...P5.U": 25,
        "...P5.I": 66,
        "...P6.C": 26,
        "...P6.E": 66,
        "...P6.F": 66,
        "...PA.C": 66,
        "...S2.S": 66,
        "...S3.U": 60,
        "...S4.P": 66,
        "...T1.F": 61,
        "...T2.C": 62,
        "...T3.F": 66,
        "...U1.U": 66,
        "...U1.F": 66,
        "...V1.R": 66,
        "...W1.S": 63,
        "...W1.F": 66,
        "...W2.U": 64,
        "...Y1.F": 65,
        "...Y1.U": 66,
        "...YI.F": 66,
        "...Z0.U": 66,
        "...Z5.U": 66,
        "B2.F": 66,
        "B3.F": 66,
        "B4.F": 66,
        "PLNH2.U": 66,
        "CNTH3.U": 66,
        "GAAL1.U": 66,
        "ANDL1.U": 66,
        "SD2L1.U": 66,
        "GA1L1.U": 66,
        "SD1L1.U": 66,
        "GA2L2.U": 66,
        "SD1L2.U": 66,
        "BSML2.U": 66,
        "ACTL2.U": 66,
        "SD2L2.U": 66,
        "GA2L3.C": 66,
        "SD2L3.U": 66,
        "GA1L3.C": 66,
        "BSML3.C": 66,
        "BSML3.U": 66,
        "SD2L3.C": 66,
        "ACTL3.C": 66,
        "BSML4.U": 66,
        "SD2L4.U": 66,
        "M1.U": 66,
        "NBSML1.U": 66,
        "P5.I": 66,
        "P6.E": 66,
        "P6.F": 66,
        "PA.C": 66,
        "GA1R2.U": 66,
        "HPDR3.T": 66,
        "SD1R5.U": 66,
        "SD2R5.U": 66,
        "SD2R7.W": 66,
        "BSMR7.W": 66,
        "SD1R7.W": 66,
        "SD1R8.T": 66,
        "GA2R8.T": 66,
        "S2.S": 66,
        "S4.P": 66,
        "T3.F": 66,
        "GA2TL4.U": 66,
        "U1.F": 66,
        "U1.U": 66,
        "V1.R": 66,
        "W1.F": 66,
        "Y1.U": 66,
        "YI.F": 66,
        "Z0.U": 66,
        "Z5.U": 66,
    }

label_text2id = {}
for k,v in _label_text2id.items():
    k = re.compile(k.lower())
    label_text2id[k] = v

def trans_labels(labels):
    #return labels
    classes = ['B1.F', 'D1.S', 'D1.U', 'E3.W', 'G1.U', 'SD1L1.R', 'ACTL1.U', 'GA1L1.R', 'GA2L1.R', 'SD2L1.R', 'BSML1.U', 'GA1L2.U', 'GA2L3.U', 'GA1L3.U', 'SD1L3.U', 'SD1L3.C', 'ACTL3.U', 'SD1L4.U', 'ACTL4.U', 'GA1L4.U', 'GA2L4.U', 'O1.F', 'P1.C', 'P1.U', 'P3.U', 'P5.U', 'P6.C', 'BSMR1.S', 'ACTR1.C', 'SD1R1.S', 'GA1R1.S', 'GA2R1.S', 'SD2R1.S', 'GA1R2.D', 'SD2R2.D', 'ACTR2.D', 'GA2R2.D', 'BSMR2.D', 'SD1R2.D', 'SD1R3.T', 'BSMR3.T', 'ACTR3.T', 'GA1R3.T', 'BSMR3.A', 'SD2R3.T', 'GA2R3.T', 'ACTR5.U', 'GA2R5.U', 'BSMR5.U', 'GA1R5.U', 'SD2R6.T', 'GA2R6.T', 'SD1R6.T', 'ACTR6.T', 'GA1R6.T', 'ACTR7.W', 'GA2R7.W', 'GA1R7.W', 'GA1R8.T', 'ACTR8.T', 'S3.U', 'T1.F', 'T2.C', 'W1.S', 'W2.U', 'Y1.F', 'other']
    global label_text2id  
    res = []
    labels = set(labels)
    for l in labels:
        l = l.lower()
        find = False
        for k,v in label_text2id.items():
            if k.fullmatch(l):
                find = True
                if v is None:
                    break
                if v>=0 and v<=len(classes):
                    res.append(classes[v])
                else:
                    res.append(str(v))
                break
        if not find:
            res.append(l)
    return res


if __name__ == "__main__":
    args = parse_args()
    if args.suffix == "auto":
        args.suffix = get_auto_dataset_suffix(args.src_dir)
    
    if args.suffix == "none":
        args.no_imgs = False
        args.allow_empty = True

    if not args.no_imgs:
        img_files = wmlu.get_files(args.src_dir,suffix=args.img_suffix)
        ann_files = [get_ann_file_path(x,args.suffix) for x in img_files]
    else:
        ann_files = wmlu.get_files(args.src_dir,suffix=args.suffix)
        img_files = [get_img_file_path(x,"jpg") for x in ann_files]
    basenames = [wmlu.base_name(x) for x in img_files]
    if len(basenames) == len(set(basenames)):
        add_nr = False
    else:
        add_nr = True
    all_files = list(zip(img_files,ann_files))
    wmlu.show_list(all_files[:100])
    if len(all_files)>100:
        print("...")
    if not args.allow_empty:
        all_files = list(filter(lambda x:osp.exists(x[1]),all_files))
    save_dir = wmlu.get_unused_path(args.out_dir)
    os.makedirs(save_dir)
    random.seed(int(time.time()))
    random.shuffle(all_files)
    print(f"Find {len(all_files)} files")

    for i,(img_f,ann_f) in enumerate(all_files):
        labels = get_labels(ann_f,args.suffix)
        labels = trans_labels(labels)
        for l in set(labels):
            t_save_dir = osp.join(save_dir,l)
            os.makedirs(t_save_dir,exist_ok=True)
            copy_files(img_f,ann_f,t_save_dir,add_nr,silent=args.silent,allow_empty=args.allow_empty,args=args)


