import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
import matplotlib.pyplot as plt
from wml.iotoolkit.labelmemckeypoints_dataset import LabelmeMCKeypointsDataset
import argparse
import os.path as osp
import wml.wml_utils as wmlu
import wml.wtorch.utils as wtu
from wml.object_detection2.metrics.toolkit import *
from wml.object_detection2.standard_names import *
from wml.wstructures import WMCKeypoints
from wml.object_detection2.keypoints import mckps_distance_matrix
from functools import partial
import shutil
import os

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('dir0', type=str, help='source video directory')
    parser.add_argument('dir1', type=str, help='output rawframe directory')
    parser.add_argument(
        '--ext',
        type=str,
        default='.jpg;;.bmp;;.jpeg;;.png',
        help='video file extensions')
    parser.add_argument('--type', type=str, default='LabelmeMCKeypointsDataset',help='Data set type')
    parser.add_argument('--save-dir', type=str, default=None,help='Data set type')
    parser.add_argument('--ignore-labels', type=str, nargs="+",default=None,help='labels to ignore')
    parser.add_argument('--sigma', type=float, default=1.1,help='sigma')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    args = parser.parse_args()

    return args

def text_fn(x,scores):
    return x

DATASETS = {}

def register_dataset(type):
    DATASETS[type.__name__] = type

register_dataset(LabelmeMCKeypointsDataset)

def simple_names(x):
    if "--" in x:
        return x.split("--")[-1]
    return x

def cmp_sample(lh_kps,lh_labels,rh_kps,rh_labels,sigma=1.1,total_same_sigma=0.1):
    '''
    lh_kps: [N,2]
    lh_labels:[N]
    rh_kps: [M,2]
    rh_labels: [M]
    return:
    match_dis,match_nr,total_same_nr,lh_unmatch_nr,rh_unmatch_nr
    '''
    if not isinstance(lh_kps,np.ndarray):
        lh_kps = np.array(lh_kps)
    if not isinstance(lh_labels,np.ndarray):
        lh_labels = np.array(lh_labels)
    
    if rh_kps.size == 0:
        if lh_kps.size == 0:
            return np.zeros([0]),np.zeros([0,2]),0,0,0,0
        return np.zeros([0]),np.zeros([0,2]),0,0,lh_labels.size,rh_kps.size
    elif lh_kps.size == 0:
        return np.zeros([0]),np.zeros([0,2]),0,0,lh_labels.size,rh_kps.size

    lh_shape = lh_kps.shape
    #indict if there have some rh_kps match with this ground-truth rh_kps
    lh_mask = np.zeros([lh_shape[0]],dtype=np.int32)
    kps_shape = rh_kps.shape
    #indict if there have some ground-truth rh_kps match with this rh_kps
    rh_mask = np.zeros(kps_shape[0],dtype=np.int32)
    lh_size = lh_labels.shape[0]
    dis_m = mckps_distance_matrix(lh_kps,rh_kps)
    match_dis = []
    match_point_dis = []
    total_same_nr = 0
    for i in range(lh_size):
        cur_dis = dis_m[i]
        idxs = np.argsort(cur_dis)
        for idx in idxs:
            if rh_mask[idx] or lh_labels[i] != rh_labels[idx]:
                continue
            cur_d = cur_dis[idx]
            if cur_d > sigma:
                break
            lh_mask[i] = 1
            rh_mask[idx] = 1
            match_point_dis.append(lh_kps[i]-rh_kps[idx])
            match_dis.append(cur_d)
            if cur_d<total_same_sigma:
                total_same_nr += 1
    
    match_nr = np.sum(lh_mask)
    lh_unmatch_nr = lh_mask.size-match_nr
    rh_unmatch_nr = rh_mask.size-match_nr
    match_dis = np.array(match_dis,dtype=np.float32)
    match_point_dis = np.array(match_point_dis)


    return match_dis,match_point_dis,match_nr,total_same_nr,lh_unmatch_nr,rh_unmatch_nr


def save_data(data,save_dir,suffix,save_raw_img=False,args=None,root_dir=None):
    full_path = data[IMG_INFO][FILEPATH]

    kps = data[GT_KEYPOINTS]
    img = wmli.imread(full_path)
    if args.new_width > 1:
        img = wmli.resize_width(img,args.new_width)
        kps = kps.resize(img.shape[:2][::-1])
        if save_raw_img:
            if root_dir is None:
                save_path = wmlu.change_dirname(full_path,save_dir)
            else:
                r_path = wmlu.get_relative_path(full_path,root_dir)
                save_path = osp.join(save_dir,r_path)
            wmli.imwrite(save_path,img)
    elif args.new_height > 1:
        img = wmli.resize_height(img,args.new_height)
        kps = kps.resize(img.shape[:2][::-1])
        if save_raw_img:
            if root_dir is None:
                save_path = wmlu.change_dirname(full_path,save_dir)
            else:
                r_path = wmlu.get_relative_path(full_path,root_dir)
                save_path = osp.join(save_dir,r_path)
            wmli.imwrite(save_path,img)
    elif save_raw_img:
        if root_dir is None:
            save_path = wmlu.change_dirname(full_path,save_dir)
        else:
            r_path = wmlu.get_relative_path(full_path,root_dir)
            save_path = osp.join(save_dir,r_path)
        wmli.imwrite(save_path,img)

    img = odv.draw_maskv2(img,classes=data[GT_LABELS],masks=kps)
    if root_dir is None:
        save_path = wmlu.change_dirname(full_path,save_dir)
    else:
        r_path = wmlu.get_relative_path(full_path,root_dir)
        save_path = osp.join(save_dir,r_path)
    if suffix is not None or save_raw_img:
        if suffix is None:
            suffix = "_0"
        save_path = osp.splitext(save_path)
        save_path = save_path[0]+suffix+save_path[1]
    wmli.imwrite(save_path,img)



def save_datas(data_info,save_dir,args):
    wmlu.create_empty_dir_remove_if(save_dir)
    for key,datas in data_info.items():
        c_save_dir = osp.join(save_dir,key)
        os.makedirs(c_save_dir,exist_ok=True)
        for d in datas:
            if d[0] is not None and d[1] is not None:
                save_data(d[0],c_save_dir,"_a",save_raw_img=True,args=args,root_dir=args.dir0)
                save_data(d[1],c_save_dir,"_b",save_raw_img=False,args=args,root_dir=args.dir1)
            elif d[0] is not None:
                save_data(d[0],c_save_dir,None,save_raw_img=True,args=args,root_dir=args.dir0)
            elif d[1] is not None:
                save_data(d[1],c_save_dir,None,save_raw_img=True,args=args,root_dir=args.dir1)

        

def cmp_datasets(lh_ds,rh_ds,sigma=1.1,**kwargs):
    '''
    :param lh_ds:
    :param rh_ds: as gt datasets
    :param num_classes:
    :param mask_on:
    :return:
    '''
    print(f"Sigma={sigma}")
    rh_ds_dict = {}
    rh_total_box_nr = 0
    lh_total_box_nr = 0
    same_sample_nr = 0
    total_same_nr = 0
    total_same_sample_nr = 0
    same_nr = 0
    diff_nr = 0
    all_dis = []
    all_point_dis = []
    sample_in_two_dataset = 0
    diff_info = wmlu.MDict(dtype=list)
    
    for data in rh_ds:
        full_path = data[IMG_INFO][FILEPATH]
        category_ids = data[GT_LABELS]
        base_name = os.path.basename(full_path)
        rh_ds_dict[base_name] = data
        rh_total_box_nr += len(category_ids)

    matched_key = set() 
    for i,data in enumerate(lh_ds):
        full_path = data[IMG_INFO][FILEPATH]
        category_ids = data[GT_LABELS]
        lh_total_box_nr += len(category_ids)

        base_name = os.path.basename(full_path)
        if base_name not in rh_ds_dict:
            if len(data[GT_LABELS])>0:
                print(f"ERROR: find {base_name} in rh_ds faild.")
            else:
                print(f"WARNING: find {base_name} in rh_ds faild, lh dataset is empty.")
            diff_info['not_find_0'].append((data,None))
            continue
        matched_key.add(base_name)
        sample_in_two_dataset += 1
        rh_data = rh_ds_dict[base_name]
        rh_kps,rh_labels = WMCKeypoints.split2single_nppoint(rh_data[GT_KEYPOINTS],rh_data[GT_LABELS])
        lh_kps,lh_labels = WMCKeypoints.split2single_nppoint(data[GT_KEYPOINTS],data[GT_LABELS])
        match_dis,match_point_dis,match_nr,tsame_nr,lh_unmatch_nr,rh_unmatch_nr = cmp_sample(lh_kps,lh_labels,rh_kps,rh_labels,sigma=sigma)
        same_nr += match_nr
        diff_nr += lh_unmatch_nr+rh_unmatch_nr
        total_same_nr += tsame_nr
        if match_dis.size>0:
            all_dis.append(match_dis)
            all_point_dis.append(match_point_dis)
        if lh_unmatch_nr==0 and rh_unmatch_nr==0:
            same_sample_nr += 1
            if match_nr == tsame_nr:
                total_same_sample_nr += 1
                diff_info['total_same'].append((data,None))
            else:
                diff_info['same'].append((data,rh_data))
        else:
            diff_info['diff'].append((data,rh_data))

    for base_name in rh_ds_dict.keys():
        if base_name in matched_key:
            continue
        print(f"Error find {base_name} in lh_ds faild.")
        diff_info['not_find_1'].append((None,data))

    if args.save_dir is not None and len(args.save_dir)>0:
        save_datas(diff_info,args.save_dir,args)
        print(f"Save dir: {args.save_dir}")
    all_dis = np.concatenate(all_dis,axis=0)
    all_point_dis = np.concatenate(all_point_dis,axis=0)
    print(f"Dataset1 len {len(lh_ds)}, dataset2 len {len(rh_ds)}") 
    print(f"Sample sample {same_sample_nr}/{len(lh_ds)+len(rh_ds)-sample_in_two_dataset}, total same sample {total_same_sample_nr}")
    print(f"Match points {same_nr}, total match points {total_same_nr}, unmatch points {diff_nr}")
    print(f"Match dis min {np.min(all_dis):.2f}, max {np.max(all_dis):.2f}, mean {np.mean(all_dis):.2f}, std {np.std(all_dis):.2f}")
    print(f"Match point dis = {np.mean(all_point_dis,axis=0)}")

def ignore_label_text2id(l,ignore_labels):
    if l.upper() in ignore_labels:
        return None
    return l

if __name__ == "__main__":

    args = parse_args()
    ignore_labels = args.ignore_labels
    if ignore_labels is None or len(ignore_labels)==0:
        label_text2id = None
    else:
        ignore_labels = [x.upper() for x in ignore_labels]
        label_text2id = partial(ignore_label_text2id,ignore_labels=ignore_labels)
    print(DATASETS,args.type)
    data0 = DATASETS[args.type](label_text2id=label_text2id,shuffle=False)
    data0.read_data(args.dir0,img_suffix=args.ext)

    data1 = DATASETS[args.type](label_text2id=label_text2id,shuffle=False)
    data1.read_data(args.dir1,img_suffix=args.ext)

    cmp_datasets(data0,data1,sigma=args.sigma)