import img_utils as wmli
import object_detection2.visualization as odv
import matplotlib.pyplot as plt
from iotoolkit.labelmemckeypoints_dataset import LabelmeMCKeypointsDataset
import argparse
import os.path as osp
import wml_utils as wmlu
import wtorch.utils as wtu
from object_detection2.metrics.toolkit import *
from object_detection2.standard_names import *
from wstructures import WMCKeypoints
from object_detection2.keypoints import mckps_distance_matrix

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
    parser.add_argument('--sigma', type=float, default=1.1,help='sigma')
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
            return np.zeros([0]),0,0,0
        return np.zeros([0]),0,lh_labels.size,rh_kps.size
    elif lh_kps.size == 0:
        return np.zeros([0]),0,lh_labels.size,rh_kps.size

    lh_shape = lh_kps.shape
    #indict if there have some rh_kps match with this ground-truth rh_kps
    lh_mask = np.zeros([lh_shape[0]],dtype=np.int32)
    kps_shape = rh_kps.shape
    #indict if there have some ground-truth rh_kps match with this rh_kps
    rh_mask = np.zeros(kps_shape[0],dtype=np.int32)
    lh_size = lh_labels.shape[0]
    dis_m = mckps_distance_matrix(lh_kps,rh_kps)
    match_dis = []
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
            match_dis.append(cur_d)
            if cur_d<total_same_sigma:
                total_same_nr += 1

    match_nr = np.sum(lh_mask)
    lh_unmatch_nr = lh_mask.size-match_nr
    rh_unmatch_nr = rh_mask.size-match_nr
    match_dis = np.array(match_dis,dtype=np.float32)


    return match_dis,match_nr,total_same_nr,lh_unmatch_nr,rh_unmatch_nr


def cmp_datasets(lh_ds,rh_ds,sigma=1.1,**kwargs):
    '''
    :param lh_ds:
    :param rh_ds: as gt datasets
    :param num_classes:
    :param mask_on:
    :return:
    '''
    rh_ds_dict = {}
    rh_total_box_nr = 0
    lh_total_box_nr = 0
    same_sample_nr = 0
    total_same_nr = 0
    total_same_sample_nr = 0
    same_nr = 0
    diff_nr = 0
    all_dis = []
    sample_in_two_dataset = 0
    
    for data in rh_ds:
        full_path = data[IMG_INFO][FILEPATH]
        category_ids = data[GT_LABELS]
        rh_ds_dict[os.path.basename(full_path)] = data
        rh_total_box_nr += len(category_ids)
    
    for i,data in enumerate(lh_ds):
        full_path = data[IMG_INFO][FILEPATH]
        category_ids = data[GT_LABELS]
        lh_total_box_nr += len(category_ids)

        base_name = os.path.basename(full_path)
        if base_name not in rh_ds_dict:
            print(f"Error find {base_name} in rh_ds faild.")
            continue
        sample_in_two_dataset += 1
        rh_data = rh_ds_dict[base_name]
        rh_kps,rh_labels = WMCKeypoints.split2single_nppoint(rh_data[GT_KEYPOINTS],rh_data[GT_LABELS])
        lh_kps,lh_labels = WMCKeypoints.split2single_nppoint(data[GT_KEYPOINTS],data[GT_LABELS])
        match_dis,match_nr,tsame_nr,lh_unmatch_nr,rh_unmatch_nr = cmp_sample(lh_kps,lh_labels,rh_kps,rh_labels,sigma=sigma)
        same_nr += match_nr
        diff_nr += lh_unmatch_nr+rh_unmatch_nr
        total_same_nr += tsame_nr
        if match_dis.size>0:
            all_dis.append(match_dis)
        if lh_unmatch_nr==0 and rh_unmatch_nr==0:
            same_sample_nr += 1
            if match_nr == tsame_nr:
                total_same_sample_nr += 1


    all_dis = np.concatenate(all_dis,axis=0)
    print(f"Dataset1 len {len(lh_ds)}, dataset2 len {len(rh_ds)}") 
    print(f"Sample sample {same_sample_nr}/{len(lh_ds)+len(rh_ds)-sample_in_two_dataset}, total sample {total_same_sample_nr}")
    print(f"Match points {same_nr}, total match points {total_same_nr}, unmatch points {diff_nr}")
    print(f"Match dis min {np.min(all_dis):.2f}, max {np.max(all_dis):.2f}, mean {np.mean(all_dis):.2f}, std {np.std(all_dis):.2f}")

if __name__ == "__main__":

    args = parse_args()
    print(DATASETS,args.type)
    data0 = DATASETS[args.type](label_text2id=None,shuffle=False)
    data0.read_data(args.dir0,img_suffix=args.ext)

    data1 = DATASETS[args.type](label_text2id=None,shuffle=False)
    data1.read_data(args.dir1,img_suffix=args.ext)

    cmp_datasets(data0,data1,sigma=args.sigma)