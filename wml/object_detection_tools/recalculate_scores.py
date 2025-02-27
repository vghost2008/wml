from argparse import ArgumentParser
import pickle
import numpy as np
from wml.wfilesystem import recurse_get_subdir_in_dir
import wml.wml_utils as wmlu
import glob
import os.path as osp
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('data', type=str,help='data dir')
    parser.add_argument('--classes', type=int,nargs="+",default=[],help='classes to calculate scores')
    parser.add_argument('--exclude', type=int,nargs="+",default=[],help='classes not to calculate scores')
    args = parser.parse_args()
    return args

def docalculate_scores(data,classes=[],exclude=[]):
    if len(classes)>0 and len(exclude)>0:
        print(f"set classes and exclude at the same time is not't allowd.")
        return
    if len(data)==0:
        return
    data = data.split("|")
    data = filter(lambda x:len(x)>0,data)
    data = [x.split("/") for x in data]
    n_data = []
    for x in data:
        x = [float(y) for y in x]
        n_data.append(x)
    n_data = np.array(n_data)
    print(f'input is {n_data}')
    if len(classes)>0:
        n_data = n_data[classes]
    elif len(exclude)>0:
        mask = np.ones([n_data.shape[0]],dtype=bool)
        mask[exclude] = False
        n_data = n_data[mask]

    print(f"processed data:")
    wmlu.show_list(n_data)
    print(np.mean(n_data,axis=0))


    

if __name__ == "__main__":
    args = parse_args()
    data = args.data
    classes = args.classes
    exclude = args.exclude
    docalculate_scores(data,classes,exclude)
