import wml.wml_utils as wmlu
from wml.object_detection2.standard_names import *
import numpy as np

def statistics_mckeypoints(dataset):
    counter = wmlu.Counter()
    samples_nr = 0
    points_per_sample = []
    max_classes_num_in_one_img = 0
    for data in dataset:
        points = data[GT_KEYPOINTS]
        labels = data[GT_LABELS]
        cur_nr = 0
        for l,p in zip(labels,points):
            counter.add(l,p.numel())
            cur_nr += p.numel()
        max_classes_num_in_one_img = max(max_classes_num_in_one_img,len(set(labels)))
        samples_nr += 1
        points_per_sample.append(cur_nr)
    
    points_per_sample = np.array(points_per_sample)
    data = list(counter.items())
    data = sorted(data,key=lambda x:x[1],reverse=True)
    print(f"Total samples nr {len(dataset)}")
    print(f"Mean samples per sample: {np.mean(points_per_sample)}, max: {np.max(points_per_sample)}, min: {np.min(points_per_sample)}")
    print(f"Max num classes per sample: {max_classes_num_in_one_img}")
    print(f"Points per classes:")
    wmlu.show_list(data)
