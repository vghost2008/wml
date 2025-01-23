import sys
import os
import wml.object_detection2.npod_toolkit as npod
import wml.wml_utils
import matplotlib.pyplot as plt
import numpy as np
import math
import wml.object_detection2.visualization as odv
import wml.img_utils as wmli
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData,read_voc_xml
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData
from wml.iotoolkit.fast_labelme import FastLabelMeData
import wml.object_detection2.bboxes as odb 
import pandas as pd
import wml.wml_utils as wmlu
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from sklearn.cluster import KMeans
from functools import partial
from argparse import ArgumentParser
from itertools import count
from wml.iotoolkit.object365v2_toolkit import Object365V2
from wml.object_detection2.data_process_toolkit import remove_class
from collections import OrderedDict
from wml.walgorithm import lower_bound

class DictDatasetReader:

    def __init__(self,dataset) -> None:
        self.dataset = dataset
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        if hasattr(self.dataset,"get_ann_info"):
            data = self.dataset.get_ann_info(idx)
        else:
            data = self.dataset[idx]

        labels_text = [str(x) for x in data['labels']]
        return data['filename'],[0,0],data['labels'],labels_text,data['bboxes'],None,None,None,None
        
'''
ratio: h/w
'''
def statistics_boxes(boxes,nr=100,name=""):
    sizes = [math.sqrt((x[2]-x[0])*(x[3]-x[1])) for x in boxes]
    sizes1 = [math.fabs(x[2]-x[0]) for x in boxes] + [math.fabs(x[3]-x[1]) for x in boxes]
    ratios = [(x[2]-x[0])/(x[3]-x[1]+1e-8) for x in boxes]
    try:
        print(f"Min area size (sqrt(w*h)) {min(sizes):.2f}, max area size {max(sizes):.2f} (pixel), mean {np.mean(sizes):.2f}, std {np.std(sizes):.2f}.")
        print(f"Min ratio {min(ratios):.2f}, max ratios {max(ratios):.2f}, mean: {np.mean(ratios):.2f}.")
        print(f"Min side length (w0,h0,w1,h1,...) {min(sizes1):.2f}, max size length {max(sizes1):.2f} (pixel), mean {np.mean(sizes1):.2f}, std {np.std(sizes1):.2f}.")
        rratios = [x if x>=1 else 1.0/max(x,1e-3) for x in ratios]
        print(f"Real Ratio:Min {min(rratios):.2f}, max {max(rratios):.2f}, mean: {np.mean(rratios):.2f}, std: {np.std(rratios):.2f}.")
    except:
        pass
    '''plt.figure(2,figsize=(10,10))
    n, bins, patches = plt.hist(sizes, nr, normed=None, facecolor='blue', alpha=0.5)
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name+" area")
    plt.figure(3,figsize=(10,10))
    n, bins, patches = plt.hist(ratios, nr, normed=None, facecolor='red', alpha=0.5)
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name+" ratio")
    plt.show()
    print(max(ratios))
    return _statistics_value(sizes,nr),_statistics_value(ratios,nr)'''
    pd_sizes = pd.Series(sizes)
    pd_ratios = pd.Series(ratios)
    pd_side = pd.Series(sizes1)
    plt.figure(0,figsize=(15,10))
    #pd_sizes.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', normed = True, label = "hist")
    #pd_sizes.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', density=True, label = "hist")
    pd_sizes.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', label = "hist")
    #pd_sizes.plot(kind = 'kde', color = 'red', label ="kde")
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name)

    plt.figure(1,figsize=(15,10))
    #pd_ratios.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', normed = True, label = "hist")
    #pd_ratios.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', desnsity= True, label = "hist")
    pd_ratios.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black',  label = "hist")
    #pd_ratios.plot(kind = 'kde', color = 'red', label ="kde")
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name+" ratio")

    plt.figure(2,figsize=(15,10))
    #pd_ratios.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', normed = True, label = "hist")
    #pd_ratios.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', desnsity= True, label = "hist")
    pd_side.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black',  label = "hist")
    #pd_ratios.plot(kind = 'kde', color = 'red', label ="kde")
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title(name+" side length")
    plt.show()
    try:
        print(max(ratios))
    except:
        pass
    return _statistics_value(sizes,nr),_statistics_value(ratios,nr)

def statistics_classes_per_img(data,nr=100):
    pd_sizes = pd.Series(data)
    plt.figure(0,figsize=(15,10))
    #pd_sizes.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', normed = True, label = "hist")
    #pd_sizes.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', density=True, label = "hist")
    pd_sizes.plot(kind = 'hist', bins = nr, color = 'steelblue', edgecolor = 'black', label = "hist")
    #pd_sizes.plot(kind = 'kde', color = 'red', label ="kde")
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    plt.title("classes nr per img")
    plt.show()

def statistics_boxes_by_different_area(boxes,nr=100,bin_size=5,level=0,size_array=[]):
    sizes = [math.sqrt((x[2]-x[0])*(x[3]-x[1])) for x in boxes]
    min_size = min(sizes)
    max_size = max(sizes)
    delta = (max_size-min_size)/bin_size
    l_bboxes = {}
    if size_array is None or len(size_array)==0:
        size_array = []
        for i in range(bin_size-1):
            size_array.append(min_size+delta+i*delta)
    else:
        bin_size = len(size_array)+1

    for i,s in enumerate(sizes):
        if s>size_array[-1]:
            index = len(size_array)
        else:
            index = lower_bound(size_array,s)
        if index in l_bboxes:
            l_bboxes[index].append(boxes[i])
        else:
            l_bboxes[index] = [boxes[i]]

    print(f"bboxes nr of each range")
    for k in range(bin_size):
        if k not in l_bboxes:
            continue
        v = l_bboxes[k]
        print(k,len(v),f"{len(v)*100.0/len(boxes):.2f}%")

    size_array = [min_size]+size_array+[max_size]
    for i in range(bin_size):
        if i not in l_bboxes:
            continue
        l_size = min_size + i* delta
        h_size = l_size + delta
        statistics_boxes(l_bboxes[i],nr,name=f"area_{size_array[i]:.3f}->{size_array[i+1]:.3f}: {len(l_bboxes[i])*100/len(sizes):.1f}%")

    if False and level<1:
        branch_thr = 0.8
        for k,v in l_bboxes.items():
            if len(v)/len(boxes)>branch_thr:
                print(f"Show branch {k} statistics")
                statistics_boxes_by_different_area(v,nr=nr,bin_size=bin_size,level=level+1)

    

def statistics_boxes_by_different_ratio(boxes,nr=100,bin_size=5):
    ratios = [(x[2]-x[0])/(x[3]-x[1]+1e-8) for x in boxes]
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    delta = (max_ratio-min_ratio)/bin_size
    l_bboxes = {}
    for i, r in enumerate(ratios):
        index = int((r - min_ratio) / delta)
        if index in l_bboxes:
            l_bboxes[index].append(boxes[i])
        else:
            l_bboxes[index] = [boxes[i]]

    for k, v in l_bboxes.items():
        print(k, len(v))
    for i in range(bin_size):
        if i not in l_bboxes:
            continue
        l_ratio= min_ratio+ i * delta
        h_ratio= l_ratio+ delta
        statistics_boxes(l_bboxes[i], nr, name=f"ratio_{l_ratio:.3f}->{h_ratio:.3f}")
    pass

def _statistics_value(values,nr=100):
    value_max = max(values)
    value_min = min(values)
    values_y = [0.]*nr

    for v in values:
        if v<=value_min:
            values_y[0] += 1.0
        elif v>=value_max:
            values_y[nr-1] += 1.0
        else:
            index = int((v-value_min)*(nr-1)/(value_max-value_min))
            values_y[index] += 1.0

    return values_y,value_min,value_max

def default_encode_label(l):
    return l

def statistics_boxes_in_dir(dir_path,label_encoder=default_encode_label,labels_to_remove=None,nr=100,aspect_range=None):
    def get_datas():
        if not os.path.exists(dir_path):
            print("path {} not exists.".format(dir_path))
        files = wml_utils.recurse_get_filepath_in_dir(dir_path,suffix=".xml")
        print("\ntotal file size {}.".format(len(files)))
        for file in files:
            shape, bboxes, labels_text, difficult, truncated = read_voc_xml(file,aspect_range=aspect_range)
            yield bboxes,labels_text,os.path.basename(file)

    return statistics_boxes_with_datas(get_datas(),label_encoder,labels_to_remove,nr)

def trans_img_short_size_to(img_size,short_size=640):
    img_short_size = min(img_size[0],img_size[1])
    scale = short_size/img_short_size
    return [x*scale for x in img_size]

def trans_img_long_size_to(img_size,long_size=512):
    img_long_size = max(img_size[0],img_size[1])
    scale = long_size/img_long_size
    return [x*scale for x in img_size]

def statistics_boxes_with_datas(datas,label_encoder=default_encode_label,labels_to_remove=None,max_aspect=None,absolute_size=False,
                                trans_img_size=None,silent=False):
    all_boxes = []
    all_labels = []
    max_examples = 0
    label_file_count={}
    labels_to_file={}
    example_nrs = []
    classeswise_boxes = {}
    total_file_nr = 0
    classes_nr_per_img = []
    no_annotation_nr = 0
    total_crowd_files = 0
    total_crowd_bboxes = 0

    for data in datas:
        file, img_size,category_ids, labels_text, bboxes, binary_mask, area, is_crowd, _ = data
        total_file_nr += 1
        if is_crowd is not None:
            is_crowd = np.array(is_crowd).astype(np.int32)
    
            if np.any(is_crowd):
                total_crowd_files += 1
                total_crowd_bboxes += np.sum(is_crowd)
        if bboxes.shape[0]<1:
            if not silent:
                print(f"{file} no annotation, skip")
            no_annotation_nr += 1
            continue
        if absolute_size:
            if trans_img_size is not None:
                img_size = trans_img_size(img_size)
            #bboxes = odb.relative_boxes_to_absolutely_boxes(bboxes,width=img_size[1],height=img_size[0])
            pass
        classes_nr_per_img.append(len(set(labels_text)))
        file = os.path.basename(file)
        if len(labels_text)==0:
            continue
        aspect = npod.box_aspect(bboxes)
        if max_aspect is not None and np.max(aspect)>max_aspect:
            print(f"asepct is too large, expect max aspect is {max_aspect}, actual get {np.max(aspect)}")
        e_nr = len(labels_text)
        example_nrs.append(e_nr)
        max_examples = max(e_nr,max_examples)
        all_boxes.extend(bboxes)
        all_labels.extend(labels_text)
        for l,box in zip(labels_text,bboxes):
            if l in classeswise_boxes:
                classeswise_boxes[l].append(box)
            else:
                classeswise_boxes[l] = [box]

        tmp_dict = {}
        for l in labels_text:
            tmp_dict[l] = 1
            if l in labels_to_file:
                labels_to_file[l].append(file)
            else:
                labels_to_file[l] = [file]

        for k in tmp_dict.keys():
            if k in label_file_count:
                label_file_count[k] += 1
            else:
                label_file_count[k] = 1

    labels_counter = {}
    org_labels_counter = {}
    encoded_labels = []
    for _l in all_labels:
        l = label_encoder(_l)
        encoded_labels.append(l)
        if l in labels_counter:
            labels_counter[l] = labels_counter[l]+1
        else:
            labels_counter[l] = 1
        if _l in org_labels_counter:
            org_labels_counter[_l] = org_labels_counter[_l]+1
        else:
            org_labels_counter[_l] = 1
    example_nrs = np.array(example_nrs)
    try:
        print(f"Max element size {np.max(example_nrs)}, element min {np.min(example_nrs)}, element mean {np.mean(example_nrs)}, element var {np.var(example_nrs)}.")
    except:
        pass
    labels_counter = list(labels_counter.items())
    labels_counter.sort(key=lambda x:x[1],reverse=True)

    classes = [x[0] for x in labels_counter]
    print(f"num classes={len(classes)}")
    print(tuple(classes))

    total_nr = 0
    for k,v in labels_counter:
        total_nr += v

    print(f"Total files contain crowd bboxes: {total_crowd_files}/{total_crowd_files*100/total_file_nr:.2f}%")
    print(f"Total crowd bboxes: {total_crowd_bboxes}/{total_crowd_bboxes*100/max(total_nr,1):.2f}%")

    print(f"Total bboxes count {total_nr}")
    print("\n--->BBoxes count:")
    for k,v in labels_counter:
        print("{:>8}:{:<8}, {:>4.2f}%".format(k,v,v*100./total_nr))

    print(f"Total file count {total_file_nr}.")
    print(f"Total no annotation file count {no_annotation_nr}, {no_annotation_nr*100/total_file_nr:.2f}%.")
    print("\n--->File count:")
    label_file_count= list(label_file_count.items())
    label_file_count.sort(key=lambda x:x[1],reverse=True)
    for k,v in label_file_count:
        print("{:>8}:{:<8}, {:>4.2f}".format(k,v,v*100./total_file_nr))
    print("\n--->org statistics:")
    org_labels_counter= list(org_labels_counter.items())
    org_labels_counter.sort(key=lambda x:x[1],reverse=True)
    total_nr = 0
    for k,v in org_labels_counter:
        total_nr += v
    for k,v in org_labels_counter:
        print(f"{k:>8}:{v:<8}, {v*100./total_nr:>4.2f}%")
    if labels_to_remove is not None:
        all_boxes,encoded_labels = remove_class(all_boxes,encoded_labels,labels_to_remove)

    #show classes per img info
    classes_nr_per_img = np.array(classes_nr_per_img)
    try:
        print(f"Classes per img, min={np.min(classes_nr_per_img)}, max={np.max(classes_nr_per_img)}, mean={np.mean(classes_nr_per_img)}, std={np.std(classes_nr_per_img)}")
    except:
        pass

    return [all_boxes,classeswise_boxes,labels_to_file,classes_nr_per_img]

def statistics_dict_dataset_boxes_with_datas(datas,*args,**kwargs):
    ndatas = DictDatasetReader(datas)
    return statistics_boxes_with_datas(ndatas,*args,**kwargs)

def show_boxes_statistics(statics):
    plt.figure(0,figsize=(10,10))
    sizes = statics[0]
    nr = len(sizes[0])
    sizes_x = sizes[1]+np.array(range(nr)).astype(np.float32)*(sizes[2]-sizes[1])/(nr-1)
    sizes_x = sizes_x.tolist()
    plt.title("Size")
    plt.xticks(ticks(sizes[1],sizes[2],-3,20))
    plt.plot(sizes_x,sizes[0])
    plt.figure(1,figsize=(10,10))
    ratios = statics[1]
    nr = len(ratios[0])
    ratios_x = ratios[1]+np.array(range(nr)).astype(np.float32)*(ratios[2]-ratios[1])/(nr-1)
    ratios_x = ratios_x.tolist()
    plt.title("Ratio")
    plt.xticks(ticks(ratios[1],ratios[2],-1,20))
    plt.plot(ratios_x,ratios[0])
    plt.show()

def show_classwise_boxes_statistics(data,nr=20,labels=None):
    for k,v in data.items():
        if labels is not None and not k in labels:
            continue
        statistics_boxes(v,nr=nr,name=k)

def ticks(minv,maxv,order,nr):
    delta = (maxv-minv)/(2.*nr)
    scale = math.pow(10,order)
    n_min = (minv-delta)//scale
    n_max = (maxv+delta)//scale
    minv = n_min*scale
    maxv = n_max*scale
    t_delta = (max(scale,((maxv-minv)/nr))//scale)*scale
    return np.arange(minv,maxv,t_delta).tolist()

def show_anchor_box(img_file,boxes,size=None):
    nr = boxes.shape[0]
    classes = []
    scores = []
    for i in range(nr):
        classes.append(0)
        scores.append(1)
    img = wmli.imread(img_file)
    if size is not None:
        img = wmli.resize_img(img,(size[1],size[0]))

    odv.bboxes_draw_on_img(img, classes, scores, boxes)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()
    return img

def test_dataset():
    data = PascalVOCData(label_text2id=None)
    data.read_data("/home/vghost/ai/mldata2/test_data_0day/test_s")

    return data.get_items()

def pascal_voc_dataset(data_dir,labels=None):
    #labels = ['MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other']
    if labels is not None and len(labels)>0:
        label_text2id = dict(zip(labels,count()))
    else:
        label_text2id = None
    
    #data = PascalVOCData(label_text2id=label_text2id,resample_parameters={6:8,5:2,7:2})
    data = PascalVOCData(label_text2id=label_text2id,absolute_coord=True,silent=True)

    '''data_path = "/mnt/data1/wj/ai/smldata/boedcvehicle/train"
    data_path = "/mnt/data1/wj/ai/smldata/boedcvehicle/wt_06"
    data_path = "/home/wj/ai/mldata1/GDS1Crack/train"
    data_path = "/home/wj/ai/mldata1/take_photo/train/coco"
    data_path = "/mnt/data1/wj/ai/mldata/MOT/MOT17/train/MOT17-09-SDP/img1"
    data_path = "/home/wj/ai/mldata1/B11ACT/datas/labeled"
    data_path = "/home/wj/ai/mldata1/B7mura/datas/data/ML3U"
    data_path = "/home/wj/ai/mldata1/B7mura/datas/data/MV1U"
    data_path = "/home/wj/ai/mldata1/B7mura/datas/data/MU4U"
    data_path = "/home/wj/ai/mldata1/B7mura/datas/data"
    data_path = "/home/wj/下载/_数据集"'''
    #data_path = "/home/wj/ai/mldata1/B7mura/datas/test_s0"
    #data_path = "/home/wj/0day/wt_06"
    #data_path = '/home/wj/0day/pyz'
    xmls = wmlu.get_files(data_dir,suffix=".xml")
    imgs = [wmlu.change_suffix(x,"jpg") for x in xmls]
    files = list(zip(imgs,xmls))
    data.read_data(files)
    '''data.read_data(data_dir,
                   silent=True,
                   img_suffix=".bmp;;.jpg")'''

    return data.get_items()

def coco2014_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_train2014.json"), 
                   image_dir=wmlu.home_dir("ai/mldata/coco/train2014"))

    return data.get_items()

def coco2017_dataset(annotations_path,labels=None):
    data = COCOData(remove_crowd=False)
    data.read_data(annotations_path)

    return data.get_items()

def objects365_dataset(annotations_path,labels=None):
    data = Object365V2(remove_crowd=False)
    data.read_data(annotations_path)

    return data.get_items()

def coco2014_val_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_val2014.json"),
                   image_dir=wmlu.home_dir("ai/mldata/coco/val2014"))

    return data.get_items()

def labelme_dataset(data_dir,labels):
    data = FastLabelMeData(label_text2id=None,absolute_coord=True)
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10")
    data.read_data(data_dir,img_suffix="bmp;;jpg;;jpeg")
    #data.read_data("/home/wj/ai/mldata1/B11ACT/datas/test_s0",img_suffix="bmp")
    return data.get_items()


lid = 0
def _mapillary_vistas_dataset():
    NAME2ID = {}
    ID2NAME = {}

    def name_to_id(x):
        global lid
        if x in NAME2ID:
            return NAME2ID[x]
        else:
            NAME2ID[x] = lid
            ID2NAME[lid] = x
            lid += 1
            return NAME2ID[x]
    ignored_labels = [
        'manhole', 'dashed', 'other-marking', 'static', 'front', 'back',
        'solid', 'catch-basin','utility-pole', 'pole', 'street-light','direction-back', 'direction-front'
         'ambiguous', 'other','text','diagonal','left','right','water-valve','general-single','temporary-front',
        'wheeled-slow','parking-meter','split-left-or-straight','split-right-or-straight','zigzag',
        'give-way-row','ground-animal','phone-booth','give-way-single','garage','temporary-back','caravan','other-barrier'
    ]
    data = MapillaryVistasData(label_text2id=name_to_id, shuffle=False, ignored_labels=ignored_labels)
    # data.read_data("/data/mldata/qualitycontrol/rdatasv5_splited/rdatasv5")
    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatav10_preproc")
    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10_neg_preproc")
    data.read_data(wmlu.home_dir("ai/mldata/mapillary_vistas/mapillary-vistas-dataset_public_v2.0"))
    return data.get_boxes_items()

def mapillary_vistas_dataset(data_dir):
    data = MapillaryVistasData(shuffle=False,use_semantic=False)
    # data.read_data("/data/mldata/qualitycontrol/rdatasv5_splited/rdatasv5")
    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatav10_preproc")
    # data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10_neg_preproc")
    data.read_data(data_dir)
    return data.get_boxes_items()
