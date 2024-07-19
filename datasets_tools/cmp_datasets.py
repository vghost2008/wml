import img_utils as wmli
import object_detection2.visualization as odv
import matplotlib.pyplot as plt
from iotoolkit.pascal_voc_toolkit import PascalVOCData
from iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from iotoolkit.coco_toolkit import COCOData
from iotoolkit.labelme_toolkit import LabelMeData
import argparse
import os.path as osp
import wml_utils as wmlu
import wtorch.utils as wtu
from object_detection2.metrics.toolkit import *
from object_detection2.metrics.build import METRICS_REGISTRY
from itertools import count
import os

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('dir0', type=str, help='source video directory') #以dir0为gt
    parser.add_argument('dir1', type=str, help='output rawframe directory')
    parser.add_argument(
        '--ext',
        type=str,
        default='.jpg;;.bmp;;.jpeg;;.png',
        help='video file extensions')
    parser.add_argument('--type', type=str, default='PascalVOCData',help='Data set type')
    parser.add_argument('--metrics', type=str, default='COCOEvaluation',help='metrics type')
    parser.add_argument('--classes-wise', action='store_true', help='is classes wise')
    args = parser.parse_args()

    return args

def text_fn(x,scores):
    return x

DATASETS = {}

def register_dataset(type):
    DATASETS[type.__name__] = type

register_dataset(PascalVOCData)
register_dataset(COCOData)
register_dataset(MapillaryVistasData)
register_dataset(LabelMeData)

def simple_names(x):
    if "--" in x:
        return x.split("--")[-1]
    return x


def cmp_datasets(lh_ds,rh_ds,mask_on=False,model=COCOEvaluation,classes_begin_value=1,args=None,**kwargs):
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

    all_labels = set()
    
    for data in rh_ds:
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        rh_ds_dict[os.path.basename(full_path)] = data
        rh_total_box_nr += len(category_names)
        [all_labels.add(name.lower()) for name in category_names]

    for i,data in enumerate(lh_ds):
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        [all_labels.add(name.lower()) for name in category_names]
    
    all_labels = list(all_labels)
    name2idx = dict(zip(all_labels,count()))
    print("Name to label")
    wmlu.show_dict(name2idx)
    num_classes = len(all_labels)
    
    #eval2 = ClassesWiseModelPerformace(num_classes=num_classes,model_type=model)
    if args.classes_wise:
        eval = ClassesWiseModelPerformace(num_classes=num_classes,model_type=model,
                                        classes_begin_value=classes_begin_value)
    else:
        eval = model(num_classes=num_classes,mask_on=mask_on,**kwargs)

    for i,data in enumerate(lh_ds):
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        lh_total_box_nr += len(category_names)

        base_name = os.path.basename(full_path)
        if base_name not in rh_ds_dict:
            print(f"Error find {base_name} in rh_ds faild.")
            continue
        rh_data = rh_ds_dict[base_name]
            
        kwargs = {}
        kwargs['gtboxes'] = rh_data[4]
        kwargs['gtlabels'] = [name2idx[name.lower()] for name in rh_data[3]]
        kwargs['boxes'] = boxes
        kwargs['labels'] = [name2idx[name.lower()] for name in category_names]
        kwargs['probability'] = np.ones([len(category_names)],np.float32)
        kwargs['img_size'] = shape
        eval(**kwargs)

        if i % 100 == 0:
            eval.show()
    
    eval.show()
    print(f"bboxes nr {lh_total_box_nr} vs {rh_total_box_nr}")
    print("Name to label")
    wmlu.show_dict(name2idx)


if __name__ == "__main__":

    args = parse_args()
    print(DATASETS,args.type)
    data0 = DATASETS[args.type](label_text2id=None,shuffle=False,absolute_coord=True)
    data0.read_data(args.dir0,img_suffix=args.ext)

    data1 = DATASETS[args.type](label_text2id=None,shuffle=False,absolute_coord=True)
    data1.read_data(args.dir1,img_suffix=args.ext)

    model = METRICS_REGISTRY.get(args.metrics)

    cmp_datasets(data0,data1,mask_on=False,model=model,classes_begin_value=0,args=args)