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
    parser.add_argument('--type', type=str, default='PascalVOCData',help='Data set type')
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


def cmp_datasets(lh_ds,rh_ds,num_classes=90,mask_on=False,model=COCOEvaluation,classes_begin_value=1,**kwargs):
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
    
    for data in rh_ds:
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        rh_ds_dict[os.path.basename(full_path)] = data
        rh_total_box_nr += len(category_ids)
    
    eval = model(num_classes=num_classes,mask_on=mask_on,**kwargs)
    #eval2 = ClassesWiseModelPerformace(num_classes=num_classes,model_type=model)
    eval2 = ClassesWiseModelPerformace(num_classes=num_classes,model_type=COCOEvaluation,
                                       classes_begin_value=classes_begin_value)
    for i,data in enumerate(lh_ds):
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        lh_total_box_nr += len(category_ids)

        base_name = os.path.basename(full_path)
        if base_name not in rh_ds_dict:
            print(f"Error find {base_name} in rh_ds faild.")
            continue
        rh_data = rh_ds_dict[base_name]
            
        kwargs = {}
        kwargs['gtboxes'] = rh_data[4]
        kwargs['gtlabels'] = rh_data[2]
        kwargs['boxes'] = boxes
        kwargs['labels'] = category_ids
        kwargs['probability'] = np.ones_like(category_ids,np.float32)
        kwargs['img_size'] = shape
        eval(**kwargs)
        eval2(**kwargs)

        if i % 100 == 0:
            eval.show()
    
    eval.show()
    eval2.show()
    print(f"bboxes nr {lh_total_box_nr} vs {rh_total_box_nr}")


if __name__ == "__main__":

    args = parse_args()
    print(DATASETS,args.type)
    data0 = DATASETS[args.type](label_text2id=None,shuffle=False,absolute_coord=True)
    data0.read_data(args.dir0,img_suffix=args.ext)

    data1 = DATASETS[args.type](label_text2id=None,shuffle=False,absolute_coord=True)
    data1.read_data(args.dir1,img_suffix=args.ext)

    cmp_datasets(data0.get_items(),data1.get_items(),num_classes=5,mask_on=False,model=COCOEvaluation,classes_begin_value=0)