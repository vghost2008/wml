import wml.img_utils as wmli
import wml.object_detection2.visualization as odv
import matplotlib.pyplot as plt
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData
import argparse
import os.path as osp
import wml.wml_utils as wmlu
import wml.wtorch.utils as wtu
from wml.object_detection2.metrics.toolkit import *
from wml.object_detection2.metrics.build import METRICS_REGISTRY
from itertools import count
from wml.iotoolkit import get_auto_dataset_type
import os

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('dir0', type=str, help='source video directory') #以dir0为gt
    parser.add_argument('dir1', type=str, help='output rawframe directory')
    parser.add_argument(
        '--ext',
        type=str,
        default=wmli.BASE_IMG_SUFFIX,
        help='video file extensions')
    parser.add_argument('--type', type=str, default='auto',help='Data set type')
    parser.add_argument('--metrics', type=str, default='COCOEvaluation',help='metrics type')
    #parser.add_argument('--metrics', type=str, default='PrecisionAndRecall',help='metrics type')
    #parser.add_argument('--metrics', type=str, default='ClsPrecisionAndRecall',help='metrics type')
    parser.add_argument('--save-dir', type=str, help='save dir for different annotation.')
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

def save_data(lh_data,rh_data,save_dir):
    save_one_data(lh_data,save_dir,"_0")
    save_one_data(rh_data,save_dir,"_1")

def save_one_data(data,save_dir,suffix):
    if save_dir is not None:
        return
    full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
    img = wmli.imread(full_path)
    img = odv.draw_bboxes_xy(img,category_names,None,boxes,show_text=True,is_relative_coordinate=False)
    if binary_masks is not None:
        img = odv.draw_maskv2_xy(img,category_names,boxes,binary_masks)
    
    os.makedirs(save_dir,exist_ok=True)

    save_name = wmlu.base_name(full_path)+suffix
    suffix = osp.splitext(full_path)[-1]
    save_path = osp.join(save_dir,save_name+suffix)
    wmli.imwrite_for_view(save_path,img)


def cmp_datasets(lh_ds,rh_ds,mask_on=False,model=COCOEvaluation,classes_begin_value=1,name2label=None,args=None,**kwargs):
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
    if name2label is None:
        name2label = dict(zip(all_labels,count()))
    else:
        tn2l = {}
        for k,v in name2label.items():
            tn2l[k.lower()] = v
        name2label = tn2l
    print("Name to label")
    wmlu.show_dict(name2label)
    num_classes = len(all_labels)
    
    #eval2 = ClassesWiseModelPerformace(num_classes=num_classes,model_type=model)
    if args.classes_wise:
        eval = ClassesWiseModelPerformace(num_classes=num_classes,model_type=model,
                                        classes_begin_value=classes_begin_value)
    else:
        eval = model(num_classes=num_classes,mask_on=mask_on,**kwargs)
    
    total_same_nr = 0
    cmp_nr = 0

    for i,data in enumerate(lh_ds):
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        lh_total_box_nr += len(category_names)

        base_name = os.path.basename(full_path)
        if base_name not in rh_ds_dict:
            print(f"Error find {base_name} in rh_ds faild.")
            continue

        cmp_nr += 1
        rh_data = rh_ds_dict[base_name]
            
        kwargs = {}
        kwargs['gtboxes'] = rh_data[4]
        kwargs['gtlabels'] = [name2label[name.lower()] for name in rh_data[3]]
        kwargs['boxes'] = boxes
        kwargs['labels'] = [name2label[name.lower()] for name in category_names]
        p,r = getPrecision(**kwargs,threshold=0.8)
        print(p,r)
        if p>=0.99 and r>=0.99:
            total_same_nr += 1
        if args.save_dir is not None:
            if p<0.990 or r<0.99:
                save_data(data,rh_data,args.save_dir)
            else:
                save_one_data(data,osp.join(args.save_dir,"same"),suffix="")
        kwargs['probability'] = np.ones([len(category_names)],np.float32)
        kwargs['img_size'] = shape
        eval(**kwargs)

        if i % 5000 == 99:
            eval.show()
    
    eval.show()
    print(f"bboxes nr {lh_total_box_nr} vs {rh_total_box_nr}")
    if args.save_dir is not None:
        print(f"Save dir {args.save_dir}")
    print(f"Total same nr {total_same_nr}, cmp nr {cmp_nr}, total nr {len(lh_ds)+len(rh_ds)-cmp_nr}.")
    print("Name to label")
    wmlu.show_dict(name2label)


if __name__ == "__main__":

    args = parse_args()
    if args.type == "auto":
        dataset_type = get_auto_dataset_type(args.dir0)
    else:
        print(DATASETS,args.type)
        dataset_type = DATASETS[args.type]
    label_text2id = None
'''
    label_text2id = {
"Line" : 0 ,
"Point" : 1 ,
"Gap" : 2 ,
"Mura" : 3 ,
"Stain" : 4 ,
"Other" : 5 ,
"Abnormal" : 6 ,
"LX01" : 0 ,
"LX02" : 0 ,
"LX08" : 0 ,
"LX09" : 0 ,
"LX10" : 0 ,
"LX04" : 0 ,
"LX05" : 0 ,
"LY01" : 0 ,
"LY02" : 0 ,
"LY03" : 0 ,
"LY04" : 0 ,
"MB01" : 1 ,
"MB02" : 1 ,
"MB05" : 1 ,
"CD01" : 1 ,
"CP01" : 1 ,
"CF02" : 1 ,
"CF03" : 1 ,
"MG01" : 2 ,
"MG02" : 2 ,
"MG03" : 2 ,
"ML14" : 2 ,
"MO08" : 2 ,
"MO09" : 2 ,
"MG07" : 2 ,
"MG08" : 2 ,
"MG10" : 2 ,
"MG06" : 2 ,
"PS01" : 2 ,
"PS02" : 2 ,
"CF01" : 2 ,
"MO45" : 2 ,
"ML01" : 3 ,
"ML02" : 3 ,
"ML05" : 3 ,
"ML06" : 3 ,
"ML08" : 3 ,
"ML09" : 3 ,
"ML12" : 3 ,
"ML13" : 3 ,
"ML04" : 4 ,
"ML11" : 4 ,
"CZ02" : 5 ,
"CR01" : 5 ,
"FF01" : 6 ,
"FF03" : 6 ,
"FF08" : 6 ,
"FF11" : 6 ,
"MS13" : 6 ,
"AP11" : -1 ,
"PO01" : -1 ,
"CZ03" : -1 ,
"OKOK" : -1,
"AP22" : -1 ,
"AP20" : -1 ,
"PO03" : -1 ,
"PO02" : -1 ,
"MO07" : -1 ,
"PO09" : -1 ,
"PO04" : -1 ,
"PO06" : -1 ,
"AP16" : -1 ,
"MO10" : -1 ,
"CP02" : -1 ,
"AP01" : -1 ,
"ms06" : -1 ,
"po05" : -1 ,
"cr02" : -1 ,
}
'''

    data0 = dataset_type(label_text2id=label_text2id,shuffle=False,absolute_coord=True)
    data0.read_data(args.dir0,img_suffix=args.ext)

    data1 = dataset_type(label_text2id=label_text2id,shuffle=False,absolute_coord=True)
    data1.read_data(args.dir1,img_suffix=args.ext)

    model = METRICS_REGISTRY.get(args.metrics)

    if args.save_dir is not None:
        wmlu.create_empty_dir_remove_if(args.save_dir)

    cmp_datasets(data0,data1,mask_on=False,model=model,classes_begin_value=0,args=args,name2label=label_text2id)