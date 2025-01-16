#coding=utf-8
import numpy as np
import wml.img_utils as wmli
from wml.iotoolkit.pascal_voc_toolkit import PascalVOCData,write_voc_xml
from wml.iotoolkit.coco_toolkit import COCOData
from wml.iotoolkit.labelme_toolkit import LabelMeData,save_labelme_datav3
import wml.object_detection2.bboxes as odb 
import wml.wml_utils as wmlu
from wml.iotoolkit.mapillary_vistas_toolkit import MapillaryVistasData
import wml.object_detection2.data_process_toolkit as odp
from argparse import ArgumentParser
from itertools import count
import os.path as osp
import wml.img_utils as wmli
import wml.semantic.mask_utils as mu 

'''
对现有数据集进行裁切生成新的数据集，目前有两种裁剪方式
0, 在现有的目标附近裁剪
1，使用滑动窗裁剪
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str, help='source video directory')
    parser.add_argument('save_dir', type=str, help='save dir')
    parser.add_argument('--type', type=str, default="xml",help='dataset type')
    parser.add_argument('--labels', nargs="+",type=str,default=[],help='Config file')
    parser.add_argument('--min-size', type=int, default=0,help='min bbox size')
    parser.add_argument('--add-classes-name', action='store_true', help='min bbox size')
    parser.add_argument('--keep-empty', action='store_true', help='save imgs without objects.')
    parser.add_argument('--cut-type', type=int, default=0,help='0: arount object, 1: normal cut')
    parser.add_argument('--size', nargs="+",type=int,default=[1024,1024],help='save size, [w,h] or s - > [s,s]')
    parser.add_argument('--step', type=float, default=-1,help='step size')
    parser.add_argument('--keep-ratio', type=float, default=0.01,help='area ratio to keep after cut')
    args = parser.parse_args()
    return args


def pascal_voc_dataset(data_dir,labels=None):
    #labels = ['MS7U', 'MP1U', 'MU2U', 'ML9U', 'MV1U', 'ML3U', 'MS1U', 'Other']
    if labels is not None and len(labels)>0:
        label_text2id = dict(zip(labels,count()))
    else:
        label_text2id = None
    
    #data = PascalVOCData(label_text2id=label_text2id,resample_parameters={6:8,5:2,7:2})
    data = PascalVOCData(label_text2id=label_text2id,absolute_coord=True)

    data.read_data(data_dir,
                   silent=True,
                   img_suffix=".bmp;;.jpg")

    return data

def coco2014_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_train2014.json"), 
                   image_dir=wmlu.home_dir("ai/mldata/coco/train2014"))

    return data.get_items()

def coco2017_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata2/coco/annotations/instances_train2017.json"),
                   image_dir=wmlu.home_dir("ai/mldata2/coco/train2017"))

    return data.get_items()

def coco2014_val_dataset():
    data = COCOData()
    data.read_data(wmlu.home_dir("ai/mldata/coco/annotations/instances_val2014.json"),
                   image_dir=wmlu.home_dir("ai/mldata/coco/val2014"))

    return data.get_items()

def labelme_dataset(data_dir,labels):
    data = LabelMeData(label_text2id=None,absolute_coord=True)
    #data.read_data("/home/vghost/ai/mldata2/qualitycontrol/rdatasv10")
    data.read_data(data_dir,img_suffix="bmp;;jpg;;jpeg")
    #data.read_data("/home/wj/ai/mldata1/B11ACT/datas/test_s0",img_suffix="bmp")
    return data


lid = 0
def mapillary_vistas_dataset():
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

def cut_arount_bboxes_and_save(dataset,data_dir,save_dir,cut_size,keep_ratio=1e-6):
    '''
    在已有的目标附近裁图
    cut_size: (h,w)
    '''
    for idx,data in enumerate(dataset):
        img_file, shape,labels, labels_names, bboxes,binary_masks,*_ = data

        r_path = wmlu.get_relative_path(img_file,data_dir)
        bs_name,suffix = osp.splitext(r_path)

        if len(labels_names)==0:
            continue
        print(f"Process {idx}/{len(dataset)}")

        cut_bboxes = odb.set_bboxes_size(bboxes,size=cut_size).astype(np.int32)
        img = wmli.imread(img_file)
        labels_names = np.array(labels_names)

        for i in range(len(labels_names)):
            new_img,new_labels,new_bboxes,new_masks = odp.cut_annotation(cut_bboxes[i],img,labels_names,bboxes,binary_masks,keep_ratio=keep_ratio)
            save_base_name = bs_name+f"_{i}"
            img_save_path = osp.join(save_dir,save_base_name+suffix)
            wmli.imwrite(img_save_path,new_img)

            if binary_masks is not None:
                new_masks,*_ = mu.cut_masks(new_masks,new_bboxes)
                new_bboxes = odb.npchangexyorder(new_bboxes)
                ann_save_path = osp.join(save_dir,save_base_name+".json")
                save_labelme_datav3(ann_save_path,img_save_path,new_labels,new_bboxes,new_masks)
            else:
                ann_save_path = osp.join(save_dir,save_base_name+".xml")
                write_voc_xml(ann_save_path,img_save_path,new_img.shape,new_bboxes,new_labels,is_relative_coordinate=False)
                
    
def cut_bboxes_and_save(dataset,save_dir,cut_size,step,save_empty=False,keep_ratio=1e-6):
    '''
    在图像上滑动窗口并裁图
    cut_size: (h,w)
    '''
    if step<=0:
        sx = cut_size[1]
        sy = cut_size[0]
    elif step<=1.0:
        sx = int(cut_size[1]*step)
        sy = int(cut_size[0]*step)
    else:
        sx = step
        sy = step

    for idx,data in enumerate(dataset):
        print(f"Process {idx}/{len(dataset)}")
        img_file, shape,labels, labels_names, bboxes,binary_masks,*_ = data
        r_path = wmlu.get_relative_path(img_file,data_dir)
        bs_name,suffix = osp.splitext(r_path)
        y0 = 0
        img = wmli.imread(img_file)
        while y0<img.shape[0]:
            x0 = 0
            while x0<img.shape[1]:
                cut_bbox = np.array((y0,x0,y0+cut_size[0],x0+cut_size[1]))
                new_img,new_labels,new_bboxes,new_masks = odp.cut_annotation(cut_bbox,img,labels_names,bboxes,binary_masks,keep_ratio=keep_ratio)

                if len(new_labels) == 0 and not save_empty:
                    x0 += sx 
                    continue

                save_base_name = bs_name+f"_{y0}_{x0}"
                img_save_path = osp.join(save_dir,save_base_name+suffix)
                wmli.imwrite(img_save_path,new_img)
    
                if binary_masks is not None:
                    new_masks,*_ = mu.cut_masks(new_masks,new_bboxes)
                    new_bboxes = odb.npchangexyorder(new_bboxes)
                    ann_save_path = osp.join(save_dir,save_base_name+".json")
                    save_labelme_datav3(ann_save_path,img_save_path,new_labels,new_bboxes,new_masks)
                else:
                    ann_save_path = osp.join(save_dir,save_base_name+".xml")
                    write_voc_xml(ann_save_path,img_save_path,new_img.shape,new_bboxes,new_labels,is_relative_coordinate=False)
                
                x0 += sx
            
            y0 += sy


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    dataset_type = args.type
    if dataset_type == "xml":
        dataset = pascal_voc_dataset(data_dir=data_dir,
                                     labels=args.labels,
                                     )
    elif dataset_type=="json":
        dataset = labelme_dataset(data_dir=data_dir,
                                  labels=args.labels
                                  )
    
    cut_size = args.size
    if len(cut_size) == 1:
        cut_size = [cut_size[0],cut_size[0]]
    else:
        cut_size = list(cut_size)[::-1] #(w,h)->(h,w)

    keep_ratio = args.keep_ratio
    wmlu.create_empty_dir_remove_if(args.save_dir)
    
    if args.cut_type == 0:
        cut_arount_bboxes_and_save(dataset,data_dir,args.save_dir,cut_size,keep_ratio=keep_ratio)
    else:
        cut_bboxes_and_save(dataset,data_dir,args.save_dir,cut_size,args.step,args.save_empty,keep_ratio=keep_ratio)
    
    
