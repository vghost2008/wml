import argparse
import glob
import os.path as osp
import wml.wml_utils as wmlu
from wml.iotoolkit.pascal_voc_toolkit import write_voc_xml
from wml.iotoolkit.labelme_toolkit_fwd import save_labelme_data,save_detdata
from wml.wstructures import DetBboxesData
import numpy as np
import wml.img_utils as wmli

'''
给图像文件生成空的xml/json标注文件
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='arguments')
    parser.add_argument('img_dir', default=None,type=str,help='img_dir')
    parser.add_argument('--type', default="json",type=str,help='img_dir')
    parser.add_argument('--label', type=str,help='default label')
    args = parser.parse_args()
    return args

def get_all_imgs(img_dir,img_suffix=".jpg;;.jpeg;;.png;;.bmp"):
    files = wmlu.get_files(img_dir,suffix=img_suffix)
    return files

def make_xml_for_imgs(img_dir,ann_type="xml",args=None):
    all_img_files = get_all_imgs(img_dir)
    bboxes = np.zeros([0,4])
    labels_text = []
    for file in all_img_files:
        save_path = wmlu.change_suffix(file,"xml")
        if osp.exists(save_path):
            print(f"Skip {save_path}")
            continue
        print(f"Save {save_path}")
        write_voc_xml(save_path,file,[0,0], bboxes, labels_text)

def make_json_for_imgs(img_dir,ann_type="json",args=None):
    all_img_files = get_all_imgs(img_dir)
    for file in all_img_files:
        try:
            save_path = wmlu.change_suffix(file,"json")
            #if "false" not in save_path.lower():
                #continue
            if osp.exists(save_path):
                wmlu.print_warning(f"Skip {save_path}")
                continue
            print(f"Save {save_path}")
            if args.label is not None and len(args.label)>0:
                label = args.label
                if label == "auto":
                    label = wmlu.base_name(osp.dirname(file))
                img_shape = wmli.get_img_size(file)
                labels = [label]
                bboxes = np.array([[0,0,img_shape[1]-1,img_shape[0]-1]])
                detbboxes = DetBboxesData(file,img_shape,labels,bboxes,[False])
                save_detdata(save_path,file,detbboxes,None)
            else:
                save_labelme_data(save_path,file,None,[])
        except Exception as e:
            wmlu.print_error(f"{e}")

if __name__ == "__main__":
    args = parse_args()
    if args.type == "xml":
        make_xml_for_imgs(args.img_dir,ann_type=args.type,args=args)
    elif args.type == "json":
        make_json_for_imgs(args.img_dir,args.type,args=args)
