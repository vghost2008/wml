import os
import wml.wml_utils as wmlu
import argparse
import os.path as osp
from wml.iotoolkit.pascal_voc_toolkit import *
import wml.img_utils as wmli

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="src dir")
    parser.add_argument("--img_ext",type=str,default="jpg",help="src dir")
    parser.add_argument("--labels",nargs='+',default="##",type=str,help="src dir")
    parser.add_argument(
        '--erase-img',
        action='store_true',
        help='erase img area')
    parser.add_argument(
        '--allow-empty',
        action='store_true',
        help='allow empty img area')
    args = parser.parse_args()
    return args

def trans_one_file(xml_file,save_dir,labels,img_suffix,erase_img=True,allow_empty=False):
    img_file = wmlu.change_suffix(xml_file,img_suffix)
    if not osp.exists(img_file):
        print(f"Find {img_file} faild.")
        return
    shape, bboxes, labels_names, difficult, truncated,probs = read_voc_xml(xml_file,absolute_coord=True)
    _bboxes = []
    _labels_name = []
    remove_nr = 0
    remove_labels = []
    bboxes2remove = []
    for i,l in enumerate(labels_names):
        if l in labels:
            remove_nr += 1
            bboxes2remove.append(bboxes[i])
            remove_labels.append(l)
            continue
        _bboxes.append(bboxes[i])
        _labels_name.append(l)

    if remove_nr==0:
        if not allow_empty:
            if len(_bboxes)==0:
                print(f"Empty annotation file {xml_file}, skip.")
                return
        wmlu.try_link(img_file, save_dir)
        shutil.copy(xml_file,save_dir)
    else:
        print(f"{wmlu.base_name(xml_file)} remove {remove_nr} labels, labels is {remove_labels}")
        if not allow_empty:
            if len(_bboxes)==0:
                print(f"Empty annotation file {xml_file}, skip.")
                return
        img_save_path = osp.join(save_dir,osp.basename(img_file))
        xml_save_path = osp.join(save_dir,osp.basename(xml_file))
        img = wmli.imread(img_file)
        if erase_img:
            img = wmli.remove_boxes_of_img(img,np.array(bboxes2remove).astype(np.int32))
        wmli.imwrite(img_save_path,img)
        write_voc_xml(xml_save_path,img_save_path,shape,_bboxes,_labels_name,is_relative_coordinate=False)

if __name__ == "__main__":
    args = parse_args()

    print(f"Erase img: {args.erase_img}")
    print(f"Allow empty: {args.allow_empty}")

    files = wmlu.recurse_get_filepath_in_dir(args.src_dir,suffix=".xml")
    wmlu.create_empty_dir(args.out_dir,remove_if_exists=False)

    for xml_file in files:
        trans_one_file(xml_file,
                       args.out_dir,args.labels,args.img_ext,
                       erase_img=args.erase_img,
                       allow_empty=args.allow_empty)
