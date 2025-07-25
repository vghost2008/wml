import sys
from wml.iotoolkit.labelme_toolkit import *
import wml.img_utils as wmli
import numpy as np
import wml.object_detection2.mask as odm
import wml.wml_utils as wmlu
import cv2
import argparse
import functools

lid = 0

'''name_to_id_dict = {
"construction--flat--road":0,
"construction--flat--sidewalk":1,
"object--street-light":2,
"construction--structure--bridge":3,
"construction--structure--building":4,
"human":5,
"object--support--pole":6,
"marking--continuous--dashed":7,
"marking--continuous--solid":8,
"marking--discrete--crosswalk-zebra":9,
"nature--sand":10,
"nature--sky":11,
"nature--snow":12,
"nature--terrain":13,
"nature--vegetation":14,
"nature--water":15,
"object--vehicle--bicycle":16,
"object--vehicle--boat":17,
"object--vehicle--bus":18,
"object--vehicle--car":19,
"object--vehicle--vehicle-group":19,
"object--vehicle--caravan":20,
"object--vehicle--motorcycle":21,
"object--vehicle--on-rails":22,
"object--vehicle--truck":23,
"construction--flat--pedestrian-area":24,
"construction--structure--tunnel":25,
"nature--wasteland":26,
}
resize_size = (2560,1440)'''
name_to_id_dict = {
}
resize_size = None

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="out dir")
    parser.add_argument("--labels",type=str,nargs="+",help="labels")
    args = parser.parse_args()
    return args

def trans_data(data_dir,save_dir,labels):
    global name_to_id_dict
    wmlu.show_dict(name_to_id_dict)
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)

    def name_to_id(x,data):
        if x in data:
            return data[x]
        elif len(data)==0:
            data[x] = 0
            print("add first label:",data)
            return 0
        else:
            ml = max(list(data.values()))
            data[x] = ml+1
            print(f"add label {x}:",data)
            return data[x]

    data = LabelMeData(label_text2id=functools.partial(name_to_id,data=name_to_id_dict), shuffle=False)
    data.read_data(data_dir)
    for i,x in enumerate(data.get_items()):
        full_path, img_info, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x

        if len(category_ids) == 0:
            print(f"Skip {full_path}")
            continue

        new_mask = odm.dense_mask_to_sparse_mask(binary_mask,category_ids,default_label=255)
        #r_base_name = wmlu.base_name(full_path)
        r_base_name = f"IMG_{i+1:05d}"
        base_name = r_base_name+".png"
        save_path = os.path.join(save_dir,base_name)
        if resize_size is not None:
            new_mask = wmli.resize_img(new_mask,resize_size,keep_aspect_ratio=True,interpolation=cv2.INTER_NEAREST)
            img  = wmli.imread(full_path)
            img = wmli.resize_img(img,resize_size,keep_aspect_ratio=True)
            img_save_path = os.path.join(save_dir,r_base_name+".jpg")
            wmli.imwrite(img_save_path,img)
        else:
            img_save_path = os.path.join(save_dir,r_base_name+".jpg")
            wmli.read_and_write_img(full_path,img_save_path)

        new_mask = new_mask.astype(np.uint8)
        if os.path.exists(save_path):
            print(f"WARNING: File {save_path} exists.")
        wmli.imwrite_mask(save_path,new_mask)
        sys.stdout.write(f"\r{i}")

if __name__ == "__main__":
    '''data_dir ="/home/wj/ai/mldata/boesemantic/videos_rgb_15"
    save_dir = os.path.join("/home/wj/ai/mldata/boesemantic/",'boe_labels_validation')
    trans_data(data_dir,save_dir)'''

    args = parse_args()

    data_dir = args.src_dir
    save_dir = args.out_dir
    trans_data(data_dir, save_dir,args.labels)
