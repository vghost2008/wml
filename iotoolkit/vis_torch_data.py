from iotoolkit.coco_toolkit import *
import img_utils as wmli
import object_detection2.visualization as odv
import os.path as osp
import random
import time

def trans_file_name(filename,image_dir):
    names = filename.split("/")[-2:]
    return osp.join(*names)
random.seed(time.time())
COCOData.trans_file_name = trans_file_name
data = TorchCOCOData("/media/vghost/Linux/constant_data/MachineLearning/mldata/objects365/val",
                     "/media/vghost/Linux/constant_data/MachineLearning/mldata/objects365/val/zhiyuan_objv2_val.json")
save_dir = "tmp/imgs"

wmlu.create_empty_dir_remove_if(save_dir)

max_show_nr = 100
idxs = list(range(len(data)))
random.shuffle(idxs)
idxs = idxs[:max_show_nr]


for idx in idxs:
    img,gts = data[idx]
    bboxes = [x['bbox'] for x in gts]
    labels = np.array([x['category_id'] for x in gts])
    bboxes = np.array(bboxes)
    bboxes[:,2:] = bboxes[:,:2]+bboxes[:,2:]
    img = np.array(img)
    img = odv.draw_bboxes_xy(
        img=img, classes=labels, scores=None, bboxes=bboxes,  color_fn=None,
        thickness=4,
        show_text=True,
        font_scale=0.8)
    save_path = osp.join(save_dir,f"{idx}.jpg")
    wmli.imwrite(save_path,img)