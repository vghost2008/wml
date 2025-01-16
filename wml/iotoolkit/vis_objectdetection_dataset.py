import wml.img_utils as wmli
import random
import matplotlib.pyplot as plt
import time
import wml.wml_utils as wmlu
import os.path as osp
import wml.object_detection2.visualization as odv
import numpy as np

def trans_name(name:str):
    return name.replace("/","_")


def vis_dataset(dataset,save_dir,max_nr_per_dir=20,max_view_imgs=2000,is_relative_coordinate=False):
    wmlu.create_empty_dir_remove_if(save_dir)
    idxs = list(range(len(dataset)))
    random.seed(time.time())
    idxs = idxs[:max_view_imgs]
    random.shuffle(idxs)

    def text_fn(classes, scores):
        return dataset.id2name[classes]

    counter = wmlu.MDict(dvalue=0)
    for id,name in dataset.id2name.items():
        counter[id] = 0

    for idx in idxs:
        x = dataset[idx]
        full_path, shape,category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        category_ids = np.array(category_ids)
        boxes = np.array(boxes)
        need_view = False
        for id in category_ids:
            if counter[id]<max_nr_per_dir:
                need_view = True
                break
        if not need_view:
            continue

        img = wmli.imread(full_path)


        t_img = odv.draw_bboxes(
            img=np.copy(img), classes=category_ids, scores=None, bboxes=boxes, color_fn=None,
            text_fn=text_fn, thickness=1,
            show_text=True,
            font_scale=0.8,
            is_relative_coordinate=is_relative_coordinate)
        file_basename = osp.basename(full_path)
        save_path = osp.join(save_dir,"all",file_basename)
        wmli.imwrite(save_path,t_img)

        for id in set(category_ids):
            if counter[id]<max_nr_per_dir:
                mask = category_ids==id
                t_labels = category_ids[mask]
                t_boxes = boxes[mask]
                t_img = odv.draw_bboxes(
                    img=np.copy(img), classes=t_labels, scores=None, bboxes=t_boxes, color_fn=None,
                    text_fn=text_fn, thickness=1,
                    show_text=True,
                    font_scale=0.8,
                    is_relative_coordinate=is_relative_coordinate)
                counter[id] = counter[id]+1
                t_save_path = osp.join(save_dir,str(id)+"_"+trans_name(dataset.id2name[id]),file_basename)
                wmlu.make_dir_for_file(t_save_path)
                wmli.imwrite(t_save_path,t_img)
        if np.all(np.array(list(counter.values()))>max_nr_per_dir):
            break