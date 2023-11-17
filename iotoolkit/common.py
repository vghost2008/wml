import os
import wml_utils as wmlu
import PIL
import img_utils as wmli
import os.path as osp
import glob

def __get_resample_nr(labels,resample_parameters):
    nr = 1
    for l in labels:
        if l in resample_parameters:
            nr = max(nr,resample_parameters[l])
    return nr

def resample(files,labels,resample_parameters):
    '''
    files: list of files
    labels: list of labels
    resample_parameters: {labels:resample_nr}
    '''
    new_files = []
    repeat_files_nr = 0
    repeat_nr = 0
    for f,l in zip(files,labels):
        nr = __get_resample_nr(l,resample_parameters)
        if nr>1:
            new_files = new_files+[f]*nr
            #print(f"Repeat {f} {nr} times.")
            repeat_files_nr += 1
            repeat_nr += nr
        elif nr==1:
            new_files.append(f)
    print(f"Total repeat {repeat_files_nr} files, total repeat {repeat_nr} times.")
    print(f"{len(files)} old files --> {len(new_files)} new files")

    return new_files

def get_shape_from_img(xml_path,img_path):
    '''
    return: [H,W]
    '''
    if not os.path.exists(img_path):
        img_path = wmlu.change_suffix(xml_path, "jpg")
        if not os.path.exists(img_path):
            img_path = wmlu.change_suffix(xml_path, "jpeg")
    return wmli.get_img_size(img_path)

def ignore_case_dict_label_text2id(name,dict_data):
    name = name.lower()
    if name not in dict_data:
        print(f"ERROR: trans {name} faild.")
    return dict_data.get(name,None)

def dict_label_text2id(name,dict_data):
    if name not in dict_data:
        print(f"ERROR: trans {name} faild.")
    return dict_data.get(name,None)

def find_imgs_for_ann_file(ann_path):
    ann_path = osp.abspath(ann_path)
    img_suffix = [".jpg",".jpeg",".bmp",".png",".gif"]
    pattern = wmlu.change_suffix(ann_path,"*")
    files = glob.glob(pattern)
    img_file = None
    for file in files:
        if file==ann_path:
            continue
        if osp.splitext(file)[1].lower() in img_suffix:
            img_file = file
        else:
            print(f"WARNING: Unknow img format file {file} for {ann_path}")
            img_file = file
    return img_file