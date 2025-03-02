from wml.iotoolkit.sports_mot_datasets import SportsMOTDatasets
from wml.iotoolkit.pascal_voc_toolkit import *

def trans_data(data_dir,use_det=False):
    dataset = SportsMOTDatasets(data_dir, absolute_coord=True,use_det=use_det)
    for img_path, shape, labels, _, bboxes,*_ in dataset.get_data_items():
        labels = [0]*len(labels)
        writeVOCXml(img_path,bboxes,labels,img_shape=shape,is_relative_coordinate=False)


if __name__ == "__main__":
    #trans_data("/home/wj/ai/mldata1/SportsMOT-2022-4-24/data/sportsmot_publish/dataset/val")
    #trans_data("/home/wj/ai/mldata/MOT/MOT16/train",use_det=False)
    trans_data("/home/wj/ai/mldata/MOT/MOT20/test",use_det=True)
    #trans_data("/home/wj/ai/mldata/MOT/MOT17/test",use_det=True)
