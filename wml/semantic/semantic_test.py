from wml.semantic.structures import *
from wml.iotoolkit.labelme_toolkit import LabelMeData
import wml.img_utils as wmli
import wml.object_detection2.visualization as odv

if __name__ == "__main__":
    data = LabelMeData(use_polygon_mask=True)
    data.read_data("~/ai/mldata1/B10CF/datasets/testv1.0/")
    d = data[0]
    mask = d[5]
    crop_bbox = [1095,500,1235,600]
    crop_bbox = [800,500,1235,600]
    crop_bbox = [800,300,1235,600]
    crop_bbox = [600,300,1600,600]
    mask = mask.crop(crop_bbox)
    img = wmli.imread(d[0])
    img = wmli.crop_img_absolute_xy(img,crop_bbox)
    img = odv.draw_maskv2(img,d[3],None,mask,is_relative_coordinate=False)
    wmli.imwrite("tmp.jpg",img)

    
