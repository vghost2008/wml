from .labelme_toolkit import *
from .fast_labelme import *
from .pascal_voc_toolkit import *
from .coco_toolkit import *
from .object365v2_toolkit import *
from .baidu_mask_toolkit import *
from .labelmemckeypoints_dataset import LabelmeMCKeypointsDataset
from .common import get_auto_dataset_suffix,check_dataset_dir
from .image_folder import ImageFolder

def get_auto_dataset_type(data_dir):
    for f in wmlu.find_files(data_dir,suffix=".json"):
        return FastLabelMeData
    for f in wmlu.find_files(data_dir,suffix=".xml"):
        return PascalVOCData

    return PascalVOCData