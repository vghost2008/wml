import wml_utils as wmlu
import os.path as osp
import img_utils as wmli
import copy

class ImageFolder:
    def __init__(self,label_text2id=None,classes=None):
        self.label_text2id = label_text2id
        self._classes = classes
        self._data = []
        pass

    def read_data(self,data_dir):
        sub_dirs = wmlu.get_subdir_in_dir(data_dir)
        self._data = []
        if self._classes is None:
            self._classes = copy.deepcopy(sub_dirs)
        for dir in sub_dirs:
            label = dir
            if self.label_text2id is not None:
                label = self.label_text2id(label)
            files = wmlu.get_files(osp.join(data_dir,dir),suffix=wmli.BASE_IMG_SUFFIX)
            for f in files:
                self._data.append((label,f))

    @property
    def classes(self):
        return self._classes
            
    def __len__(self):
        return len(self._data)

    def __getitem__(self,idx):
        return self._data[idx]
