import img_utils as wmli
import wml_utils as wmlu
from wtorch.data.dataloader import DataLoader
from wtorch.data._utils.collate import null_convert

class ImgsDataset:
    def __init__(self,data_dir):
        self.files = wmlu.get_files(data_dir,suffix=".jpg;;.jpeg;;.bmp;;.png")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path = self.files[item]
        return path,wmli.imread(path)

class ImgsReader:
    def __init__(self, data_dir, thread_nr=8):
        self.dataset = ImgsDataset(data_dir,)

        dataloader_kwargs = {"num_workers": thread_nr, "pin_memory": False}
        dataloader_kwargs["sampler"] = list(range(len(self.dataset)))
        dataloader_kwargs["batch_size"] = None
        dataloader_kwargs["batch_split_nr"] = 1
        dataloader_kwargs['collate_fn'] = null_convert

        data_loader = DataLoader(self.dataset, **dataloader_kwargs)

        self.data_loader = data_loader
        self.data_loader_iter = iter(self.data_loader)

    def __iter__(self):
        return self.data_loader_iter


if __name__ == "__main__":
    import os.path as osp
    import os
    reader = ImgsReader("/home/vghost/pic1")
    save_path = "/home/vghost/pic2"
    os.makedirs(save_path,exist_ok=True)
    for path,img in reader:
        sp = osp.join(save_path,osp.basename(path))
        wmlu.try_link(path,sp)
        sp = wmlu.change_name(sp,suffix="_a")
        wmli.imwrite(sp,img)

