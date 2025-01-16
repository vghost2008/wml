import wml.wml_utils as wmlu
import glob
import os.path as osp
import numpy as np
import os
import wml.object_detection2.visualization as odv
import cv2
import shutil

class MOTDataset(object):

    img_name_pattern="{:06d}"
    img_suffix=".jpg"
    data_filter = lambda cls:cls<=1

    def __init__(self):
        self.raw_data = None
        self._fid_dict = None
        self._pid_dict = None
        self.img_dir = None
        pass

    def read(self,txt_path,img_dir=None):
        with open(txt_path, encoding="utf-8") as f:
            content = f.readlines()
        self.raw_data = []
        self._fid_dict = None
        self._pid_dict = None
        self.img_dir = img_dir
        for line in content:
            f_id, p_id, xmin, ymin, w, h, _,cls,*_ = [float(x) for x in line.strip().split(",")]
            f_id = int(f_id)
            p_id = int(p_id)
            cls = int(cls)
            if not MOTDataset.data_filter(cls):
                continue
            bbox = (xmin,ymin,xmin+w,ymin+h)
            self.raw_data.append([f_id,p_id,bbox])
    
    def bboxes(self):
        if self.raw_data is None:
            return np.zeros([0,4],dtype=np.float32)
        bboxes = []
        for x in self.raw_data:
            bboxes.append(x[2])
        return np.array(bboxes,dtype=np.float32)

    @property
    def fid_dict(self):
        if self._fid_dict is None and self.raw_data is not None:
            self._fid_dict = wmlu.MDict(dtype=list)
            for fid,pid,bbox in self.raw_data:
                self._fid_dict[fid].append([fid,pid,bbox])
        return self._fid_dict

    @property
    def pid_dict(self):
        if self._pid_dict is None and self.raw_data is not None:
            self._pid_dict = wmlu.MDict(dtype=list)
            for fid,pid,bbox in self.raw_data:
                self._pid_dict[pid].append([fid,pid,bbox])
        return self._pid_dict

    def draw_tracks(self,save_dir,img_dir=None):
        if img_dir is not None:
            self.img_dir = img_dir
        else:
            img_dir = self.img_dir
        total_files_nr = len(glob.glob(osp.join(img_dir,"*"+self.img_suffix)))
        mot_data = self.fid_dict

        os.makedirs(save_dir,exist_ok=True)
        img_name_pattern = self.img_name_pattern+self.img_suffix

        for fid in range(1,total_files_nr+1):
            img_name = img_name_pattern.format(fid)
            img_path = osp.join(img_dir,img_name)
            cur_data = mot_data[fid]
            pids = []
            bboxes = []
            for _,pid,bbox in cur_data:
                pids.append(pid)
                bboxes.append(bbox)
            if len(pids)>0:
                pids = np.array(pids,dtype=np.int32)
                bboxes = np.array(bboxes,dtype=np.float32)
                img = cv2.imread(img_path)
                img = odv.draw_bboxes_xy(img, pids, bboxes=bboxes,
                                       color_fn=odv.fixed_color_fn,
                                       is_relative_coordinate=False,
                                       show_text=True,
                                       thickness=2,
                                       font_scale=1.0)
                cv2.imwrite(osp.join(save_dir,img_name),img)
            else:
                shutil.copy(img_path,save_dir)

    @staticmethod
    def draw_one_tracks(txt_path,img_dir,save_dir):
        if not osp.exists(txt_path):
            print(f"{txt_path} not exists.")
            return

        if not osp.isdir(img_dir):
            print(f"{img_dir} is not a dir.")
            return

        ds = MOTDataset()
        ds.read(txt_path,img_dir)
        ds.draw_tracks(save_dir)

    @staticmethod
    def draw_multi_tracks(txt_dir,img_dir,save_dir,img_loc_format="{img_dir}/{seq}/img1"):
        txt_files = glob.glob(osp.join(txt_dir,"*.txt"))
        for i,txt_file in enumerate(txt_files):
            basename = wmlu.base_name(txt_file)
            print(f"Draw {basename} {i+1}/{len(txt_files)}")
            cur_img_dir = img_loc_format.format(img_dir=img_dir,seq=basename)
            cur_save_dir = osp.join(save_dir,basename)
            MOTDataset.draw_one_tracks(txt_file,cur_img_dir,cur_save_dir)

    @staticmethod
    def draw_multi_tracksv2(dir_path,save_dir,img_dir="img1",txt_file ="gt/gt.txt"):
        dirs = wmlu.get_subdir_in_dir(dir_path)
        for i,basename in enumerate(dirs):
            print(f"Draw {basename} {i+1}/{len(dirs)}")
            cur_img_dir = osp.join(dir_path,basename,img_dir)
            txt_path = osp.join(dir_path,basename,txt_file)
            cur_save_dir = osp.join(save_dir,basename)
            MOTDataset.draw_one_tracks(txt_path,cur_img_dir,cur_save_dir)
