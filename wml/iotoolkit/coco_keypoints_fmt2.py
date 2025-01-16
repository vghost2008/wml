from pycocotools.coco import COCO
import numpy as np
import copy

'''
返回image层面的结果
'''
class COCOKeypointsFmt2(object):
    def __init__(self):
        self.pixel_std = 200
        self.num_joints = 17
        pass

    def read_data(self,file_path):
        self.coco = COCO(file_path)
        self.image_set_index = self.coco.getImgIds()
        self.gt_db = self._load_coco_keypoint_annotations()
        return self.gt_db

    def image_path_from_index(self, index):
        file_name = '%012d.jpg' % index
        return file_name

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            res = self._load_coco_keypoint_annotation_kernal(index)
            if res is not None:
                gt_db.append(res)
        return gt_db


    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        if len(objs) == 0:
            return None

        res = {}
        res['image'] = self.image_path_from_index(index)
        all_kps = []
        all_bboxes = []
        all_areas = []
        all_iscrowd = []
        for obj in objs:
            '''cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue'''
            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d[ipt, 2] = t_vis

            clean_bbox = np.array(obj['clean_bbox'])
            bbox = clean_bbox
            bbox[2:] = bbox[:2]+bbox[2:]
            all_kps.append(joints_3d)
            all_bboxes.append(bbox)
            all_areas.append(obj['area'])
            all_iscrowd.append(obj['iscrowd'])

        res["keypoints"] = np.array(all_kps,dtype=np.float32)
        res['bboxes'] = np.array(all_bboxes,dtype=np.float32)
        res['area'] = np.array(all_areas,dtype=np.float32)
        res['iscrowd'] = np.array(all_iscrowd,dtype=np.float32)
        return res
