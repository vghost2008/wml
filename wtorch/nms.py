import time
import numpy as np
import torch
import torchvision


def __soft_nms(dets, box_scores, thresh=0.001, sigma=0.5):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function, decay=exp(-(iou*iou)/sigma), smaller sigma, decay faster.
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    device = dets.device
    indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    indexes = indexes.to(device)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos + i + 1] = dets[maxpos + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos + i + 1] = scores[maxpos + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = torch.maximum(dets[i, 0][None], dets[pos:, 0])
        xx1 = torch.maximum(dets[i, 1][None], dets[pos:, 1])
        yy2 = torch.minimum(dets[i, 2][None], dets[pos:, 2])
        xx2 = torch.minimum(dets[i, 3][None], dets[pos:, 3])

        zeros = torch.zeros_like(yy1)
        w = torch.maximum(zeros, xx2 - xx1)
        h = torch.maximum(zeros, yy2 - yy1)
        inter = w * h
        ious = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ious* ious) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    valid_mask = scores>thresh
    keep = dets[:, 4][valid_mask].long()
    valid_scores = scores[valid_mask]

    return keep,valid_scores

def soft_nms(dets, box_scores, thresh=0.001, sigma=0.5):
    keep,_ = __soft_nms(dets,box_scores,thresh=thresh,sigma=sigma)
    return keep

def soft_nmsv2(dets, box_scores, thresh=0.001, sigma=0.5, cuda=0):
    keep,scores  = __soft_nms(dets,box_scores,thresh=thresh,sigma=sigma)

    return keep,scores

def group_nms(bboxes,scores,ids,nms_threshold:float=0.7):
    '''
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
    '''
    max_value = torch.max(bboxes)+1.0
    tmp_bboxes = bboxes+ids[:,None].to(bboxes.dtype)*max_value
    idxs = torchvision.ops.nms(tmp_bboxes,scores,nms_threshold)
    return idxs