import torch

def soft_one_hot(labels,confidence,smooth_cfg,num_classes):
    '''
    labels: any shape in, value in [0,num_classes)
    smooth_cfg:Example:
    {1:[3,4], #标签1向3，4平滑
     3:[1,4],
     5:{2:0.04,3:0.06}, #标签5向2,3平滑，2得到0.04的置信度，3得到0.06的置信度
    }
    '''
    old_shape = labels.shape
    labels = torch.reshape(labels,[-1])
    targets = torch.zeros([labels.shape[0],num_classes],dtype=torch.float32)
    smooth_labels = set(list(smooth_cfg.keys()))
    h_labels = labels.cpu().numpy()
    targets[torch.arange(targets.shape[0]),h_labels] = 1.0
    for i,l in enumerate(h_labels):
        if l in smooth_labels:
            d = smooth_cfg[l]
            if isinstance(d,dict):
                for k,v in d.items():
                    targets[i,k] = v
            else:
                smooth_v = (1-confidence)/len(d)
                for k in d:
                    targets[i,k] = smooth_v
            targets[i,l] = confidence

    targets = targets.to(labels.device)
    ret_shape = list(old_shape)+[num_classes]
    targets = torch.reshape(targets,ret_shape)

    return targets