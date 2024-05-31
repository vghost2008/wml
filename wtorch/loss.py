import torch
import torch.nn.functional as F

def focal_loss_for_heat_map(labels,logits,pos_threshold=0.99,alpha=2,beta=4,sum=True):
    '''
    focal loss for heat map, for example CenterNet2's heat map loss
    '''
    logits = logits.to(torch.float32)
    zeros = torch.zeros_like(labels)
    ones = torch.ones_like(labels)
    num_pos = torch.sum(torch.where(torch.greater_equal(labels, pos_threshold), ones, zeros))

    probs = F.sigmoid(logits)
    pos_weight = torch.where(torch.greater_equal(labels, pos_threshold), ones - probs, zeros)
    neg_weight = torch.where(torch.less(labels, pos_threshold), probs, zeros)
    '''
    用于保证数值稳定性，log(sigmoid(x)) = log(1/(1+e^-x) = -log(1+e^-x) = x-x-log(1+e^-x) = x-log(e^x +1)
    pos_loss = tf.where(tf.less(logits,0),logits-tf.log(tf.exp(logits)+1),tf.log(probs))
    '''
    pure_pos_loss = -torch.minimum(logits,logits.new_tensor(0,dtype=logits.dtype))+torch.log(1+torch.exp(-torch.abs(logits)))
    pos_loss = pure_pos_loss*torch.pow(pos_weight, alpha)
    if sum:
        pos_loss = torch.sum(pos_loss)
    '''
    用于保证数值稳定性
    '''
    pure_neg_loss = F.relu(logits)+torch.log(1+torch.exp(-torch.abs(logits)))
    neg_loss = torch.pow((1 - labels), beta) * torch.pow(neg_weight, alpha) * pure_neg_loss
    if sum:
        neg_loss = torch.sum(neg_loss)
    loss = (pos_loss + neg_loss) / (num_pos + 1e-4)
    return loss

def focal_mse_loss_for_head_map(gt_value,pred_value,alpha=0.25,gamma=2):
    '''
    alpha: pos loss x alpha, neg loss x (1-alpha)
    gamma: loss x loss^gamma
    gt_value: value range [0,1]
    注：
    
    '''
    assert gt_value.numel()==pred_value.numel(), f"ERROR: unmatch gt_value's shape {gt_value.shape} and pred_value's shape {pred_value.shape}"
    gt_value = gt_value.float()
    pred_value = pred_value.float()
    loss = torch.square(gt_value-pred_value)
    bin_gt_value = torch.greater(gt_value,0).to(pred_value.dtype)
    total_nr = bin_gt_value.new_tensor(torch.numel(bin_gt_value))
    #neg_scale = torch.sum(bin_gt_value)/total_nr
    #pos_scale = bin_gt_value.new_tensor(1.0)-neg_scale
    num_pos = torch.sum(bin_gt_value)
    num_neg = total_nr-num_pos

    neg_scale = total_nr/(num_neg*2.0+2e-1)
    pos_scale = total_nr/(num_pos*2.0+2e-1)

    auto_t = bin_gt_value*pos_scale+(1-bin_gt_value)*neg_scale
    loss = loss*auto_t


    if alpha is not None and alpha>0:
        alpha_t = bin_gt_value*alpha+(1-bin_gt_value)*(1-alpha)
        alpha_t = alpha_t/torch.clip(torch.mean(alpha_t).detach(),min=1e-6)
        loss = loss*alpha_t
    if gamma is not None and gamma>0:
        gamma_t = torch.pow(torch.abs(gt_value-pred_value),gamma)
        gamma_t = gamma_t/torch.clip(torch.mean(gamma_t).detach(),min=1e-6)
        loss = loss*gamma_t
    
    return loss