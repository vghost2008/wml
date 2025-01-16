import numpy as np

'''
根据scores得分,从输入中仅取一个类别
'''
def classes_suppression(bboxes,labels,scores,test_nr=1):
    if len(labels)==0:
        return bboxes,labels,scores
    max_scores = -1
    max_l = -1
    u_labels = set(labels.tolist())
    for l in u_labels:
        m = labels==l
        idx = np.argsort(-scores[m])[:test_nr]
        t_scores = np.mean(scores[m][idx])
        if t_scores>max_scores:
            max_scores = t_scores
            max_l = l
    
    m = labels==max_l

    return bboxes[m],labels[m],scores[m]


