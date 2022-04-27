import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def cal_mAP(target, output):
    aps = []
    for key in target:
        precision, recall, _ = precision_recall_curve(target[key], output[key])
        precision = np.fliplr([precision])[0]
        recall = np.fliplr([recall])[0]
        ap = voc_ap(recall, precision)
        aps.append(ap)
        
    mAP = sum(aps)/len(aps)
    return mAP

def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
   
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))  #[0.  0.0666, 0.1333, 0.4   , 0.4666,  1.]
    mpre = np.concatenate(([0.], prec, [0.])) #[0.  1.,     0.6666, 0.4285, 0.3043,  0.]

    # compute the precision envelope
    # 计算出precision的各个断点(折线点)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  #[1.     1.     0.6666 0.4285 0.3043 0.    ]

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]  #precision前后两个值不一样的点
    # print(mrec[1:], mrec[:-1])
    # print(i) #[0, 1, 3, 4, 5]

    # AP= AP1 + AP2+ AP3+ AP4
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
