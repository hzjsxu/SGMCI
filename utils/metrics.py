'''
Author: xujs
Date: 2024-04-17 10:20:10
LastEditors: jsxu
LastEditTime: 2024-04-17 10:26:47
FilePath: /H2P/utils/metrics.py
Contact me: hzaujsxu@163.com
Description: 
'''
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
import numpy as np


def binaryf1(pred, label):
    '''
    pred, label are numpy array
    can process multi-label target
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return f1_score(label_i, pred_i, average="micro")


def microf1(pred, label):
    '''
    multi-class micro-f1
    '''
    pred_i = np.argmax(pred, axis=1)
    return f1_score(label, pred_i, average="micro")


# def auroc(pred, label):
#     '''
#     calculate auroc
#     '''
#     return roc_auc_score(label, pred)

def auroc(pred, label, size_list):
    '''
    calculate auroc
    '''
    roc_str = ""

    ### For all size.
    y_t = (label > 0.5).reshape((-1, 1))
    y_p = pred.reshape((-1, 1))
    y_t_ = y_t > 0.5
    y_p_ = y_p > 0.5
    roc_all = roc_auc_score(y_t, y_p)

    roc_str += "%s %.4f " % ('all', roc_all)

    ### For each size.
    # for s in np.unique(size_list):  ## 用这个会出现size=10时, test为0从而报错的情况
    for s in [3, 4, 5]:
        y_t = (label[size_list == s] > 0.5).reshape((-1, 1))
        y_p = (pred[size_list == s]).reshape((-1, 1))
        roc = roc_auc_score(y_t, y_p)

        roc_str += "%s %.4f " % (str(s), roc)
    
    return roc_str[:-1], roc_all
        



    return roc_auc_score(label, pred)


def aupr(pred, label):
    '''
    calculate aupr
    '''
    return average_precision_score(label, pred)


def acc(pred, label):
    '''
    calculate accuracy
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return accuracy_score(label_i, pred_i)

def prec(pred, label):
    '''
    calculate precision
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return precision_score(label_i, pred_i)

def rec(pred, label):
    '''
    calculate recall
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return recall_score(label_i, pred_i)

def all_score(pred, label):
    auc_ = auroc(pred, label)
    aupr_ = aupr(pred, label)
    acc_ = aupr(pred, label)
    prec_ = prec(pred, label)
    rec_ = rec(pred, label)
    f1_ = binaryf1(pred, label)
    return auc_, aupr_, acc_, prec_, rec_, f1_
    