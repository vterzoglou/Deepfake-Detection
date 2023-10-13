import numpy as np
from tensorflow import math, reduce_sum, cast, float32
# [1,0]->real/negative, [0,1]->fake/positive


def TP(y_true, y_pred):
    yt = y_true[:, 1]
    yp = y_pred[:, 1]
    pos_idx = yp > 0.5
    return reduce_sum(cast(math.abs(yp[pos_idx]-yt[pos_idx]) < 0.5, float32))


# FP = negative classified as positive
def FP(y_true, y_pred):
    yt = y_true[:, 1]
    yp = y_pred[:, 1]
    pos_idx = yp > 0.5
    return reduce_sum(cast(math.abs(yp[pos_idx]-yt[pos_idx]) > 0.5, float32))


# TN = negative classified as negative
def TN(y_true, y_pred):
    yt = y_true[:, 0]
    yp = y_pred[:, 0]
    neg_idx = yp > 0.5
    return reduce_sum(cast(math.abs(yp[neg_idx]-yt[neg_idx]) < 0.5, float32))


# FN = positive classified as negative
def FN(y_true, y_pred):
    yt = y_true[:, 0]
    yp = y_pred[:, 0]
    neg_idx = yp > 0.5
    return reduce_sum(cast(math.abs(yp[neg_idx]-yt[neg_idx]) > 0.5, float32))


def precision(y_true, y_pred):
    return TP(y_true, y_pred)/(TP(y_true, y_pred) + FP(y_true, y_pred))


def recall(y_true, y_pred):
    return TP(y_true, y_pred)/(TP(y_true, y_pred) + FN(y_true, y_pred))


def accuracy(y_true,y_pred):
    return (TP(y_true,y_pred)+TN(y_true,y_pred))/(TP(y_true,y_pred)+TN(y_true,y_pred)+FP(y_true,y_pred)+FN(y_true,y_pred))


def TPR(y_true,y_pred):
    return TP(y_true,y_pred)/(TP(y_true,y_pred)+FN(y_true,y_pred))


def TNR(y_true,y_pred):
    return TN(y_true,y_pred)/(TN(y_true,y_pred)+FP(y_true,y_pred))


def balanced_accuracy(y_true,y_pred):
    tpr = TPR(y_true,y_pred)
    tnr = TNR(y_true,y_pred)
    return (tpr+tnr)/cast(2,float32)