#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:07:56 2018

@author: amazinger
"""
import numpy as np
from proj1_helpers import *
from project_func import *
from preprocessing import *

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    yx = zip(y,x)
    np.random.shuffle(yx)
    yx = np.array(yx)
    ind_cut = int(y.shape[0]*ratio)
    x_tr, x_te = yx[:ind_cut,1:], yx[ind_cut:,1:]
    y_tr, y_te = yx[:ind_cut,0], yx[ind_cut:,0]
    return x_tr, x_te, y_tr, y_te
    # split the data based on the given ratio: TODO
    # ***************************************************
    raise NotImplementedError
    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    k_fold = int(k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_logistic(y, tx, initial_w, max_iters, alpha, k_fold, seed):
    scores = []
    k_indices = build_k_indices(y, k_fold, seed)
    print k_indices
    for k in k_indices:
        y_va = y[k]
        tx_va = tx[k]
        y_tr = y[k_indices[k_indices != k]]
        tx_tr = tx[k_indices[k_indices != k]]
        u_tr = (1+y_tr)/2.0
        ws = logistic_regression(u_tr, tx_tr, initial_w, max_iters, alpha)
        print ws
        y_va_pre = predict_labels(tx_va,ws)
        score = (y_va_pre == y_va).mean()
        print score
        scores.append(score)
    return scores ## contain the mean and the std infrmations