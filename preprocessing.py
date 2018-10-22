#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:35:29 2018

@author: amazinger
"""
import numpy as np
import numpy.ma as ma
from project_func import *

    
def replace_invalid(x,value_invalid):
    
    x_masked = ma.masked_equal(x,value_invalid) 
    while np.sum(x_masked.mask > 0):
        x_masked = ma.masked_equal(x,value_invalid)         # a mask_matrix with the mask indicating where = -999, if = True
        if_clean = np.sum(x_masked.mask, axis = 0)
        if_clean = ma.masked_greater_equal(if_clean, 1).mask    # to indicate that which column is clean. If = 1 then is not clean
        x_clean = (x_masked.T[~if_clean]).T        # The column which don't contain the -999, extracted from the x 
        x_dirty = (x_masked.T[if_clean]).T         # The column which do contain the -999, extracted from the x 
        index_clean = np.where(if_clean == False)  # the column index in x that don't contain -999
        index_dirty = np.where(if_clean == True)   # the column index in x that do contain -999
        correlation = np.zeros([x_dirty.shape[1],x_clean.shape[1]])   # initialize the matrix storing the correlation coef
        for i in range(x_dirty.shape[1]):
            for j in range(x_clean.shape[1]):      # get rid the dirty number then compute the correlation between the dirty column and the clean column 
                correlation[i,j] = np.corrcoef(x_dirty[:,i][~x_dirty.mask[:,i]], x_clean[:,j][~x_dirty.mask[:,i]])[1,0] 
        most_correlated = np.abs(correlation) > 0.5         # get the mask where the correlation is bigger than the threshold
        print np.sum(most_correlated)
        if np.sum(most_correlated) == 0.:
            for i in range(x_dirty.shape[1]):
                x_masked[:,index_dirty[0][i]][x_dirty.mask[:,i]] = np.mean(x_masked[:,index_dirty[0][i]][~x_dirty.mask[:,i]])
            break
        num_correlated_ = np.sum(most_correlated, axis = 1) # for each row, compute the total number of the big correlation
        index_target = np.argmax(num_correlated_)           # pick the row that has more correlation votes. One row here corresponds one column in x_dirty
        index_correlated = np.where(most_correlated[index_target] > 0.5)  # from the x_clean, pick the index of the column that votes the correlation for the dirty x.
        x_clean_correlated = x_clean.T[most_correlated[index_target]].T   # build the matrix of the most correlated clean x, extracted from x_clean
        tx = build_poly(x_clean_correlated[~x_dirty.mask[:,index_target]].data, 4) # from the correlated x clean matrix, extract the row that correspond the invalid in the dirty x, and get ready for the regression
        tx_to_estimate = build_poly(x_clean_correlated[x_dirty.mask[:,index_target]].data, 4) 
        y = x_dirty[:,index_target][~x_dirty.mask[:,index_target]].data # in the target dirty x, extract the row invalid. 
        [mse, w] = least_squares(y, tx)                                  # Regression between the valid dirty x and the valid clean x. Get the model w
        y_estimated = np.dot(tx_to_estimate,w) 
        x_masked[:,index_dirty[0][index_target]][x_dirty.mask[:,index_target]] = y_estimated
        x = x_masked.data
    
    return x_masked.data

   
def replace_invalid_mean(x,value_invalid):
    
    x_masked = ma.masked_equal(x,value_invalid)
    for i in range(x.shape[1]):
        x_masked[:,i][x_masked.mask[:,i]] = np.mean(x_masked[:,i][~x_masked.mask[:,i]])
    
    return x_masked.data

def replace_invalid_crude(x,value_invalid):
    
    x_masked = ma.masked_equal(x,value_invalid)         # a mask_matrix with the mask indicating where = -999, if = True
    if_clean = np.sum(x_masked.mask, axis = 0)
    if_clean = ma.masked_greater_equal(if_clean, 1).mask    # to indicate that which column is clean. If = 1 then is not clean
    x_clean = (x_masked.T[~if_clean]).T        # The column which don't contain the -999, extracted from the x 
    x_dirty = (x_masked.T[if_clean]).T         # The column which do contain the -999, extracted from the x 
    index_dirty = np.where(if_clean == True)   # the column index in x that do contain -999
    for i in range(x_dirty.shape[1]):
        tx = build_poly(x_clean[~x_dirty.mask[:,i]].data, 3) # from the correlated x clean matrix, extract the row that correspond the invalid in the dirty x, and get ready for the regression
        tx_to_estimate = build_poly(x_clean[x_dirty.mask[:,i]].data, 3) 
        y = x_dirty[:,i][~x_dirty.mask[:,i]].data # in the target dirty x, extract the row invalid. 
        [mse, w] = least_squares(y, tx)                                  # Regression between the valid dirty x and the valid clean x. Get the model w
        y_estimated = np.dot(tx_to_estimate,w) 
        x_masked[:,index_dirty[0][i]][x_dirty.mask[:,i]] = y_estimated
    return x_masked.data
                               