import numpy as np
from proj1_helpers import *
from project_func import *
from preprocessing import replace_invalid

## load the data
path = '../data/train.csv'
label_train, input_train, ids = load_csv_data(path, sub_sample=False)
path = '../data/test.csv'
label_test, input_test, ids_test = load_csv_data(path, sub_sample=False)

input_train_reduced = input_train[:,[1,2,6,11,12,24]]
input_test_reduced = input_test[:,[1,2,6,11,12,24]]

## data preprocessing
value_invalid = -999
x_valid = replace_invalid_crude(input_train,value_invalid)
x_test_valid = replace_invalid_crude(input_test,value_invalid)

## logistic regression and prediction
alpha = 0.01
max_iters = 600
poly_degree = 3
x_std = standardize(x_valid)
tx = build_poly(x_std, poly_degree)
y = label_train
initial_w = np.ones([tx.shape[1]])
u = (1+y)/2.0
likelihoods, ws = logistic_regression(u, tx, initial_w, max_iters, alpha)

u_estimated = sigmoid(tx.dot(ws[-1]))
u_estimated[u_estimated >= 0.5] = 1.
u_estimated[u_estimated < 0.5] = 0.
y_estimated = 2.0*u_estimated - 1
correct_rate = np.sum(np.abs(y_estimated - y))/2.0
correct_rate = 1 - correct_rate/y.shape

## Cross validation

## Prediction
x_test_std = standardize(x_test_valid)
tx_test = build_poly(x_test_std, poly_degree)
u_pre = sigmoid(tx_test.dot(ws[-1]))
u_pre[u_pre >= 0.5] = 1.
u_pre[u_pre < 0.5] = 0.
y_pre = 2.0*(u_pre)-1


## generate the submission 
create_csv_submission(ids_test, y_pre, 'Submission_2210_3.csv')

