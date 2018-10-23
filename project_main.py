import numpy as np
from proj1_helpers import *
from project_func import *
from preprocessing import replace_invalid

## load the data
path = 'train.csv'
label_train, input_train, ids = load_csv_data(path, sub_sample=False)
path = 'test.csv'
label_test, input_test, ids_test = load_csv_data(path, sub_sample=False)

input_train_reduced = input_train[:,[1,2,6,11,12,24]]
input_test_reduced = input_test[:,[1,2,6,11,12,24]]

## data preprocessing
value_invalid = -999
x_valid = replace_invalid_crude(input_train,value_invalid)
x_test_valid = replace_invalid_crude(input_test,value_invalid)

##calculate correlation
correlation=np.zeros((30,30))
ar_n=[]
acc_s=np.zeros(30)
for i in range(30):
    smallnumber=0
    for x in range(30):
       corre=np.corrcoef(x_valid[:,i],x_valid[:,x])
       correlation[i,x]=abs(corre[0,1])
       if correlation[i,x]<0.2: #account the time that correlation<0.2
           smallnumber=smallnumber+1
    if smallnumber>25:
        ar_n=np.append(ar_n,i)
    acc_s[i]=smallnumber  
#delete the line which is super low correlation(less than 0.2 and the time more than 25) with the others
ar_n=ar_n[::-1]
for i in range(np.size(ar_n)):
    x_test_valid=np.delete(x_test_valid,ar_n[i],1)
    x_valid=np.delete(x_valid,ar_n[i],1)
    
## logistic regression and prediction
alpha = 0.01
max_iters = 600
poly_degree = 3 #2 and 3 are the best
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
create_csv_submission(ids_test, y_pre, 'Submission_2210_7.csv')
