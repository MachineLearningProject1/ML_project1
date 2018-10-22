## least squares GD

#import matplotlib.pyplot as plt

import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - tx.dot(w)
    grad_L = -np.dot(tx.T,e)/N
    return grad_L

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        # TODO: compute gradient and loss
        loss = compute_loss(y, tx, w)
        # TODO: update w by gradient
        w = w - gamma*compute_gradient(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

## least squares SGD
    
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    N = y.shape[0]
    e = y - tx.dot(w)
    batch_grad = -np.dot(tx.T,e)/N
    return batch_grad 


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    # TODO: implement stochastic gradient descent. 
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            w = w - gamma*compute_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(y, tx, w)
    ws.append(w)
    losses.append(loss)
    print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

## least squares
    
def least_squares(y, tx):
    """calculate the least squares solution."""
    # least squares: TODO
    N = y.shape[0]
    w_opt = np.dot(np.dot(np.linalg.inv(np.dot(tx.T,tx)),tx.T),y)
    e = y - np.dot(tx,w_opt)
    mse_opt = np.dot(e.T,e)/(2*N)
    return mse_opt, w_opt

    
## logistic regression
def sigmoid(x):
    return 1./(1+np.exp(-x))

def compute_log_likelihood(y, tx, w):
    
    l = np.sum(y*np.log(sigmoid(tx.dot(w)))+(1-y)*np.log(1-sigmoid(tx.dot(w))))
    
    return l
def compute_likelihood_gradient(y, tx, w):
    
    gradient = np.dot(tx.T,(y - sigmoid(tx.dot(w))))
    gradient = -gradient
    return gradient

def logistic_regression(y, tx, initial_w, max_iters, alpha):
# Define parameters to store w and loss
    ws = [initial_w]
    likelihoods = []
    w = initial_w
    for n_iter in range(max_iters):
        
        # TODO: compute gradient and loss
        likelihood = compute_log_likelihood(y, tx, w)
        # TODO: update w by gradient
        w = w - alpha*compute_likelihood_gradient(y, tx, w)
        # store w and loss
        ws.append(w)
        likelihoods.append(likelihood)
#        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
#              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return likelihoods, ws

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    poly_x = np.ones([x.shape[0]])
    for i in range(1,degree+1):
        poly_x = np.c_[poly_x, np.power(x,i)]
    poly_x = np.array(poly_x)
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    return poly_x
    # ***************************************************
    raise NotImplementedError
    
def standardize(x):
    ''' fill your code in here...
    '''
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data

    
    
    