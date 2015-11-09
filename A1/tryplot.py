# __author__ = 'omrim'

import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *



def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)
    # print data[7]
    # print weights[:-1]
    # print weights[-1]
    # print np.dot(data[7], weights[:-1]) + weights[-1]

    N = data.shape[0]
    M = data.shape[1]
    z = np.zeros(N)
    ws = weights[:-1]
    b = weights[-1]
    dd = data[7]
    dd.shape = (10,)
    #print dd
    d = np.dot(dd, ws)
    d += b

    print d
    # diff = check_grad(logistic,      # function to check
    #                   weights,
    #                   0.001,         # perturbation
    #                   data,
    #                   targets,
    #                   hyperparameters)
    #
    # print "diff =", diff


run_check_grad([])



