""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    N = data.shape[0]
    M = data.shape[1]
    z = np.zeros(shape=(N, ))
    ws = weights[:-1]
    b = weights[-1]
    for i in range(0, N):
        dd = data[i]
        dd.shape = (M, )
        d = np.dot(dd, ws)
        z[i] = d + b

    y = sigmoid(z)


    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function

    return y

def evaluate(targets, y, reg=1 ,pen=False):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of binary targets. Values should be either 0 or 1
        y       : N x 1 vector of probabilities. = p(x)
    Outputs:
        ce           : (scalar) Cross entropy.  CE(p, q) = E_p[-log q].  Here
                       we want to compute CE(targets, y).
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    N = targets.size
    tz = np.ones(shape=(N, 1)) - targets    # col vec
    pco = y                                 #row vec
    pcz = np.ones(shape=(N, )) - pco        #row
    left = np.dot(np.log(pcz), tz)
    right = np.dot(np.log(pco), targets)
    ce = -(left + right)

    correct = 0
    for f in range(targets.size):
        if np.abs(targets[f] - y[f]) < 0.5:
            correct += 1

    frac_correct = float(correct)/float(targets.size)

    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):

    N = data.shape[0]
    M = data.shape[1]
    z = np.zeros(shape=(N, ))
    ws = weights[:-1]
    b = weights[-1]
    for i in range(0, N):
        dd = data[i]
        dd.shape = (M, )
        d = np.dot(dd, ws)
        z[i] = d + b


    zg = (sigmoid(z))
    zg.shape = (N, 1)
    df = np.zeros(shape=(M+1,))
    for j in range(0, M):

        df[j] = np.dot(data[:, j], (zg - targets))

    aa = (1 - targets)
    b = np.dot(sigmoid(z), -np.exp(-z)) + np.dot(np.ones(N), aa)

    df[M] = b
    df.shape = (M+1, 1)
    y = logistic_predict(weights, data)
    f = evaluate(targets, y)[0]
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    f, tdf, y = logistic(weights, data, targets, hyperparameters)
    lam = hyperparameters['weight_regularization']
    M = data.shape[1]
    ws = np.power(weights[:-1], 2) / (2.0 / lam)
    pis = 1.0 / (np.sqrt(2 * np.pi / lam))
    pis *= M
    add2f = pis + np.sum(ws)
    f += add2f
    add2df = (weights * lam)
    add2df[M] = 0.0
    add2df.shape = (M+1, 1)
    df = tdf + add2df

    return f, df, y

    # Calculate negative log likelihood and its derivatives with respect to weights.
    # Also return the predictions.
    #
    # Note: N is the number of examples and
    #       M is the number of features per example.
    #
    # Inputs:
    #     weights:    (M+1) x 1 vector of weights, where the last element
    #                 corresponds to bias (intercepts).
    #     data:       N x M data matrix where each row corresponds
    #                 to one data point.
    #     targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
    #     hyperparameters: The hyperparameters dictionary.
    #
    # Outputs:
    #     f:             The sum of the loss over all data points. This is the objective that we want to minimize.
    #     df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    # """
    #
    # # TODO: Finish this function


