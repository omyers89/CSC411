import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *
_LR = 0
_WR = 1
_NOI = 2


def run_logistic_regression(hyps, small=False, pen=False, to_print=False):
    if small:
        train_inputs, train_targets = load_train_small()
    else:
        train_inputs, train_targets = load_train()

    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': hyps[_LR],
                    'weight_regularization': hyps[_WR],
                    'num_iterations': hyps[_NOI]
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.random(M+1) - 0.5

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    # run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    max_correct_train = (0, 0)
    max_correct_valid = (0, 0)
    min_cE_valid = (1000000, 0)
    CE_vec_valid = np.zeros(shape=(hyperparameters['num_iterations'], ))
    CE_vec_train = np.zeros(shape=(hyperparameters['num_iterations'], ))
    for t in xrange(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        if pen:
            f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        else:
            f, df, predictions = logistic(weights, train_inputs, train_targets, [])


        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        df.shape = (M+1, )
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        CE_vec_valid[t] = cross_entropy_valid
        CE_vec_train[t] = cross_entropy_train

        if frac_correct_train > max_correct_train[0]:
            max_correct_train = frac_correct_train, t
        if frac_correct_valid > max_correct_valid[0]:
            max_correct_valid = frac_correct_valid, t
        if cross_entropy_valid < min_cE_valid[0]:
            min_cE_valid = cross_entropy_valid, t

        # print some stats
        if to_print:
            stat_msg1 = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
            stat_msg1 += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
            print stat_msg1.format(t+1,
                                  float(f / N),
                                  float(cross_entropy_train),
                                  float(frac_correct_train*100),
                                  float(cross_entropy_valid),
                                  float(frac_correct_valid*100))
        if t == hyperparameters['num_iterations'] - 1:
            return (float(f / N),
                    float(cross_entropy_train),
                    float(frac_correct_train*100),
                    float(cross_entropy_valid),
                    float(frac_correct_valid*100),
                    max_correct_train,
                    max_correct_valid,
                    min_cE_valid,
                    CE_vec_train, CE_vec_valid,
                    )


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

    diff = check_grad(logistic_pen,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

def run_logistics(l = 0.17, IT = 500, LAMBDA = 0.01,smalls=False, pens=False):
    res = run_logistic_regression([l, LAMBDA, IT], smalls, pens)
    return res

