from kmeans import *
from nn import *
import sys
import matplotlib.pyplot as plt

plt.ion()


def mogEM(x, K, iters, minVary=0.01, randConst=1, isKNN=False):
    """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
    N, T = x.shape

    # Initialize the parameters
    p = randConst + np.random.rand(K, 1)
    p = p / np.sum(p)
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)

    # Change the initializaiton with Kmeans here
    # --------------------  Add your code here --------------------
    if isKNN:
        mu = KMeans(x, K, 5)
    else:
        mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)

    # ------------------------------------------------------------
    vary = vr * np.ones((1, K)) * 2
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logProbX = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in xrange(iters):
        # Do the E step
        respTot = np.zeros((K, 1))
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        logProb = np.zeros((1, T))
        ivary = 1 / vary
        logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
        logPcAndx = np.zeros((K, T))
        for k in xrange(K):
            dis = (x - mu[:, k].reshape(-1, 1)) ** 2
            logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1)
        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logProb = np.log(Px) + mx
        logProbX[i] = np.sum(logProb)

        print 'Iter %d logProb %.5f' % (i, logProbX[i])

        # Plot log prob of data
        plt.figure(1);
        plt.clf()
        plt.plot(np.arange(i), logProbX[:i], 'r-')
        plt.title('Log-probability of data versus # iterations of EM')
        plt.xlabel('Iterations of EM')
        plt.ylabel('log P(D)');
        plt.draw()

        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        for k in xrange(K):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)
            respDist[:, k] = np.mean((x - mu[:, k].reshape(-1, 1)) ** 2 * PcGivenx[k, :].reshape(1, -1), axis=1)

        # Do the M step
        p = respTot
        mu = respX / respTot.T
        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    return p, mu, vary, logProbX


def mogLogProb(p, mu, vary, x):
    """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
    K = p.shape[0]
    N, T = x.shape
    ivary = 1 / vary
    logProb = np.zeros(T)
    for t in xrange(T):
        # Compute log P(c)p(x|c) and then log p(x)
        logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
                    - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
                    - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu) ** 2, axis=0).reshape(-1, 1)

        mx = np.max(logPcAndx, axis=0)
        logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
    return logProb


def q2():
    randConst2 = 5
    randConst3 = 1
    minVary = 0.01
    k = 2
    iter2 = 20
    iter3 = 30
    inputs_train_second, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz',
                                                                                                       True,
                                                                                                       False)
    inputs_train_third, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz',
                                                                                                      False,
                                                                                                      True)
    p2, mu2, vary2, lp2 = mogEM(inputs_train_second, k, iter2, minVary, randConst2, False)
    raw_input('Press Enter to continue.')
    p3, mu3, vary3, lp3 = mogEM(inputs_train_third, k, iter3, minVary, randConst3, False)
    raw_input('Press Enter to continue.')
    ShowMeans(mu2)
    ShowMeans(mu3)
    ShowMeans(vary2)
    ShowMeans(vary3)
    print "log probability for the 2 number is: {0}".format(lp2)
    print "log probability for the 3 number is: {0}".format(lp3)
    print "Probability of clusters for the 2 number is: {0}".format(p2)
    print "Probability of clusters for the 3 number is: {0}".format(p3)


def q3():
    iters = 10
    minVary = 0.01
    randConst = 1
    k = 20
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
    # Train a MoG model with 20 components on all 600 training
    # vectors, with both original initialization and kmeans initialization.
    # ------------------- Add your code here ---------------------
    p, mu, vary, logProbX = mogEM(inputs_train, k, iters, minVary, randConst, False)
    raw_input('Press Enter to continue. - kMeans is next')
    pknn, muknn, varyknn, logProbXknn = mogEM(inputs_train, k, iters, minVary, randConst, True)

    raw_input('Press Enter to continue.')


def q4():
    iters = 10
    minVary = 0.01
    errorTrain = np.zeros(4)
    errorTest = np.zeros(4)
    randConst = 1
    errorValidation = np.zeros(4)
    print(errorTrain)
    numComponents = np.array([2, 5, 15, 25])
    T = numComponents.shape[0]
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
    train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
    train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)

    for t in xrange(T):
        K = numComponents[t]
        # Train a MoG model with K components for digit 2
        # -------------------- Add your code here --------------------------------

        p2, mu2, vary2, lp2 = mogEM(train2, K, iters, minVary, randConst, True)

        # Train a MoG model with K components for digit 3
        # -------------------- Add your code here --------------------------------

        p3, mu3, vary3, lp3 = mogEM(train3, K, iters, minVary, randConst, True)

        # Caculate the probability P(d=1|x) and P(d=2|x),
        # classify examples, and compute error rate
        # Hints: you may want to use mogLogProb function
        # -------------------- Add your code here --------------------------------

        errorTrain[t] = calc_error(p2, mu2, vary2, p3, mu3, vary3, inputs_train, target_train)
        errorTest[t] = calc_error(p2, mu2, vary2, p3, mu3, vary3, inputs_test, target_test)
        errorValidation[t] = calc_error(p2, mu2, vary2, p3, mu3, vary3, inputs_valid, target_valid)

    # Plot the error rate
    plt.clf()
    # -------------------- Add your code here --------------------------------
    plt.plot(numComponents, errorTrain, color="blue", label='Traning Set')
    plt.plot(numComponents, errorValidation, color="red", label='Validation Set')
    plt.plot(numComponents, errorTest, color="green", label='Test Set')
    plt.xlabel('Number of Gaussians')
    plt.ylabel('Mean Classification Error Rate')
    plt.legend()
    plt.axes()
    plt.draw()
    raw_input('Press Enter to continue.')


def calc_error(p2, mu2, vary2, p3, mu3, vary3, input, targets):
    lp2 = mogLogProb(p2, mu2, vary2, input)
    lp3 = mogLogProb(p3, mu3, vary3, input)

    # Bayes rule
    denominator = np.exp(lp2) + np.exp(lp3)
    p2given = np.exp(lp2) / denominator
    p3given = np.exp(lp3) / denominator

    corrects = []

    for i in xrange(len(targets.T)):
        corrects.append((p3given[i] > p2given[i]) == targets.T[i])

    return 1 - float(sum(corrects)) / float(len(corrects))


def q5():
    # Choose the best mixture of Gaussian classifier you have, compare this
    # mixture of Gaussian classifier with the neural network you implemented in
    # the last assignment.

    # Train neural network classifier. The number of hidden units should be
    # equal to the number of mixture components.

    # Show the error rate comparison.
    # -------------------- Add your code here --------------------------------

    num_hiddens = 30
    eps = 0.5
    momentum = 0.9
    num_epochs = 501
    W1, W2, b1, b2, train_error, valid_error = TrainNN(num_hiddens, eps, momentum, num_epochs)
    ShowMeans(W1[:,:6])
    ShowMeans(W1[:,-6:])

    raw_input('Press Enter to continue.')


if __name__ == '__main__':
    # q2()
    # q3()
    #q4()
    q5()
