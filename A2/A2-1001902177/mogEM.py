from kmeans import *
import sys
import matplotlib.pyplot as plt

plt.ion()


def mogEM(x, K, iters, minVary=0.0, randConst=1.0, init_kmeans=False):
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
    # randConst = 1
    p = randConst + np.random.rand(K, 1)
    p = p / np.sum(p)
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)

    # Change the initializaiton with Kmeans here
    # --------------------  Add your code here --------------------
    if init_kmeans:
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
    iters = 10
    minVary = 0.01
    train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
    train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
    k = 2
    crand = 1
    p2, mu2, vary2, logprobX2 = mogEM(train2, k, iters, minVary, crand)
    print 'the mixing proportion for 2 model is:\n {}\n and the logProb is:\n{}'.format(p2, logprobX2)
    print "models for number 2"
    ShowMeans(mu2, "Mean model for digit 2")
    ShowMeans(vary2, "Var model for digit 2")

    print '********models for digit 3*********'
    p3, mu3, vary3, logprobX3 = mogEM(train3, k, iters, minVary, crand)
    print 'the mixing proportion for 3 model is:\n {}\n and the logProb is: {}\n'.format(p3, logprobX3)

    print "models for number 3"
    ShowMeans(mu3, "Mean model for digit 3")
    ShowMeans(vary3, "Var model for digit 3")


#
# results:
#  the mixing proportion for 2 model is:
#  [[ 0.46666666]
#  [ 0.53333334]]
#  and the logProb is:
# [-3890.79258653]
#
# the mixing proportion for 3 model is:
#  [[ 0.52986667]
#  [ 0.47013333]]
#  and the logProb is:
# [ 2294.22172789]


def q3():
    iters = 10
    minVary = 0.01
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
    # Train a MoG model with 20 components on all 600 training
    # vectors, with both original initialization and kmeans initialization.
    # ------------------- Add your code here ---------------------
    k = 20
    crand = 1
    p, mu, vary, logprobX = mogEM(inputs_train, k, iters, minVary, crand, False)
    print 'Regular the mixing proportion is:\n {}\n and the logProb is:\n{}'.format(p, logprobX[iters - 1])
    raw_input('Press Enter to continue.')
    pi, mui, varyi, logprobXi = mogEM(inputs_train, k, iters, minVary, crand, True)
    print 'Init with k-means the mixing proportion is:\n {}\n and the logProb is:\n{}'.format(pi, logprobXi[iters - 1])
    raw_input('Press Enter to continue.')


def error_rate(prob_d2, prob_d3, targets):
    errors = 0
    for i, t in enumerate(targets.T):
        if ((prob_d2[i] >= prob_d3[i]) and t == 1) or ((prob_d3[i] > prob_d2[i]) and t == 0):
            errors += 1

    return (float(errors) / float(targets.size)) * 100




def q4():
    iters = 10
    minVary = 0.01
    errorTrain = np.zeros(4)
    errorTest = np.zeros(4)
    errorValidation = np.zeros(4)
    print(errorTrain)
    numComponents = np.array([2, 5, 15, 25])
    T = numComponents.shape[0]
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
    train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
    train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
    crand = 1
    eTrain = np.zeros(T)
    eValid = np.zeros(T)
    eTest = np.zeros(T)
    for t in xrange(T):
        K = numComponents[t]
        # Train a MoG model with K components for digit 2
        # -------------------- Add your code here --------------------------------
        pi2, mui2, varyi2, logprobXi2 = mogEM(train2, K, iters, minVary, crand, True)

        # Train a MoG model with K components for digit 3
        # -------------------- Add your code here --------------------------------
        pi3, mui3, varyi3, logprobXi3 = mogEM(train3, K, iters, minVary, crand, True)

        # Caculate the probability P(d=1|x) and P(d=2|x),
        # classify examples, and compute error rate
        # Hints: you may want to use mogLogProb function
        # -------------------- Add your code here --------------------------------
        ln_p2_train = np.exp(mogLogProb(pi2, mui2, varyi2, inputs_train))
        ln_p2_valid = np.exp(mogLogProb(pi2, mui2, varyi2, inputs_valid))
        ln_p2_test = np.exp(mogLogProb(pi2, mui2, varyi2, inputs_test))

        ln_p3_train = np.exp(mogLogProb(pi3, mui3, varyi3, inputs_train))
        ln_p3_valid = np.exp(mogLogProb(pi3, mui3, varyi3, inputs_valid))
        ln_p3_test = np.exp(mogLogProb(pi3, mui3, varyi3, inputs_test))

        eTrain[t] = error_rate(ln_p2_train, ln_p3_train, target_train)
        eValid[t] = error_rate(ln_p2_valid, ln_p3_valid, target_valid)
        eTest[t] = error_rate(ln_p2_test, ln_p3_test, target_test)





    # Plot the error rate
    plt.clf()
    # -------------------- Add your code here --------------------------------
    plt.plot(numComponents, eTrain, label='Train set')
    plt.plot(numComponents, eValid, label='Valid set')
    plt.plot(numComponents, eTest, label='Test set')
    plt.xlabel('Number of components')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.draw()
    raw_input('Press Enter to continue.')


def q5():
    # Choose the best mixture of Gaussian classifier you have, compare this
    # mixture of Gaussian classifier with the neural network you implemented in
    # the last assignment.

    # Train neural network classifier. The number of hidden units should be
    # equal to the number of mixture components.

    # Show the error rate comparison.
    # -------------------- Add your code here --------------------------------

    raw_input('Press Enter to continue.')


if __name__ == '__main__':
    x = raw_input("choose func")
    options = {2: q2,
               3: q3,
               4: q4,
               5: q5}
    options[int(x)]()
