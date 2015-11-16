from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def mogEM(x, K, iters, ind, minVary=0):
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
  randConst = 1
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  if ind ==0:
    mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  else:
    mu = KMeans(x,K,iters)
  #------------------------------------------------------------  
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
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
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
    plt.grid('on')
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

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
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb


def q2():
  iters = 10
  minVary = 0.01
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  p2, mu2, vary2, logProbX2 = mogEM(train2, 2, iters, 0, minVary = 0.01) # keep the original init
  p3, mu3, vary3, logProbX3 = mogEM(train3, 2, iters, 0, minVary = 0.01)
  ShowMeans(mu2)
  ShowMeans(vary2)
  raw_input('Press to see plot for digits 3')
  ShowMeans(mu3)
  ShowMeans(vary3)

def q3():
  iters = 10
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  p_o, mu_o, vary_o, logProbX_o = mogEM(inputs_train, 20, iters, 0, minVary = 0.01) # orignal init
  p_km, mu_km, vary_km, logProbX_km = mogEM(inputs_train, 20, iters, 1, minVary = 0.01) # kmean init
  #raw_input('Press to see plot for kmean initialization')
  raw_input('Press Enter to continue.')
def q4():
  iters = 10
  minVary = 0.01
  numComponents = np.arange(1,28,2)#np.array([2, 5, 15, 25])
  T = numComponents.shape[0] 
  errorTrain = np.zeros(T)
  errorTest = np.zeros(T)
  errorValidation = np.zeros(T)
  print(errorTrain) 
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  
  for t in xrange(T): 
    K = numComponents[t]
    # Train a MoG model with K components for digit 2
    #-------------------- Add your code here --------------------------------
    p2, mu2, vary2, logProbX2 = mogEM(train2, K, iters, 1, minVary=0.01)
    
    # Train a MoG model with K components for digit 3
    #-------------------- Add your code here --------------------------------
    p3, mu3, vary3, logProbX3 = mogEM(train3, K, iters, 1, minVary=0.01)
    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    # Hints: you may want to use mogLogProb function
    #-------------------- Add your code here --------------------------------
    p2_lh_train = np.exp(mogLogProb(p2, mu2, vary2, inputs_train))
    p3_lh_train = np.exp(mogLogProb(p3, mu3, vary3, inputs_train))

    p2_lh_valid = np.exp(mogLogProb(p2, mu2, vary2, inputs_valid))
    p3_lh_valid= np.exp(mogLogProb(p3, mu3, vary3, inputs_valid))
    
    p2_lh_test = np.exp(mogLogProb(p2, mu2, vary2, inputs_test))
    p3_lh_test = np.exp(mogLogProb(p3, mu3, vary3, inputs_test))
  
    p2_post_tr = p2_lh_train*p2
    p3_post_tr = p3_lh_train*p3
    p2_post_val = p2_lh_valid*p2
    p3_post_val = p3_lh_valid*p3
    p2_post_test= p2_lh_test*p2
    p3_post_test= p3_lh_test*p3

    errorTrain[t] = helperf(target_train, p2_post_tr, p3_post_tr, K)
    errorValidation[t] = helperf(target_valid, p2_post_val, p3_post_val, K)
    errorTest[t] = helperf(target_test, p2_post_test, p3_post_test,K)

  # Plot the error rate
  plt.clf()
  #-------------------- Add your code here --------------------------------
  plt.plot(numComponents, errorTrain,'o')
  plt.plot(numComponents, errorValidation,'o')
  plt.plot(numComponents, errorTest,'o')
  f1,=plt.plot(numComponents, errorTrain)
  f2,=plt.plot(numComponents, errorValidation)
  f3,=plt.plot(numComponents, errorTest)
  plt.legend([f1,f2,f3],['Train','Validation','Test'],loc='best')
  plt.xlabel('K')
  plt.ylabel('Correct Percentage')
  #plt.ylim(ymax=1.005)
  plt.grid('on')
  plt.draw()
  raw_input('Press Enter to continue.')

def helperf(target_data, p2_post, p3_post, K):
  '''return correct percentage'''
  p_stack = np.append(p2_post, p3_post, axis=0)
  result = np.argmax(p_stack, axis=0) + 1
  guess = np.zeros(target_data.shape[1])
  for i in range(len(result)):
    if result[i]<= K:
      guess[i] = 0
    else:
      guess[i] = 1
  n_correct = np.count_nonzero(np.equal(guess, target_data[0]))
  return float(n_correct)/target_data.shape[1]

def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.

  # Show the error rate comparison.
  #-------------------- Add your code here --------------------------------

  raw_input('Press Enter to continue.')

if __name__ == '__main__':
  #q2()
  #q3()
  q4()
  #q5()