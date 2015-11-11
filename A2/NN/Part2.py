


from nn import *
import run_knn as knn

def q2_3():
    num_hiddens = 10
    epss = [0.01, 0.1, 0.2, 0.5]
    momentums = [0.0, 0.5, 0.9]
    min_crossEntropy = float('inf')
    min_classError = 100.0
    num_epochs = 100
    bestEps_CE = 0
    bestMomentum_CE = 0
    bestEps_Err = 0
    bestMomentum_Err = 0
    for eps in epss:
        for momentum in momentums:
            W1, W2, b1, b2, train_error, valid_error,train_MCE_arr, valid_MCE_arr= TrainNN(num_hiddens, eps, momentum, num_epochs)
            # DisplayErrorPlot(train_error, valid_error, "Cross entropy")
            # DisplayErrorPlot(train_MCE_arr, valid_MCE_arr, "Mean classification error")
            temp_min_CE = min(valid_error)
            temp_min_Err = min(valid_MCE_arr)
            if (temp_min_CE < min_crossEntropy):
                min_crossEntropy = temp_min_CE
                bestEps_CE = eps
                bestMomentum_CE = momentum
            if (temp_min_Err < min_classError):
                min_classError = temp_min_Err
                bestEps_Err = eps
                bestMomentum_Err = momentum
    print "the min crossEntropy on the validation set is: {}, achieved for eps {}, and momentum {} ".format(min_crossEntropy,
                                                                                      bestEps_CE, bestMomentum_CE )
    print "the min ClassError on the validation set is: {}, achieved for eps {}, and momentum {} ".format(min_classError,
                                                                                      bestEps_Err, bestMomentum_Err )


def q2_4():
    num_hiddenss = [2,5,10]
    eps = 0.02
    momentum = 0.5
    min_crossEntropy = float('inf')
    min_classError = 100.0
    num_epochs = 100
    best_num_hiden_CE = 0
    best_num_hiden_Err = 0
    for num_hiddens in num_hiddenss:
        W1, W2, b1, b2, train_error, valid_error,train_MCE_arr, valid_MCE_arr= TrainNN(num_hiddens, eps, momentum, num_epochs)
        # DisplayErrorPlot(train_error, valid_error, "Cross entropy")
        # DisplayErrorPlot(train_MCE_arr, valid_MCE_arr, "Mean classification error")
        temp_min_CE = min(valid_error)
        temp_min_Err = min(valid_MCE_arr)
        if (temp_min_CE < min_crossEntropy):
            min_crossEntropy = temp_min_CE
            best_num_hiden_CE = num_hiddens

        if (temp_min_Err < min_classError):
            min_classError = temp_min_Err
            best_num_hiden_Err = num_hiddens

    print "the min crossEntropy on the validation set is: {}, achieved for {} hidden layers ".format(min_crossEntropy,
                                                                                      best_num_hiden_CE )
    print "the min ClassError on the validation set is: {}%, achieved for {} hidden layers".format(min_classError,
                                                                                      best_num_hiden_Err )


def q2_5():
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
    k = 7   #found out that works the best on ass 1
    out_hyp = knn.run_knn(k, inputs_train.T, target_train.T, inputs_valid.T)
    target_valid.shape = out_hyp.shape
    errors = 0
    for i, o in enumerate(out_hyp):
        errors += abs(o - target_valid[i])

    class_err = (float(errors) / float(target_valid.size)) * 100
    print ("the class error for K-NN is :{}%".format(class_err))



# q2_3()
q2_4()
q2_5()

