# __author__ = 'omrim'

from utils import *
import run_knn as knn
from datetime import datetime
import logistic_regression_template
import plot_digits
import nb
import matplotlib.pyplot as plt
_FOUR = 0
_NINE = 1




def run_test2_1(k,in_train, out_train, ins, outs):
    tstart = datetime.now()
    out_hyp = knn.run_knn(k, in_train, out_train, ins)
    tend = datetime.now()
    delta_time = tend - tstart
    total_time = int(delta_time.total_seconds() * 1000)
    tp9 = 0
    tp4 = 0
    fp9 = 0
    fp4 = 0
    pos = 0
    tot_valid = outs.size
    for v in outs:
        pos += v

    i = 0

    for h in out_hyp:
        if h == 1:
            if h == outs[i]:
                tp9 += 1
            else:
                fp9 += 1
        else:
            if h == outs[i]:
                tp4 += 1
            else:
                fp4 += 1
        i += 1

    precision9 = float(tp9) / float(tp9+fp9)
    precision4 = float(tp4) / float(tp4+fp4)
    recall9 = float(tp9) / float(pos)
    recall4 = float(tp4) / float(tot_valid - pos)
    accuracy = float(tp9 + tp4) / float(tot_valid)
    return precision4, recall4, precision9, \
        recall9, accuracy, total_time,



#run_test2_2()
#
def Q_21():
    train = load_train()
    in_train = train[0]
    out_train = train[1]
    valid = load_valid()
    in_valid = valid[0]
    out_valid = valid[1]
    test = load_test()
    in_test = test[0]
    out_test = test[1]
    vis_k = np.arange(1, 10, 2)
    vis_acc = np.zeros(shape=(5, ))
    vis_acct = np.zeros(shape=(5, ))

    for ck in range(5):
        results = run_test2_1(vis_k[ck],in_train, out_train, in_valid, out_valid)
        vis_acc[ck] = results[4] * 100
        print "for k = {}".format(ck)
        print "\t 9-Precision:           %5.3f%%" % (results[2] * 100)
        print "\t 9-Recall:              %5.3f%%" % (results[3] * 100)
        print "\t Accuracy:              %5.3f%%" % (results[4] * 100)

    fig = plt.plot(vis_k, vis_acc, 'ro', vis_k, vis_acc)
    plt.axis([0, 10, 70, 100])
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Classification rate on the validation set as function of k')
    plt.show()


def run_test2_2(l, IT, LAMBDA = 1,smalls=False, pens=False ):
    #run_logistics(l = 0.17, IT = 500, LAMBDA = 0.01,smalls=False, pens=False):

    res = logistic_regression_template.run_logistics(l,IT,LAMBDA,smalls, pens)
    print "learnigrate:{}".format(l)
    stat_msg = "TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
    stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
    stat_msg += "iteration at max frac train:{}  iteration at max frac valid:{}  \n iteration at min CE valid:{}"
    print stat_msg.format(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7])

    # res[8] is training CE, res[9] is valid CE
    fig_train = plt.plot(np.arange(0, IT), res[8], 'g')
    fig_valid = plt.plot(np.arange(0, IT), res[9], 'b')
    plt.axis([0, IT, 0, 150])
    plt.xlabel('Iteration')
    plt.ylabel('CE')
    gtitle = '''CE as function of the iter number on mnist_train, \n
                    Learning rate: {}, Number of iterations: IT \n green:=train, blue:=validation'''
    plt.title(gtitle.format(l,IT))
    plt.show()

def Q_22():
    print "****running test 2.2 l=0.17, IT=500***"
    run_test2_2(0.17, 500, 1)
    print "****running test 2.2 small l=0.02, IT=1000***"
    run_test2_2(0.02, 1000, 1)



def plot2_3(l,IT,smalls,finals):
    fig_train = plt.plot(finals[0], finals[1], '*', finals[0], finals[1],'g')
    fig_valid = plt.plot(finals[0], finals[3], '*', finals[0], finals[3], 'b')
    plt.axis([0, 1, 0, 50])
    plt.xlabel('LAMBDA - penalty parameter')
    plt.ylabel('CE')
    gtitle = '''CE as function of LAMBDA - penalty parameter on mnist_train, \n
                Learning rate: {}, Number of iterations: {} \n green:=train, blue:=validation '''
    plt.title(gtitle.format(l,IT))
    plt.show()
    #plt.close([fig_train, fig_valid])
    fig_trainc = plt.plot(finals[0], finals[2], '*', finals[0], finals[2], 'g')
    fig_validc = plt.plot(finals[0], finals[4], '*', finals[0], finals[4], 'b')
    plt.axis([0, 1, 60, 110])
    plt.xlabel('LAMBDA - penalty parameter')
    plt.ylabel('classification rate')
    gtitle = '''classification rate as function of LAMBDA - penalty parameter on mnist_train, \n
                Learning rate: {}, Number of iterations: {} \n green:=train, blue:=validation'''
    plt.title(gtitle.format(l,IT))
    plt.show()

def run_test2_3(l,IT,smalls):
    final_results = [[],[],[],[],[]]
    for lam in [0.001, 0.01, 0.1, 0.5, 0.8, 1.0]:
    #for lam in [0.001, 0.01, 0.02]:

        res = np.zeros(shape=(5, ))
        res[0] = lam
        for it in range(10):
            tres = logistic_regression_template.run_logistics(l,IT,lam,smalls, True)

            res[1] += tres[1]
            res[2] += tres[2]
            res[3] += tres[3]
            res[4] += tres[4]

        avg_res = res / 10.0
        avg_res[0] *= 10.0

        print  "for Lambda = {} the average results are: \n train CE:{},    train class err:{}," \
               "    valid CE:{},    valid calss err:{}".format(lam, avg_res[1], avg_res[2], avg_res[3],avg_res[4])

        for i in range(5):
            final_results[i].append(avg_res[i])
    plot2_3(l,IT,smalls,final_results)

def Q_23():
    print "**** runing test 2.3 l=0.17, IT=500 smalls=False **** "
    run_test2_3(0.17, 500, False)
    print "**** runing test 2.3 small l=0.02, IT=1000 smalls=True **** "
    run_test2_3(0.02, 1000, True)

def Q_24():
    nb.main()

#Q_21()
#Q_22()
#Q_23()
#Q_24()

