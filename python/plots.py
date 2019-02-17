import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import *
from sklearn import linear_model

def plot():
    logs12=pickle.load(open("{}/logs_full_folder_complete_MGU_lasts_64.log".format('.'),'rb'))

    logs34=pickle.load(open("{}/logs_full_folder_complete_MGU_lasts_64_patch.log".format('.'),'rb'))
    logs34 = np.array([np.array(xi) for xi in logs34])
    logs34 = np.array(logs34)
    logs34 = np.reshape(logs34, (1, -1, 2, 5))
    print(logs34.shape)
    logs12 = np.array([np.array(xi) for xi in logs12])
    #logs3 = np.array([np.array(xi) for xi in logs3])
    #logs5 = np.array([np.array(xi) for xi in logs5])
    logs12=np.array(logs12)
    #logs3 = np.array(logs3)
    #logs5 = np.array(logs5)
    #print(logs12.shape)
    logs12=np.reshape(logs12, (10,-1,2,5))
    #print(logs12[4][0])
    logs12[3]=logs34[0]
    print(logs12.shape)
    #print(logs12[4][0])
    #print(logs12.shape)
    plt.figure(figsize=(10, 6))
    plt.xlabel('drum part')
    plt.ylabel('validation loss')
    linr=linear_model.LinearRegression()
    #print(len(logs12))
    lines=[]
    colors=['r:','g:','b--', 'k--','m','c','g', 'r','b','y']
    for i in range(len(logs12)-2):
        print(
              '%0i' % sum((logs12[i][:, 1, 4]) - (logs12[i][:, 0, 4])),
              '%0.4f' %np.mean(logs12[i][:, 1, 0]),
              '%0.4f' %np.mean(logs12[i][:, 1, 1]),
              '%0.4f' %np.mean(logs12[i][:,1,2]),
              '%0.4f' %np.mean(logs12[i][:, 1, 3]),

              )
        print(
              )
        n=2
        for j in range(n):

            start=int(len(logs12[i][:,1,2])/n*j)
            stop = int((len(logs12[i][:, 1, 2]) / n)* (j+1))
            X_train=range(start,stop)
            y_train=logs12[i][start:stop,1,2]
            X_train=np.reshape(X_train, (len(X_train),1))
            linr.fit(X_train,y_train)
            #X_test=range(0,100)
            #X_test=np.reshape(X_test,(len(X_test),1))
            #lines.append(np.clip(linr.predict(X_test), 0, None))
            #plt.plot(lines[-1],colors[i], label='_nolegend_')
            if j==0:
                lines.append(list(linr.predict(X_train)))
            else:
                lines[-1].extend(list(linr.predict(X_train)))
        plt.plot(lines[-1],colors[i])


    # plt.ylim(ymax=1)
    #print(np.argmax(logs12[0][:,2]))
    n=1
    w=15
    #plt.plot(movingAverage(logs12[0][:,n,2],w), 'r:', )
    #####plt.vlines(np.argmin(logs12[0][:,2]),0,  np.min(logs12[0][:,2]),colors='g', linestyles='-')
    #plt.plot(movingAverage(logs12[1][:,n,2],w),'g:')
    ####plt.vlines(np.argmin(logs12[1][:, 2]), 0, np.min(logs12[1][:,2]), colors='b', linestyles='-')
#
    #plt.plot(movingAverage(logs12[2][:,n,2],w),'b--')
    ####plt.vlines(np.argmin(logs12[2][:, 2]), 0, np.min(logs12[2][:, 2]), colors='r', linestyles='-')
    #plt.plot(movingAverage(logs12[3][:,n,2],w),'k--')
    ####plt.vlines(np.argmin(logs12[3][:, 2]), 0, np.min(logs12[3][:, 2]), colors='y', linestyles='-')
#
    ##plt.plot(movingAverage(logs12[5][:, 2], w), 'c')
    #plt.plot(movingAverage(logs12[5][:,n, 2], w), 'c')
    #plt.plot(movingAverage(logs12[4][:, n, 2], w), 'm')
    #plt.plot(movingAverage(logs12[6][:,n, 2], w), 'g')
#
    #plt.plot(movingAverage(logs12[7][:,n, 2], w), 'r')


    #plt.plot(movingAverage(logs12[8][:, n, 2], w), 'b')
    #plt.plot(movingAverage(logs12[9][:, n, 2], w), 'y')
    #plt.vlines(np.argmin(logs12[4][:, 2]), 0, np.min(logs12[4][:, 2]), colors='m', linestyles='-')
   # plt.plot(logs12[0][:, 2] - logs12[0][:, 0], 'g:')
   # plt.plot(logs12[1][:, 2] - logs12[1][:, 0], 'b:')
    #plt.plot(logs3[0][:, 2] - logs3[0][:, 0], 'r:')
   # plt.plot(logs12[3][:, 2] - logs12[3][:, 0], 'y:')
    #plt.plot(logs5[0][:, 2] - logs5[0][:, 0], 'm:')
    plt.hlines(0,0,130)
    plt.gca().legend((#'TDC_parallel_mgu', 'time_dist_conv_mgu','parallel_mgu','stacked_mgu','single_mgu','conv_mgu'
                      'TDC_Parallel_MGU',
                      'TDC_MGU',
                      'Parallel MGU',
                      'Stacked MGU',
                      'Single MGU',
                      'Conv MGU',

                      'Single GRU',
                      'Single LSTM',
                      ),loc='upper right')
    'TDC_parallel_mgu', 'time_dist_conv_mgu', 'parallel_mgu', 'stacked_mgu', 'single_mgu', 'conv_mgu',
    'single_gru', 'single_lstm', 'single_mgu_relu', 'single_mgu_elu'

    plt.show()

if __name__ == "__main__":
    plot()
