
import matplotlib.cm as cmaps
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import *
from sklearn import linear_model

def plot():

    logs12=pickle.load(open("{}/logs_full_folder_complete_MGU_lasts_16.log".format('.'),'rb'))
    537.41
    logs34=pickle.load(open("{}/logs_full_folder_complete_MGU_lasts_64_tcn.log".format('.'),'rb'))
    logs34 = np.array([np.array(xi) for xi in logs34])
    logs34 = np.array(logs34)
    logs34 = np.reshape(logs34, (1, -1, 2, 5))
    #print(logs34.shape)
    logs12 = np.array([np.array(xi) for xi in logs12])
    #logs3 = np.array([np.array(xi) for xi in logs3])
    #logs5 = np.array([np.array(xi) for xi in logs5])
    logs12=np.array(logs12)
    #logs3 = np.array(logs3)
    #logs5 = np.array(logs5)
    print(logs12.shape)
    logs12=np.reshape(logs12, (8,-1,2,5))
    #print(logs12[4][0])
    logs12=np.concatenate((logs12,logs34))
    print(logs12.shape)
    #print(logs12[4][0])
    #print(logs12.shape)
    plt.figure(figsize=(10, 6))
    plt.xlabel('drum part')
    plt.ylabel('validation loss')
    linr=linear_model.LinearRegression()
    #print(len(logs12))
    lines=[]
    colors=['r:','g:','b--', 'k--','m','c','g', 'r','k','y']
    for i in range(len(logs12)):
        print(
              '%0i' % sum((logs12[i][:, 1, 4]) - (logs12[i][:, 0, 4])),
              '%0.4f' %np.mean(logs12[i][:, 1, 0]),
              '%0.4f' %np.mean(logs12[i][:, 1, 1]),
              '%0.4f' %np.mean(logs12[i][:,1,2]),
              '%0.4f' %np.mean(logs12[i][:, 1, 3]),

              )
        print(
              )
        n=1
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
            #plt.plot(lines[-1],colors[i])#


    # plt.ylim(ymax=1)
    #print(np.argmax(logs12[0][:,2]))
    n=1
    w=1
    plt.plot(movingAverage(logs12[0][:,n,2],w), 'r:', )
    ####plt.vlines(np.argmin(logs12[0][:,2]),0,  np.min(logs12[0][:,2]),colors='g', linestyles='-')
    plt.plot(movingAverage(logs12[1][:,n,2],w),'g:')
    ###plt.vlines(np.argmin(logs12[1][:, 2]), 0, np.min(logs12[1][:,2]), colors='b', linestyles='-')

    plt.plot(movingAverage(logs12[2][:,n,2],w),'b--')
    ###plt.vlines(np.argmin(logs12[2][:, 2]), 0, np.min(logs12[2][:, 2]), colors='r', linestyles='-')
    plt.plot(movingAverage(logs12[3][:,n,2],w),'k--')
    ###plt.vlines(np.argmin(logs12[3][:, 2]), 0, np.min(logs12[3][:, 2]), colors='y', linestyles='-')

    #plt.plot(movingAverage(logs12[5][:, 2], w), 'c')
    plt.plot(movingAverage(logs12[5][:,n, 2], w), 'c')
    plt.plot(movingAverage(logs12[4][:, n, 2], w), 'm')
    plt.plot(movingAverage(logs12[6][:,n, 2], w), 'g')

    plt.plot(movingAverage(logs12[7][:,n, 2], w), 'r')
    plt.plot(movingAverage(logs12[8][:, n, 2], w), 'k')


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
#
#


##TO USE NEXT TWO FUNCTIONS UNCOMMENT matplotlib IMPORTS; CONFLICTS WITH KIVY
def showEnvelope(env, legend=None, labels=None):
    """
    Plots an envelope
    i.e. an onset envelope or some other 2d data
    :param env: the data to plot
    :return: None
    """
    if 0 > 1:
        f, axarr = plt.subplots(len(env), 1, sharex=True, sharey=True)
        #for i in range(len(env)):
        #    axarr[i].plot(env[i][0], label='NMF')
        #    axarr[i].vlines(env[i][1], 0, 1, color='r', alpha=0.9, linestyle='--', label='Onsets')
        #    axarr[i].get_xaxis().set_visible(False)
        #    axarr[i].get_yaxis().set_ticks([])
        #f.subplots_adjust(hspace=0.1)
        axarr[0].set(ylabel='precision')
        axarr[1].set(ylabel='recall')
        axarr[2].set(ylabel='f-score')
        axarr[len(env) - 1].get_xaxis().set_visible(True)
        plt.savefig("NMFD_test_iterations.png")
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 6))

        # plt.ylim(ymax=1)
        plt.plot(env)
        # plt.plot(env[0], label='Onset envelope')
        #
        # #plt.hlines(env[2],0,8000, color='r', linestyle=':')
        # plt.hlines(env[2], 0,500, color='r', alpha=0.8, label='threshold', linestyles='-')
        # plt.hlines(env[3], 0, 500, color='k', alpha=0.8, label='highest optimal threshold', linestyles='--')
        # plt.hlines(env[1], 0, 500, color='g', alpha=0.8, label='lowest optimal threshold', linestyles=':')
        #
        #plt.xticks(np.geomspace(1, 50, 5).astype(int), np.geomspace(1, 50, 5).astype(int),1)
        plt.gca().legend(('precision','recall','f-score'), loc='right')
        plt.xlabel(labels[0], fontsize=12)
        plt.ylabel(labels[1], fontsize=12)
        plt.show()
        #ax = plt.subplot(111)
        #if legend != None:
        #    box = ax.get_position()
        #    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #    plt.gca().legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
        #    plt.gca().legend(legend, loc='right')
        #if labels != None:
        #    # my_xticks=[str(x)[:4] for x in np.linspace(0.2,0.4,int(env.shape[0]/2))]
        #    # my_xticks = np.arange(0, env.shape[0], .01)
        #    # plt.xticks(np.arange(1, env.shape[0], 1), range(1,20))
#
        #    plt.xlabel(labels[0], fontsize=12)
        #    plt.ylabel(labels[1], fontsize=12)
#
        # plt.savefig("nadam_lr.png")
        # plt.tight_layout()


def showFFT(env, ticks=None, stft=False):
    """
    Plots a spectrogram

    :param env: the data to plot
    :return: None
    """
    if 0 > 1:

        f, axarr = plt.subplots(1, len(env), sharex=True, sharey=True)

        for i in range(len(env)):
            axarr[i].imshow(env[i], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))

            # axarr[i].get_xaxis().set_visible(False)
        # axarr[1].imshow(env[1], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
        # axarr[1].set(xlabel='Snare frames')
        # axarr[2].imshow(env[2], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
        # axarr[2].set(xlabel='Closed Hi-Hat frames')
        f.subplots_adjust(wspace=0.03)
        axarr[0].set(ylabel='STFT bin')
        f.text(0.5, 0.02, 'k', ha='center', va='center')

        plt.savefig("Templates3.png")

    if ticks != None:
        top = len(ticks)
        plt.imshow(np.flipud(env), aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
        my_yticks = ticks
        plt.xlabel('frame', fontsize=12)
        plt.ylabel('tempo', fontsize=12)
        plt.yticks(np.arange(0, top, 10), np.rint(np.fliplr([ticks[0:top:10], ]))[0])

    else:
        plt.figure(figsize=(10, 3))

        plt.imshow(env, aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
        plt.show()
    # plt.xlabel('frame', fontsize=12)
    # plt.ylabel('stft bin', fontsize=12)
    # # f, axarr = plt.subplots(3, sharex=True, sharey=True)
    # plt.subplot(131, sharex=True, sharey=True)
    # plt.imshow(env[0], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
    # plt.subplot(132)
    # plt.imshow(env[1], aspect='auto', origin='lower', cmap=cmaps.get_cmap('inferno'))
    # plt.subplot(133)



if __name__ == "__main__":
    plot()
