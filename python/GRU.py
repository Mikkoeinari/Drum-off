from utils import *
from time import time
from random import randint
import pickle
t0 = time()
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import os
import sys
import tensorflow as tf

sys.setrecursionlimit(10000)
from sklearn.utils import resample
import numpy as np
from MGU import MGU
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,Callback
from keras.models import Model, Sequential,load_model
from keras.layers import Dense, GRU, BatchNormalization, GRUCell, Dropout, TimeDistributed, Reshape, LSTM, Activation
from keras.utils import Sequence,to_categorical
from keras import regularizers
from keras.optimizers import nadam,RMSprop
from collections import Counter
import ISRLU
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)




# Thresholdit checkissä.
seqLen =64
numDiffHits = 128
partLength = 0
lastLoss=0
Ichar = {}
charI = {}
def buildVocabulary(filename):
    global numDiffHits, charI, Ichar, partLength
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))
    # sort by frequency
    counts = Counter(data)
    #print([x[0] for x in counts.most_common()])
    new_list = sorted(data, key=counts.get, reverse=True)
    diffHits = set(data)
    #diffHits=[x[0] for x in counts.most_common()]
    charI = dict((c, i) for i, c in enumerate(diffHits))
    Ichar = dict((i, c) for i, c in enumerate(diffHits))
    print(len(charI))
    print('vocabulary built')

def getVocabulary():
    global numDiffHits, charI, Ichar, partLength
    return Ichar, charI


#print(Ichar)
global model, graph
BigX=[]
BigY=[]
# try:
#     BigX= pickle.load(open("./bigx.big", 'rb'))
#     BigY = pickle.load(open("./bigy.big", 'rb'))
# except:
#     print('bigs not found')
def vectorizeCSV(filename, seqLen=32, sampleMul=1., forceGen=False, bigFile=None):
    global numDiffHits, charI, Ichar, partLength
    if filename is not None:
        data = []
        d = pd.read_csv(filename, header=None, sep="\t").values
        if bigFile is 'extreme':
            target =randint(0, d[-1,0]-200)
            data.extend(list(d[target:target + 200, 1]))
        else:
            data.extend(list(d[:, 1]))

    elif bigFile is not None:
        data=bigFile
    data = data[:128]
    # print('corpus length:', len(data))
    partLength = 360#max([len(data),720])
    if forceGen:
        data=data[:180]
        #return
    words = []
    outchar = []
    for i in range(0, len(data) - seqLen, 1):
        words.append(data[i: i + seqLen])
        outchar.append(data[i + seqLen])
    #print('nb sequences:', len(words))
    #of data too short.
    if len(words)<1:
        return None, None, None

    #reduce training bias by sampling small size data more
    if bigFile=='separate':
        sampleMul=1000/len(words)
    #print('Vectorization...')
    X = np.zeros((len(words), seqLen, numDiffHits), dtype=np.bool)
    X0=X[0]
    y = np.zeros((len(words), numDiffHits), dtype=np.bool)
    y0=y[0]
    BigX.extend(data)
    #BigY.extend(outchar)
    for i, word in enumerate(words):
        for t, char in enumerate(word):
            # If we find a hit not in vocabulary of #numDiffHits we simply omit that
            # Pitääkö järjestää??
            try:
                X[i, t, charI[char]] = 1
            except:
                #print(char)
                pass
        try:
            y[i, charI[outchar[i]]] = 1
        except:
            pass

    #BX,BY=resample(np.array(X), np.array(y), n_samples=len(words), replace=False)
    samples=np.max([int(len(words)*sampleMul),333])

    X, y = resample(np.array(X), np.array(y), n_samples=samples, replace=True, random_state=2)
    X[0]=X0
    y[0]=y0
    return X, y, numDiffHits


# Just the hits: Not working
def vectorize(filename, seqLen=32, sampleMul=1., forceGen=False):
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))

    def binarize(a):
        if a < 0:
            return map(int, ['1', '0'] + list(format(a, "0{}b".format(nrOfDrums)))[1:])
        else:
            return map(int, list(format(a, "0{}b".format(nrOfDrums + 1))))

    def binarize2(a):
        return [x for x in bin(a)[2:]]

    data = [list(binarize(i)) for i in data]
    #print(data)
    words = []
    outchar = []
    #print('corpus length:', len(data))
    for i in range(0, len(data) - seqLen, 1):
        words.append(data[i: i + seqLen])
        outchar.append(data[i + seqLen])
    #print('nb sequences:', len(words))
    X = np.array(words)
    y = np.array(outchar)
    X, y = resample(np.array(X), np.array(y), n_samples=len(words), replace=True)
    return X, y, numDiffHits

def label_encode(filename, seqLen=32, sampleMul=1., forceGen=False):
    'Returns a DataFrame with encoded columns'
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))
    data=pd.Series(data)
    factorised = pd.factorize(d[:, 1])[1]
    labels = pd.Series(range(len(factorised)), index=factorised)
    encoded_col = data.map(labels)
    encoded_col[encoded_col.isnull()] = -1
    words = []
    outchar = []
    #print('corpus length:', len(encoded_col))
    for i in range(0, len(encoded_col) - seqLen, 1):
        words.append(encoded_col[i: i + seqLen])
        outchar.append(encoded_col[i + seqLen])
    #print('nb sequences:', len(words))
    X = np.array(words)
    y = np.array(outchar)
    X, y = resample(np.array(X), np.array(y), n_samples=len(words)*sampleMul, replace=True)
    return X, y, numDiffHits
def freq_encode(filename, seqLen=32, sampleMul=1., forceGen=False):
    '''Returns a DataFrame with encoded columns'''
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))
    data=pd.Series(data)
    freqs_cat = data.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
    #freqs_cat = data.count()/data.shape[0]
    encoded_col = data.map(freqs_cat)
    encoded_col[encoded_col.isnull()] = 0
    words = []
    outchar = []
    #print('corpus length:', len(encoded_col))
    for i in range(0, len(encoded_col) - seqLen, 1):
        words.append(encoded_col[i: i + seqLen])
        outchar.append(encoded_col[i + seqLen])
    #print('nb sequences:', len(words))
    X = np.array(words)
    y = np.array(outchar)
    X, y = resample(np.array(X), np.array(y), n_samples=len(words) * sampleMul, replace=True)
    return X, y, numDiffHits

def keras_encode(filename, seqLen=32, sampleMul=1., forceGen=False):
    'Returns a DataFrame with encoded columns'
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))
    #data=pd.Series(data)
    encoded_col = to_categorical(data,numDiffHits*2)
    #encoded_col[encoded_col.isnull()] = -1
    words = []
    outchar = []
    #print('corpus length:', len(encoded_col))
    for i in range(0, len(encoded_col) - seqLen, 1):
        words.append(encoded_col[i: i + seqLen])
        outchar.append(encoded_col[i + seqLen])
    #print('nb sequences:', len(words))
    X = np.array(words)
    y = np.array(outchar)
    X, y = resample(np.array(X), np.array(y), n_samples=len(words)*sampleMul, replace=True)
    return X, y, numDiffHits

def initModel(seqLen=32, destroy_old=False):
    global model
    layerSize = 128
    try:
        if destroy_old:
            raise ValueError('old model destroyed!!!')
        else:
            model = load_model('./Kits/model.hdf5', custom_objects={'MGU': MGU})
    except Exception as e:
        print('new model!',e)
        model = Sequential()
        # print (numDiffHits )
        model.add(MGU(layerSize, activation='elu',  # kernel_initializer='lecun_normal',
                      return_sequences=False, dropout=0.2, recurrent_dropout=0.2,
                      input_shape=(seqLen, numDiffHits),implementation=1))
        #model.add(MGU(int(layerSize), activation='elu', return_sequences=False,implementation=1))
        model.add(Dense(1024, activation='elu'))
        model.add(Dropout(0.2))
        #model.add(Dense(128, activation='elu'))
        #model.add(Dropout(0.2))
        model.add(Dense(layerSize, activation='elu'))
        model.add(Dropout(0.15))
        model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        print(model.summary())
        optr = nadam(lr=0.003)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optr)
        if destroy_old:
            model.save_weights("./Kits/weights_testivedot2.hdf5")
            model.save('./Kits/model.hdf5')
    global graph
    graph = tf.get_default_graph()
    return model


# print("learning...")
# rerun = True
# if rerun == True or not os.path.isfile('weights_testivedot2.hdf5'):
def train(filename=None, seqLen=seqLen, sampleMul=1., forceGen=False, bigFile=None, updateModel=False):
    global lastLoss
    class drumSeq(Sequence):

        def __init__(self,filename, batch_size=200, shuffle=True):
            'Initialization'
            self.batch_size = batch_size
            self.y = y_train
            self.X = X_train
            self.shuffle = shuffle
        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.X) / self.batch_size))

        def __getitem__(self, idx):

            return resample(np.array(self.X), np.array(self.y), n_samples=self.batch_size, replace=self.shuffle)

    def klik(epoch):
        #return 0.003/((epoch+1))+0.0007
        return np.max([0.001,0.0005*np.log2(epoch+1)])

    #Callbacks#
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    modelsaver = ModelCheckpoint(filepath="./Kits/weights_testivedot2.hdf5", verbose=1, save_best_only=True)
    temporarysaver = ModelCheckpoint(filepath="./Kits/temp.hdf5", verbose=0, save_best_only=True)
    genMdelSaver=ModelCheckpoint(filepath="./Kits/weights_testivedot_ext.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=2, mode='auto')
    history = LossHistory()


    learninratescheduler=LearningRateScheduler(klik, verbose=1)
    if filename is not None and updateModel is not 'extreme':
        #X_train, y_train, numDiffHits=freq_encode(filename, seqLen, sampleMul, forceGen=forceGen)
        #X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_train, y_train, numDiffHits = vectorizeCSV(filename, seqLen, sampleMul, forceGen=forceGen)
        if X_train is None:
            return None
        #X_train, y_train, numDiffHits = keras_encode(filename, seqLen, sampleMul, forceGen=forceGen)
    elif bigFile is not None and updateModel is not 'extreme':
        #X_train, y_train, numDiffHits = vectorizeCSV(filename, seqLen, sampleMul, forceGen=forceGen, bigFile=bigFile)
        X_train,y_train=bigFile
        if X_train is None:
            return None
    elif updateModel is not 'extreme':
        #print(len(BigX))
        X_train, y_train=resample(np.array(BigX), np.array(BigY), n_samples=len(BigX), replace=True)
    if updateModel is 'extreme':
        def myGenerator():
            while True:
                x, y,_ = vectorizeCSV(filename, seqLen, sampleMul, forceGen=forceGen, bigFile='extreme')
                yield (x, y)

        X_test, y_test,_ = vectorizeCSV(filename, seqLen, sampleMul, forceGen=forceGen, bigFile='extreme')
        X_train=X_test
        tr_gen = myGenerator()
    if forceGen:
        pass
        return X_train[0]
    # model=getModel()
    # model.fit_generator(generator=tr_gen, steps_per_epoch=200, max_queue_size=10,callbacks=[modelsaver, earlystopper],
    #                    workers=8, use_multiprocessing=True, verbose=1)
    with graph.as_default():
        #model=getModel()
        if updateModel=='extreme':
            model.save_weights("./Kits/weights_testivedot_ext.hdf5")
            model.fit_generator(generator=tr_gen,epochs=20, steps_per_epoch=20, max_queue_size=10,callbacks=[genMdelSaver, earlystopper],
                              workers=8, use_multiprocessing=False, verbose=1, validation_data=(X_test, y_test))
            model.load_weights("./Kits/weights_testivedot_ext.hdf5")
            model.save('./Kits/model_ext.hdf5')
        #print(X_train.shape)
        if updateModel==True:
            model.fit(X_train, y_train, batch_size=500, epochs=20,
                  callbacks=[modelsaver, earlystopper, history],# learninratescheduler],
                  validation_split=0.33,
                  verbose=2)
            model.load_weights("./Kits/weights_testivedot2.hdf5")
            model.save('./Kits/model.hdf5')
        elif updateModel==False:
            model.load_weights("./Kits/weights_testivedot2.hdf5")
            model.fit(X_train, y_train, batch_size=50, epochs=20,
                      callbacks=[temporarysaver,history],  # learninratescheduler],
                      validation_split=(1/3.),
                      verbose=2)
            #take mean of recent iteration losses for fuzz scaler
            lastLoss=np.mean(history.losses[-10:])
            print(lastLoss)
            model.load_weights("./Kits/temp.hdf5")
        #model.save_weights("./Kits/weights_testivedot2.hdf5")
    return X_train[0]


#
# def getModel(model=None):
#     try:
#         model.load_weights("./Kits/weights_testivedot2.hdf5")
#     except Exception as e:
#         print(e)
#         model = initModel()
#         return model
#     return model


# seed_index = random.randint(0, len(data) - seqLen - 1)
def generatePart(data):
    # model=getModel()
    print(data.shape)
    seed = data
    data = seed

    # print('Model learning time:%0.2f' % (time() - t0))
    # t0 = time()
    print('generating new sequence.')

    generated=[]
    #save seed
    for i in data:
        try:
            pass
            #generated.append(Ichar.get(np.where(i==True)[0][0],0))
        except Exception as e:
            print('gen-init: ', e)
            pass
        #print([[np.where(value==True)[0]] for value in i])

    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.asarray(a).astype('float64')
        a=a ** (1 / temperature)
        a_sum=a.sum()
        a=a/a_sum
        #a = np.log(a) / temperature
        #a = np.exp(a) / np.sum(np.exp(a))
        # choices = range(len(a))
        # return np.random.choice(choices, p=a)
        return np.argmax(np.random.multinomial(1, a, 1))

    for i in range(partLength):
        # x = np.zeros((1, seqLen, numDiffHits,1))
        # x = np.zeros((1, seqLen, numDiffHits))
        # for t, k in enumerate(seed):
        #    x[0, t, charI[k]] = 1
        # print('here')
        data = data.reshape(1, seqLen, numDiffHits)

        # print(data.shape)
        with graph.as_default():
            pred = model.predict(data, verbose=0)
        # print (np.argmax(pred[0]))
        fuzz=np.max([1.0-lastLoss,0.1])
        next_index = sample(pred[0],fuzz)
        # next_index=np.argmax(pred[0])

        next_char = Ichar.get(next_index,0)
        generated.append(next_char)

        # print(data.shape)
        next = np.zeros((numDiffHits,), dtype=bool)
        next[next_index] = True
        next = next.reshape(1, 1, next.shape[0])
        data = np.concatenate((data[:, 1:, :], next), axis=1)

        # print(data.shape)
        # seed = seed[1:]
        # seed.append(next_char)
        # print(seed)

    # generated = splitrowsanddecode(generated)
    gen = pd.DataFrame(generated, columns=['inst'])
    filename = 'generated{}.csv'.format(time())
    gen.to_csv(filename, index=True, header=None, sep='\t')
    print('valmis')
    #return filename
    # change to time and midinotes
    generated = splitrowsanddecode(generated)
    gen = pd.DataFrame(generated, columns=['time', 'inst'])
    #Cut over 30s.
    maxFrames=300/(1/SAMPLE_RATE*Q_HOP)
    gen = gen[gen['time'] < maxFrames]
    gen['time'] = frame_to_time(gen['time'], hop_length=Q_HOP)
    gen['inst'] = to_midinote(gen['inst'])
    gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
    gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))

    madmom.io.midi.write_midi(gen.values, 'midi_testit_gen.mid')

#####################################
#Testing from here on end
#make midi from source file
if False:
    generated=pd.read_csv('dataklimp0b.csv', header=None, sep="\t", usecols=[1])
    print(generated.head())
    generated = splitrowsanddecode(generated[1][:5000])
    gen = pd.DataFrame(generated, columns=['time', 'inst'])

    gen['time'] = frame_to_time(gen['time'], hop_length=Q_HOP)

    gen['inst'] = to_midinote(gen['inst'])
    gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
    gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))

    madmom.io.midi.write_midi(gen.values, 'midi_testit_original.mid')
    exit(0)
def rebuild_vocabulary(vocab=None, newData=None):
    data=[]
    for i in newData:
        d = pd.read_csv('./Kits/mcd2/takes/{}'.format(i), header=None, sep="\t").values
        data.extend(list(d[:, 1]))
    d = pd.read_csv(vocab, header=None, sep="\t").values
    data.extend(list(d[:, 1]))
    huge=pd.DataFrame(data)
    huge.to_csv('./huge.csv', index=True, header=None, sep='\t')
    counts = Counter(data)
    #Thow out rare hits
    vals=[x for x, _ in counts.most_common(127)]
    newVocab=list(set(vals))
    newVocab = pd.DataFrame(newVocab)
    filename='newVocab.csv'
    newVocab.to_csv(filename, index=True, header=None, sep='\t')
    return filename

buildVocabulary('./newVocab.csv')
new=False
model = initModel(seqLen, destroy_old=new)
##vectorizeCSV('./Kits/Default/takes/testbeat1535385910.271116.csv')
#
##print(takes)
if new:
    takes = [f for f in os.listdir('./Kits/mcd2/takes/') if not f.startswith('.')]
    takes.sort()
    data = []
    filename=rebuild_vocabulary('./newVocab.csv', takes)
    buildVocabulary(filename)
    trainingRuns=12
    xss = []
    yss = []
    for j in range(trainingRuns):
        for i in takes:
            #print(i)
            X,y, _=vectorizeCSV('./Kits/mcd2/takes/{}'.format(i),seqLen, sampleMul=1)
            if X is None:
                continue
            xss.append(X)
            yss.append(y)
        Xs = np.concatenate((xss))
        ys = np.concatenate((yss))
        #print(Xs.shape, ys.shape)
           #train('./Kits/mcd2/takes/{}'.format(i),seqLen=seqLen,sampleMul=4)
            #d = pd.read_csv('./Kits/mcd2/takes/{}'.format(i), header=None, sep="\t").values
            #data.extend(list(d[:, 1]))
        #train(filename=None, seqLen=seqLen, sampleMul=4, bigFile=data, updateModel=True)
        #print(len(Xs))
        train(filename=None,seqLen=seqLen,sampleMul=1, bigFile=[Xs,ys], updateModel=True)
    print('training a model from scratch:%0.2f' % (time() - t0))
#t0=time()
#print('going big')
#generatePart(train('dataklimp0b.csv', seqLen, sampleMul=1.5, forceGen=False, updateModel='extreme'))
generatePart(train('./Kits/mcd2/takes/testbeat1538990686.408342.csv', seqLen, sampleMul=1.5, forceGen=False))
#print('gen:%0.2f' % (time() - t0))
    #bigdata=pd.DataFrame(BigX, columns=['inst'])
    #bigdata.to_csv('./big.csv', index=True, header=None, sep='\t')
    #pickle.dump(BigX, open("./bigx.big", 'wb'))
    #pickle.dump(BigY, open("./bigy.big", 'wb'))
#t0=time()
#see what hits we have and how much
# unique, counts = np.unique(BigX, return_counts=True)
# a=np.array([counts,unique], dtype=int)
# print(len(a[:,a[1]<0][0]),len(a[0]>1),(a[:,a[0]>2][1]))

#generatePart(train('./big.csv',seqLen,sampleMul=4, forceGen=True))
#generatePart(train('./Kits/mcd2/takes/testbeat1537979079.645227.csv', seqLen, sampleMul=4,forceGen=True))
#print('train & gen:%0.2f' % (time() - t0))




"""

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
mgu_1 (MGU)                  (None, 32, 128)           65792
_________________________________________________________________
mgu_2 (MGU)                  (None, 128)               65792
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512
=================================================================
Total params: 148,096
Trainable params: 148,096
Non-trainable params: 0
#- 1s - loss: 1.4464 - acc: 0.5304 - val_loss: 1.4337 - val_acc: 0.5514
#training a model from scratch: 202.95tanh mgu implementation 1
#- 2s - loss: 1.1917 - acc: 0.6077 - val_loss: 1.1101 - val_acc: 0.6555
#training a model from scratch:216.93 elu mgu implementation 1

#- 1s - loss: 1.5954 - acc: 0.4930 - val_loss: 1.5124 - val_acc: 0.5476
#training a model from scratch:182.24 tanh mgu implementation 2
#- 1s - loss: 1.5348 - acc: 0.5051 - val_loss: 1.3849 - val_acc: 0.5656
#training a model from scratch:191.90 elu mgu implementation 2

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru_1 (GRU)                  (None, 32, 128)           98688
_________________________________________________________________
gru_2 (GRU)                  (None, 128)               98688
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512
=================================================================
Total params: 213,888
Trainable params: 213,888
Non-trainable params: 0

#- 2s - loss: 1.1512 - acc: 0.6242 - val_loss: 1.0700 - val_acc: 0.7031
#training a model from scratch:292.84 elu GRU imp1

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)                (None, 32, 128)           131584    
_________________________________________________________________
lstm_2 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
=================================================================
Total params: 279,680
Trainable params: 279,680
Non-trainable params: 0

 - 3s - loss: 1.2746 - acc: 0.5843 - val_loss: 1.3662 - val_acc: 0.6015
training a model from scratch:349.96
"""
