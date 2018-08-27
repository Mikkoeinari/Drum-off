from utils import *
from time import time

t0 = time()
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import os
import sys

sys.setrecursionlimit(10000)
from sklearn.utils import resample
import numpy as np
from MGU import MGU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Dense, GRU, BatchNormalization, GRUCell, Dropout,TimeDistributed, Reshape
from keras.utils import Sequence
# Thresholdit checkissä.
seqLen =32
numDiffHits=100
layerSize = 128
Ichar={}
# data = []
# d=pd.read_csv('funkydrummer.csv',header=None, sep="\t").values
# data=list(truncZeros(np.array(d[:, 1])))
# d=pd.read_csv('kakkosnelonen.csv',header=None, sep="\t").values
# data.extend(list(truncZeros(np.array(d[:, 1]))))
# for i in range(3):
#     d = pd.read_csv('testbeat{}.csv'.format(i), header=None, sep="\t").values
#     data.extend(list(d[:, 1]))
# # d=pd.read_csv('funkydrummer.csv',header=None, sep="\t").as_matrix()
# # data=list(d[:, 1])
#
# print('corpus length:', len(data))
#
# vocab = data
# diffHits = set(data)
# data = vocab
#
# charI = dict((c, i) for i, c in enumerate(diffHits))
# Ichar = dict((i, c) for i, c in enumerate(diffHits))
# numDiffHits = len(charI)
# print('total chars:', numDiffHits)
# words = []
# outchar = []
# for i in range(0, len(data) - seqLen, 1):
#     words.append(data[i: i + seqLen])
#     outchar.append(data[i + seqLen])
# print('nb sequences:', len(words))
# print('Vectorization...')
# X = np.zeros((len(words), seqLen, numDiffHits), dtype=np.bool)
# y = np.zeros((len(words), numDiffHits), dtype=np.bool)
# for i, word in enumerate(words):
#     for t, char in enumerate(word):
#         X[i, t, charI[char]] = 1
#     y[i, charI[outchar[i]]] = 1
# X, y = resample(np.array(X), np.array(y), n_samples=len(words), replace=True)

def vectorizeCSV(filename, seqLen=32):
    global numDiffHits, charI, Ichar
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))

    print('corpus length:', len(data))

    vocab = data
    diffHits = set(data)
    data = vocab

    charI = dict((c, i) for i, c in enumerate(diffHits))
    Ichar = dict((i, c) for i, c in enumerate(diffHits))
    numDiffHits = len(Ichar)
    print('total chars:', numDiffHits)
    words = []
    outchar = []
    for i in range(0, len(data) - seqLen, 1):
        words.append(data[i: i + seqLen])
        outchar.append(data[i + seqLen])
    print('nb sequences:', len(words))
    print('Vectorization...')
    X = np.zeros((len(words), seqLen, numDiffHits), dtype=np.bool)
    y = np.zeros((len(words), numDiffHits), dtype=np.bool)
    for i, word in enumerate(words):
        for t, char in enumerate(word):
            X[i, t, charI[char]] = 1
        y[i, charI[outchar[i]]] = 1
    X, y = resample(np.array(X), np.array(y), n_samples=2000, replace=True)
    return X,y, numDiffHits


def initModel():
    model = Sequential()
    print (numDiffHits )
    model.add(MGU(layerSize, activation='elu',#  kernel_initializer='lecun_normal',
                  return_sequences=False, dropout=0.1, recurrent_dropout=0.1,
                  input_shape=(seqLen, numDiffHits)))

    model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='nadam')
    return model

#print("learning...")
#rerun = True
#if rerun == True or not os.path.isfile('weights_testivedot2.hdf5'):
def train( filename,model=None):
    class drumSeq(Sequence):

        def __init__(self, X_train, y_train, batch_size=200, shuffle=True):
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


    modelsaver = ModelCheckpoint(filepath="./Kits/weights_testivedot2.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", patience=3, mode='auto')
    X_train, y_train, numDiffHits = vectorizeCSV(filename, seqLen)
    tr_gen = drumSeq(X_train, y_train, batch_size=200, shuffle=True)
    if model==None:
        model=getModel()
    #model.fit_generator(generator=tr_gen, steps_per_epoch=200, max_queue_size=10,callbacks=[modelsaver, earlystopper],
    #                    workers=8, use_multiprocessing=True, verbose=1)
    model.fit(X_train, y_train, batch_size=50, epochs=20,
                callbacks=[modelsaver, earlystopper],
                validation_split=0.33
               , verbose=2)

    # #Vectorize a seed x
    model.save_weights("./Kits/weights_testivedot2.hdf5")
    return X_train[-1], model

def getModel(model=None):
    model=initModel()
    try:
        model.load_weights("./Kits/weights_testivedot2.hdf5")
    except Exception as e:
        return model
    return model


# seed_index = random.randint(0, len(data) - seqLen - 1)
def generatePart(data, model=None):
    if model==None:
        model=getModel()
    #print(data[0])
    seed = data[0]
    data=seed
    #print('Model learning time:%0.2f' % (time() - t0))
    #t0 = time()
    print('generating new sequence.')
    generated = []


    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.asarray(a).astype('float64')
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        # choices = range(len(a))
        # return np.random.choice(choices, p=a)
        return np.argmax(np.random.multinomial(1, a, 1))


    for i in range(2048):
        # x = np.zeros((1, seqLen, numDiffHits,1))
        #x = np.zeros((1, seqLen, numDiffHits))
        #for t, k in enumerate(seed):
        #    x[0, t, charI[k]] = 1
        #print('here')
        data=data.reshape(1,seqLen, numDiffHits)
        #print(data.shape)
        pred = model.predict(data, verbose=0)
        # print (np.argmax(pred[0]))
        next_index = sample(pred[0], 0.8)
        # next_index=np.argmax(pred[0])
        next_char = Ichar[next_index]
        generated.append(next_char)

        #print(data.shape)
        next=np.zeros((numDiffHits,), dtype=bool)
        next[next_index]=True
        next=next.reshape(1,1,next.shape[0])
        data=np.concatenate((data[:, 1:, :],next), axis=1)
        
        #print(data.shape)
        #seed = seed[1:]
        #seed.append(next_char)
        #print(seed)

    #generated = splitrowsanddecode(generated)
    gen = pd.DataFrame(generated, columns=[ 'inst'])
    filename='generated{}.csv'.format(time())
    gen.to_csv(filename, index=True, header=None, sep='\t')
    print('valmis')
    return filename
    #print('pattern generating time:%0.2f' % (time() - t0))
    # change to time and midinotes
    gen['time'] = frame_to_time(gen['time'])
    gen['inst'] = to_midinote(gen['inst'])
    gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
    gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))

    madmom.io.midi.write_midi(gen.values, 'midi_testit_gen.mid')
#generatePart(train('./Kits/Default/takes/testbeat1535385910.271116.csv'))