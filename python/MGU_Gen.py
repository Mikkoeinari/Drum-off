import threading
import time
from utils import *
t0 = time.time()
import os
import sys
sys.setrecursionlimit(10000)
from sklearn.utils import resample
import numpy as np
from MGU import MGU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.utils import Sequence

# Thresholdit checkissä.
seqLen=64
layerSize = 128
VOCABULARY=()
numDiffHits=0
def vectorizeCSV(filename, seqLen):
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))

    print('corpus length:', len(data))

    vocab = data
    diffHits = set(data)
    data = vocab

    charI = dict((c, i) for i, c in enumerate(diffHits))
    VOCABULARY = dict((i, c) for i, c in enumerate(diffHits))
    numDiffHits = len(VOCABULARY)
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
    return X,y, 200


class drumSeq(Sequence):

    def __init__(self, X_train,y_train, batch_size=200, shuffle=True):
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



X_train,y_train,numDiffHits=vectorizeCSV('testbeat0.csv', seqLen)
tr_gen = drumSeq(X_train,y_train,batch_size=200,shuffle=True)

model = Sequential()
model.add(MGU(layerSize, activation='selu',  kernel_initializer='lecun_normal',
              return_sequences=False, dropout=0.1, recurrent_dropout=0.1,
              input_shape=(seqLen, numDiffHits)))
model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='nadam')

model.fit_generator(generator=tr_gen, steps_per_epoch=200, max_queue_size=10,
                   workers=3, use_multiprocessing=True)