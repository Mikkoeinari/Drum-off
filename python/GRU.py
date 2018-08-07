from utils import *
from time import time
t0=time()

import os
import sys
sys.setrecursionlimit(10000)
from sklearn.utils import resample
import numpy as np
import random
import keras
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import Callback
from scipy.io.wavfile import read, write
from keras.activations import relu
from keras.models import Model, Sequential
from keras.layers import Dense, Activation,GRU, Dropout, LSTM, Convolution1D, TimeDistributed, MaxPooling1D, Flatten
# Thresholdit checkissä.
seqLen=64
layerSize=64
data=[]
d=pd.read_csv('funkydrummer.csv',header=None, sep="\t").as_matrix()
data=list(d[:, 1])
d=pd.read_csv('kakkosnelonen.csv',header=None, sep="\t").as_matrix()
data.extend(list(d[:, 1]))
for i in range(1):
   d=pd.read_csv('testbeat{}.csv'.format(i+1),header=None, sep="\t").as_matrix()
   data.extend(list(d[:, 1]))
#d=pd.read_csv('funkydrummer.csv',header=None, sep="\t").as_matrix()
#data=list(d[:, 1])

print('corpus length:', len(data))

vocab=data
diffHits = set(data)
data = vocab

charI = dict((c, i) for i, c in enumerate(diffHits))
Ichar = dict((i, c) for i, c in enumerate(diffHits))
numDiffHits = len(charI)
print('total chars:', numDiffHits)
words = []
outchar = []
for i in range(0, len(data) - seqLen,1):
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
X,y=resample(np.array(X),np.array(y), n_samples=len(words))
#X = X.reshape(X.shape[0], X.shape[1],X.shape[2], 1)
model = Sequential()

# model.add(TimeDistributed(Convolution1D(1,1, activation='relu',input_shape=(seqLen, numDiffHits,1)), input_shape=(seqLen, numDiffHits,1)))
# model.add(TimeDistributed(MaxPooling1D(1)))
# model.add(TimeDistributed(Flatten()))
model.add(GRU(layerSize,
       return_sequences=False,input_shape=(seqLen, numDiffHits)))
#model.add(Dropout(0.2))
#model.add(GRU(layerSize,return_sequences=False))
model.add(Dropout(0.2))
# model.add(GRU(layerSize))
# model.add(Dropout(0.2))
model.add(Dense(numDiffHits, init='normal', activation='softmax'))
#print(model.summary())

modelsaver=ModelCheckpoint(filepath="weights_testivedot2.hdf5", verbose=1, save_best_only=True)
earlystopper=EarlyStopping(monitor="val_loss", patience=3, mode='auto')
model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer='nadam')
print("learning...")
rerun =True
if rerun == True or not os.path.isfile('weights_testivedot2.hdf5'):
    model.fit(X, y, batch_size=10, nb_epoch=40
              , callbacks=[modelsaver, earlystopper]
              ,validation_split=0.1
              ,verbose=0)
    #model.save_weights("weights_testivedot2.hdf5")
    #model.load_weights("weights_testivedot2.hdf5")
    print("Loaded model from disk")
    #preds = model.predict(X_val)
    #model.save('keras.model_testivedot2.h5')
    print('Model saved to disk.')

# #Vectorize a seed x
model.load_weights("weights_testivedot2.hdf5")
#seed_index = random.randint(0, len(data) - seqLen - 1)

seed=data[-seqLen:]
print('Model learning time:%0.2f' % (time()-t0))
t0=time()
print('generating new sequence.')
generated=[]

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.asarray(a).astype('float64')
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    #choices = range(len(a))
    #return np.random.choice(choices, p=a)
    return np.argmax(np.random.multinomial(1, a, 1))

for i in range(2560):
    x = np.zeros((1, seqLen, numDiffHits))
    for t, k in enumerate(seed):
        x[0, t, charI[k]] = 1
    pred = model.predict(x, verbose=0)
    #print (np.argmax(pred[0]))
    next_index = sample(pred[0], 0.92)
    #next_index=np.argmax(pred[0])
    next_char = Ichar[next_index]
    generated.append(next_char)
    seed = seed[1:]
    seed.append(next_char)
    #print(seed)

generated=splitrowsanddecode(generated)
gen=pd.DataFrame(generated, columns=[ 'time','inst'])

gen.to_csv('generated.csv', index=False, header=None, sep='\t')
print('pattern generating time:%0.2f' % (time()-t0))
#change to time and midinotes
gen['time']=frame_to_time(gen['time'])
gen['inst']=to_midinote(gen['inst'])
gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))



madmom.io.midi.write_midi(gen.values, 'midi_testit_gen.mid')