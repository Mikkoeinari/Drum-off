from hmmlearn import hmm
import pandas as pd
import numpy as np
from collections import Counter
from utils import *
from sklearn.externals import joblib

seqLen =1
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
def vectorize(filename):
    global Ichar, charI
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))
    data=np.array(data)
    atoms=np.unique(data)
    charI = dict((c, i) for i, c in enumerate(atoms))
    Ichar = dict((i, c) for i, c in enumerate(atoms))
    for i in range(data.size):
       data[i] = charI[data[i]]
    words=[]
    for i in range(0, len(data) - seqLen, 1):
        words.append(data[i: i + seqLen])
    X = np.array(words)

    return X, np.unique(data), numDiffHits

def init_model(n_components):
    return hmm.GaussianHMM(n_components=n_components,n_iter=300, verbose=True,
                           algorithm='viterbi',tol=0.1)

def prepare_data(filename):
    if filename is not None:
        data = []
        d = pd.read_csv(filename, header=None, sep="\t").values
        data.extend(list(d[:, 1]))
        data=np.reshape(np.array(data),(-1,1))
        print(data)
    return data

def train_model(model, X, lengths=None, use_new_model=True):
    if not use_new_model:
        try:
            model=joblib.load('HMMmodel.pkl')
            print('model loaded')
        except:
            print('virihe')
    else:
        model.fit(X, lengths)
        joblib.dump(model, "HMMmodel.pkl")
    return model

def gen_x_samples(model,x):
    return model.sample(x)

def sample_prob(a, temperature=.2):
    # helper function to sample an index from a probability array
    a = np.asarray(a).astype('float64')
    a=a ** (1 / temperature)
    a_sum=a.sum()
    a=a/a_sum
    return np.argmax(np.random.multinomial(1, a, 1))

def predict_next(model, sequence, n_samples, temp=1):
    ret_seq=[]
    for i in range(n_samples):
        x=model.predict(sequence)
        prob_next_step = model.transmat_[x[-1], :]
        sample=sample_prob(prob_next_step, temperature=temp)
        ret_seq.append(Ichar.get(sample))

        sample=np.array([sample])
        sample = sample.reshape(1, sample.shape[0])
        sequence=np.concatenate((sequence[1:, :], sample), axis=0)
    return ret_seq

#buildVocabulary('./newVocab.csv')
sequences, targets, atoms=vectorize('../../midi_data_set/mididata0.csv')
print(sequences.shape, targets.shape)
sequence=sequences.ravel().reshape(-1,1)
model=init_model(targets.shape[0])
model=train_model(model,sequence, use_new_model=True)
#print(sequences[0])
#gen=model.predict(sequences.reshape(-1,1))

if False:
    probs, gen = gen_x_samples(model, 1000)
    print(probs.shape)
    data=[]
    for i in gen:
        data.append(Ichar.get(i))
    gen=data
else:
    gen = predict_next(model, sequence[:, -100:], 100, temp=1)
generated=splitrowsanddecode(gen)
gen = pd.DataFrame(generated, columns=['time', 'inst'])
gen['time'] = frame_to_time(gen['time'], hop_length=HOP_SIZE)
gen['inst'] = to_midinote(gen['inst'])
gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))

madmom.io.midi.write_midi(gen.values, 'midi_testit_gen.mid')