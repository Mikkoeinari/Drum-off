from keras.layers import Dense, Input, LSTM,GRU
from MGU import MGU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model,load_model
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.utils import resample
from collections import Counter
from utils import *
import os
import sys
import tensorflow as tf
from time import time
t0 = time()
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

seqLen =64
predLen=32
numDiffHits = 127
partLength = 0
layerSize = 128
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
#print(Ichar)
global model, graph
BigX=[]
BigY=[]

def vectorizeCSV(filename, seqLen=32, sampleMul=2, forceGen=False):
    global numDiffHits, charI, Ichar, partLength
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))
    words = []
    targets=[]
    seeds=[]
    for i in range(0, len(data) - seqLen-predLen, 1):
        words.append(data[i: i + seqLen])
        target=data[i + seqLen:i + seqLen+predLen]
        targets.append(target)
        seeds.append([0,0,0,0,0,0,0,0]+target[:-8])
        #seeds.append(data[i + seqLen-8:i + seqLen] + target[:-8])
    # for i in words:
    #     target=i[seqLen-predLen:]
    #     targets.append(target)
    #     seeds.append( [0] + target[:-1])

    print(len(words), len(targets),len(seeds))
    print('nb sequences:', len(words))
    print('Vectorization...')
    if len(words)<1:
        return None, None, None
    def ohe_data(book=None,seqLen=32):
        x=np.zeros((len(book), seqLen, numDiffHits), dtype=np.int)
        print(x.shape)
        for i, word in enumerate(book):
            for t, char in enumerate(word):
                try:
                    x[i, t, charI[char]] = 1
                except:
                    print(i,t,char)
                    break
                    pass
        return x

    X1=ohe_data(words, seqLen)
    X2=ohe_data(targets, predLen)
    y=ohe_data(seeds, predLen)
    if sampleMul is None:
        X1, X2, y = resample(np.array(X1), np.array(X2), np.array(y), n_samples=1, replace=True,
                             random_state=2)
    else:
        X1,X2, y = resample(np.array(X1),np.array(X2), np.array(y), n_samples=len(words)*sampleMul, replace=True, random_state=2)
    return X1,X2, y

def reverse_coding(data):
    x=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        if Ichar.get(np.argmax(data[i])) is None:
            print('jee',data[i], np.argmax(data[i]))
            continue
        x[i] = int(Ichar.get(np.argmax(data[i])))
        #print(x[i])
    return x.astype(np.int)

#buildVocabulary('./big.csv')
#print(Ichar)
#inputs, seeds, outputs = vectorizeCSV('./Kits/mcd2/takes/testbeat1538069795.562399.csv', seqLen=seqLen, sampleMul=1)
#inputs, seeds, outputs = vectorizeCSV('./big.csv', seqLen=seqLen, sampleMul=4)

#print("Shapes: ", inputs.shape, seeds.shape, outputs.shape)
#print("Here is first categorically encoded input sequence looks like: ", inputs[0][0])

def initModel(destroy_old=False):
    global model, enc_model, dec_model
    model, enc_model, dec_model = define_models(numDiffHits, numDiffHits)
    try:
        if destroy_old:
            raise ValueError('old model destroyed!!!')
        else:
            pass
            #model=load_model('./Kits/var_model.hdf5', custom_objects={'MGU': MGU})
    except Exception as e:
        print('virhe1',e)
        if destroy_old:
            model.save_weights("./Kits/weights_varMGU.hdf5")
            model.save('./Kits/var_model.hdf5')
    global graph
    graph = tf.get_default_graph()
    model.load_weights("./Kits/weights_varMGU.hdf5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model, enc_model, dec_model


def define_models(n_input, n_output):
    ## define the encoder architecture
    ## input : sequence
    ## output : encoder states
    encoder_inputs = Input(shape=(None, n_input))
    encoder = MGU(layerSize, return_state=True, dropout=0.5, recurrent_dropout=0.5)
    encoder_outputs, state_h = encoder(encoder_inputs)
    encoder_states = [state_h]

    ## define the encoder-decoder architecture
    ## input : a seed sequence
    ## output : decoder states, decoded output
    decoder_inputs = Input(shape=(None, n_output))
    decoder_mgu = MGU(layerSize, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)
    decoder_outputs, _ = decoder_mgu(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    ## define the decoder model
    ## input : current states + encoded sequence
    ## output : decoded sequence
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(layerSize,))
    decoder_states_inputs = [decoder_state_input_h]
    decoder_outputs, state_h= decoder_mgu(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    model.summary()
    encoder_model.summary()
    decoder_model.summary()
    return model, encoder_model, decoder_model

def predict_sequence(encoder, decoder, sequence):
    output = []
    target_seq = np.array([0.0 for _ in range(numDiffHits)])
    target_seq = target_seq.reshape(1, 1, numDiffHits)

    current_state = encoder.predict(sequence)
    #print(target_seq[0],'poikki',current_state[0],'pokki', np.array(target_seq).shape,np.array(current_state).shape)
    for t in range(predLen):
        #print(t)
        pred, h= decoder.predict([np.array(target_seq),np.array(current_state)])

        def sample(a, temperature=1.0):
            # helper function to sample an index from a probability array
            #x=[]
            #print(a.size)
            i=a
            i = np.asarray(i).astype('float64')
            i = np.log(i) / temperature
            i = np.exp(i) / np.sum(np.exp(i))
            # choices = range(len(a))
            # return np.random.choice(choices, p=a)
            x=(np.random.multinomial(1, i, 1))
            #print(x)
            return x[0]
        output.append(sample(pred[0, 0, :],0.9))
        #output.append(pred[0, 0, :])
        current_state = h
        target_seq = pred
    return np.array(output)


# def define_models(n_input, n_output):
#     ## define the encoder architecture
#     ## input : sequence
#     ## output : encoder states
#     encoder_inputs = Input(shape=(None, n_input))
#     encoder = LSTM(128, return_state=True)
#     encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#     encoder_states = [state_h, state_c]
#
#     ## define the encoder-decoder architecture
#     ## input : a seed sequence
#     ## output : decoder states, decoded output
#     decoder_inputs = Input(shape=(None, n_output))
#     decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
#     decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
#     decoder_dense = Dense(n_output, activation='softmax')
#     decoder_outputs = decoder_dense(decoder_outputs)
#     model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#
#     ## define the decoder model
#     ## input : current states + encoded sequence
#     ## output : decoded sequence
#     encoder_model = Model(encoder_inputs, encoder_states)
#     decoder_state_input_h = Input(shape=(128,))
#     decoder_state_input_c = Input(shape=(128,))
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#     decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]
#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
#     model.summary()
#     encoder_model.summary()
#     decoder_model.summary()
#     return model, encoder_model, decoder_model
#
# def predict_sequence(encoder, decoder, sequence):
#     output = []
#     target_seq = np.array([0.0 for _ in range(numDiffHits)])
#     target_seq = target_seq.reshape(1, 1, numDiffHits)
#
#     current_state = encoder.predict(sequence)
#     for t in range(predLen):
#         pred, h, c = decoder.predict([target_seq] + current_state)
#         output.append(pred[0, 0, :])
#         current_state = [h, c]
#         target_seq = pred
#     return np.array(output)

#autoencoder, encoder_model, decoder_model = define_models(numDiffHits, numDiffHits)
# = EarlyStopping(monitor="val_loss", min_delta=0.1, patience=20, mode='auto')
#autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

def train(filename=None, seqLen=32, sampleMul=2, forceGen=False):
    modelsaver = ModelCheckpoint(filepath="./Kits/weights_varMGU.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", min_delta=0.1, patience=6, mode='auto')

    if filename!=None:
        X1, X2, y = vectorizeCSV(filename, seqLen, sampleMul, forceGen=forceGen)
        if X1 is None:
            return None
        #X_train, y_train, numDiffHits = keras_encode(filename, seqLen, sampleMul, forceGen=forceGen)


    with graph.as_default():
        model.load_weights("./Kits/weights_varMGU.hdf5")
        model.fit([X1,X2], y, batch_size=50, epochs=100,
                  callbacks=[modelsaver, earlystopper],
                  validation_split=0.33,
                  verbose=2)

        model.load_weights("./Kits/weights_varMGU.hdf5")
        #model.save('./Kits/var_model.hdf5')
        #model.save_weights("./Kits/weights_testivedot2.hdf5")
    return X1[0]

#autoencoder.fit([inputs, seeds], outputs,batch_size=50, epochs=100, validation_split=0.33, verbose=2, callbacks=[earlystopper])
buildVocabulary('./newVocab.csv')
model, enc_model, dec_model = initModel(destroy_old=False)
##vectorizeCSV('./Kits/Default/takes/testbeat1535385910.271116.csv')

#
##print(takes)
if False:
    takes = [f for f in os.listdir('./Kits/mcd2/takes/') if not f.startswith('.')]
    takes.sort()
    for i in takes:
        print(i)
        #vectorizeCSV('./Kits/mcd2/takes/{}'.format(i),seqLen, sampleMul=4)
        train('./Kits/mcd2/takes/{}'.format(i),seqLen=seqLen,sampleMul=1)
    print('training a model from scratch:%0.2f' % (time() - t0))



X1, X2, y = vectorizeCSV('./Kits/mcd2/takes/testbeat1538990686.408342.csv', seqLen=seqLen, sampleMul=None)
train('./Kits/mcd2/takes/testbeat1538990686.408342.csv',seqLen=seqLen,sampleMul=4)
generated=[]
print(X1.shape)
for i in range(30):
    new_seq=predict_sequence(enc_model, dec_model, X1)
    generated.extend(reverse_coding(new_seq))
    #print(new_seq.shape)
    new_seq=new_seq.reshape(1, predLen, numDiffHits)
    X1 = np.concatenate((X1[:,predLen:,:], new_seq), axis=1)

gen = pd.DataFrame(generated, columns=['inst'])
filename = 'generated{}.csv'.format(time())
gen.to_csv(filename, index=True, header=None, sep='\t')
print('valmis')
# change to time and midinotes
generated = splitrowsanddecode(generated)
gen = pd.DataFrame(generated, columns=['time', 'inst'])
#Cut over 30s.
maxFrames=30/(1/SAMPLE_RATE*Q_HOP)
gen = gen[gen['time'] < maxFrames]
gen['time'] = frame_to_time(gen['time'], hop_length=Q_HOP)
gen['inst'] = to_midinote(gen['inst'])
gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))

madmom.io.midi.write_midi(gen.values, 'midi_testit_gen_var.mid')
    # print('\nInput Sequence=%s SeedSequence=%s, PredictedSequence=%s'
    #       % (reverse_coding(X1[0]), reverse_coding(y[0]), reverse_coding(target)))