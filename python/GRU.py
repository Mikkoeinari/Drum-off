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
import pandas as pd

sys.setrecursionlimit(10000000)
from sklearn.utils import resample
import numpy as np
from MGU import MGU
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,Callback
from keras.models import Model, Sequential,load_model
from keras.layers import *#,Dense,Conv1D,Flatten,TimeDistributed,Input,  MaxPooling1D,GlobalAveragePooling1D,GRU, BatchNormalization, GRUCell, Dropout, TimeDistributed, Reshape, LSTM, Activation
from keras.utils import Sequence,to_categorical
from keras import regularizers
from keras.optimizers import *#nadam
from collections import Counter
from keras import metrics
#import ISRLU
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

seqLen=64
dvs=[8,16,32,64]
numDiffHits = 0
partLength = 0
lastLoss=0
Ichar = {}
charI = {}
def buildVocabulary(filename=None, hits=None):
    global numDiffHits, charI, Ichar, partLength
    data = []
    if filename is not None:
        d = pd.read_csv(filename, header=None, sep="\t").values
        data.extend(list(d[:, 1]))
        # sort by frequency
        counts = Counter(data)
        #print([x[0] for x in counts.most_common()])
        new_list = sorted(data, key=counts.get, reverse=True)
    elif hits is not None:
        data=list(hits)
    else:
        print('filename and hits can not both be none')
    diffHits = set(data)
    numDiffHits=len(diffHits)
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
def vectorizeCSV(filename, seqLen=32, sampleMul=1., forceGen=False, bigFile=None, generator=False):
    global numDiffHits, charI, Ichar, partLength
    if filename is not None:
        #Init with pauses before drumming starts
        data = list(np.full(seqLen,-32))
        d = pd.read_csv(filename, header=None, sep="\t").values
        if bigFile is 'extreme':
            target =randint(0, d[-1,0]-200)
            data.extend(list(d[target:target + 200, 1]))
        else:
            data.extend(list(d[:, 1]))

    if forceGen:
        data=data[-180:]
        #return
    words = []
    outchar = []
    if generator:
        for i in range(generator):
            loc=np.random.randint(0, len(data) - seqLen)
            words.append(data[loc: loc + seqLen])
            outchar.append(data[loc + seqLen])
    else:
        for i in range(0, len(data) - seqLen, 1):
            words.append(data[i: i + seqLen])
            outchar.append(data[i + seqLen])

    if len(words)<1:
        return None, None, None

    X = np.zeros((len(words), seqLen, numDiffHits), dtype=np.bool)
    y = np.zeros((len(words), numDiffHits), dtype=np.bool)

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
    if generator:
        return np.array(X), np.array(y), numDiffHits

    samples=np.max([int(len(words)*sampleMul),33])
    X, y = resample(np.array(X), np.array(y), n_samples=samples, replace=True, random_state=2)
    return X, y, numDiffHits

def get_sequences(filename, seqLen=32, sampleMul=1., forceGen=False):
    #If you use this you must use Embedding layer and sparse_categorical_crossentropy
    # i.e. model.add(Embedding(numDiffHits+1,numDiffHits, input_length=seqLen))
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))

    data = [charI[i] for i in data]
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
    samples = np.max([int(len(words) * sampleMul), 33])
    X, y = resample(np.array(X), np.array(y), n_samples=samples, replace=True)
    return X, y, numDiffHits

def initModel(seqLen=64,kitPath=None, destroy_old=False, model_type='parallel_mgu'):
    global model
    layerSize =numDiffHits
    if kitPath is None:
        filePath='./Kits/'
    else:
        filePath=kitPath
    print(kitPath)
    try:
        if destroy_old:
            raise ValueError('old model destroyed!!!')
        else:
            model = load_model('{}model_{}.hdf5'.format(filePath, model_type), custom_objects={'MGU': MGU})
            optr = nadam(lr=0.00001)
            # optr=SGD(lr=0.1)
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optr)
    except Exception as e:
        print('Making new model')
        model = Sequential()
        # print (numDiffHits )
        if model_type=='conv_mgu':
            #Conv1D before MGU
            model.add(Conv1D(64,16, activation='elu', input_shape=(seqLen, numDiffHits)))
            model.add(Dropout(0.3))
            model.add(MGU(layerSize, activation='elu', return_sequences=False, dropout=0.3, recurrent_dropout=0.3,
                          implementation=1))
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type=='parallel_mgu' or model_type=='parallel_conc_mgu':
            drop=0.4
            unroll=False
            use_bias=False
            ret_seq=False
            #model1
            in1 = Input(shape=(seqLen, numDiffHits))
            #reshape1=Reshape((1,seqLen, numDiffHits))(in1)
            #conv1=TimeDistributed(Conv1D(64,4, activation='elu'), input_shape=(1,)+(seqLen, numDiffHits))(reshape1)
            #flat1=TimeDistributed(Flatten())(conv1)
            #drop1=TimeDistributed(Dropout(0.85))(flat1)
            mgu1=(MGU(layerSize, activation='elu', return_sequences=ret_seq, dropout=drop, recurrent_dropout=0,
                          implementation=1, unroll=unroll, use_bias=use_bias))(in1)
            #dense1=Dense(layerSize*10, activation='elu')(mgu1)

            #model2
            in2=Input(shape=(int(seqLen/dvs[0]), numDiffHits))
            #reshape2 = Reshape((1, int(seqLen/2), numDiffHits))(in2)
            #conv2 = TimeDistributed(Conv1D(64, 4, activation='elu'), input_shape=(1,) + (seqLen, numDiffHits))(reshape2)
            #flat2 = TimeDistributed(Flatten())(conv2)
            #drop2 = TimeDistributed(Dropout(0.85))(flat2)
            mgu2=(MGU(int(layerSize), activation='elu',return_sequences=ret_seq, dropout=drop, recurrent_dropout=0,
                     implementation=1, unroll=unroll, use_bias=use_bias))(in2)
            #dense2 = Dense(layerSize*10, activation='elu')(mgu2)

            # model3
            in3 = Input(shape=(int(seqLen/dvs[1]), numDiffHits))
            #reshape3 = Reshape((1, int(seqLen / 4), numDiffHits))(in3)
            #conv3 = TimeDistributed(Conv1D(64, 4, activation='elu'), input_shape=(1,) + (seqLen, numDiffHits))(reshape3)
            #flat3 = TimeDistributed(Flatten())(conv3)
            #drop3 = TimeDistributed(Dropout(0.85))(flat3)
            mgu3 = (MGU(int(layerSize), activation='elu', return_sequences=ret_seq, dropout=drop, recurrent_dropout=0,
                       implementation=1, unroll=unroll, use_bias=use_bias))(in3)
            #dense3 = Dense(layerSize*10, activation='elu')(mgu3)

            # model4
            in4 = Input(shape=(int(seqLen/dvs[2]), numDiffHits))
            #reshape4 = Reshape((1, int(seqLen / 8), numDiffHits))(in4)
            #conv4 = TimeDistributed(Conv1D(64, 4, activation='elu'), input_shape=(1,) + (seqLen, numDiffHits))(reshape4)
            #flat4 = TimeDistributed(Flatten())(conv4)
            #drop4 = TimeDistributed(Dropout(0.85))(flat4)
            mgu4 = (MGU(int(layerSize), activation='elu', return_sequences=ret_seq, dropout=drop, recurrent_dropout=0,
                       implementation=1, unroll=unroll, use_bias=use_bias))(in4)
            #dense4 = Dense(layerSize*10, activation='elu')(mgu4)

            # model5
            in5 = Input(shape=(int(seqLen/dvs[3]), numDiffHits))
            #reshape5 = Reshape((1, int(seqLen / 16), numDiffHits))(in5)
            #conv5 = TimeDistributed(Conv1D(64, 4, activation='elu'), input_shape=(1,) + (seqLen, numDiffHits))(reshape5)
            #flat5 = TimeDistributed(Flatten())(conv5)
            #drop5 = TimeDistributed(Dropout(0.85))(flat5)
            mgu5 = (MGU(int(layerSize), activation='elu', return_sequences=ret_seq, dropout=drop, recurrent_dropout=0,
                       implementation=1, unroll=unroll, use_bias=use_bias))(in5)
            #dense5 = Dense(layerSize*10, activation='elu')(mgu5)

            #Merging
            merged=Add()([mgu1,mgu2,mgu3,mgu4,mgu5])
            #bn=BatchNormalization(momentum=.5)(merged)
            #reshape = Reshape((128,1))(bn)
            #conv=Conv1D(32,8,activation='elu')(reshape)
            #conv2 = Conv1D(32, 8, activation='elu')(conv)
            #flat = Flatten()(conv2)
            #dense = Dense(layerSize * 10, activation='elu')(bn)
            out=Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal")(merged)
            model = Model([in1, in2, in3, in4, in5], out)
        elif model_type=='time_dist_conv_mgu':
            #TimeDistributed version
            model.add(Reshape((1,seqLen, numDiffHits), input_shape=(seqLen, numDiffHits)))
            model.add(TimeDistributed(Conv1D(64,4, activation='elu'), input_shape=(1,)+(seqLen, numDiffHits)))
            #model.add(TimeDistributed(MaxPooling1D(3)))
            model.add(TimeDistributed(Flatten()))
            model.add(TimeDistributed(Dropout(0.5)))
            model.add(MGU(layerSize, activation='elu', return_sequences=False, dropout=0.5, recurrent_dropout=0.5,
                          implementation=1))
            #model.add(BatchNormalization(momentum=0.5))
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type == 'time_dist_mp_conv_mgu':
            # TimeDistributed version
            model.add(Reshape((1, seqLen, numDiffHits), input_shape=(seqLen, numDiffHits)))
            model.add(TimeDistributed(Conv1D(64, 4, activation='elu'), input_shape=(1,) + (seqLen, numDiffHits)))
            model.add(TimeDistributed(MaxPooling1D(3)))
            model.add(TimeDistributed(Flatten()))
            model.add(TimeDistributed(Dropout(0.4)))
            model.add(MGU(layerSize, activation='elu', return_sequences=False, dropout=0.4, recurrent_dropout=0.4,
                          implementation=1))
            # model.add(BatchNormalization(momentum=0.5))
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type=='single_mgu':
            model.add(MGU(layerSize, activation='elu',input_shape=(seqLen, numDiffHits),  # kernel_initializer='lecun_normal',
                      return_sequences=False, dropout=0.15, recurrent_dropout=0.15,implementation=1))
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type=='single_gru':
            model.add(GRU(layerSize, activation='elu',input_shape=(seqLen, numDiffHits),  # kernel_initializer='lecun_normal',
                      return_sequences=False, dropout=0.15, recurrent_dropout=0.15,implementation=1))
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type == 'single_lstm':
            model.add(LSTM(layerSize, activation='elu', input_shape=(seqLen, numDiffHits),
                          # kernel_initializer='lecun_normal',
                          return_sequences=False, dropout=0.05, recurrent_dropout=0.05, implementation=1))
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))


        print(model.summary())
        optr = nadam(lr=0.003)
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=optr)
        if destroy_old:
            model.save_weights("{}weights_testivedot_{}.hdf5".format(filePath, model_type))
            model.save('{}model_{}.hdf5'.format(filePath,model_type))
    global graph
    graph = tf.get_default_graph()

    #writer = tf.summary.FileWriter('./graphs/', tf.get_default_graph())
    return model

def train(filename=None, seqLen=seqLen, sampleMul=1., forceGen=False, bigFile=None, updateModel=False, model_type='parallel_mgu'):
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
            self.accs=[]
            self.val_losses=[]
            self.val_accs=[]
            self.hist=[]

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accs.append(logs.get('acc'))
        def on_epoch_end(self, epoch, logs={}):
            self.hist.append([logs.get('loss'),logs.get('acc'),logs.get('val_loss'),logs.get('val_acc')])

    modelsaver = ModelCheckpoint(filepath="./Kits/weights_testivedot_{}.hdf5".format(model_type), verbose=1, save_best_only=True)
    temporarysaver = ModelCheckpoint(filepath="./Kits/temp.hdf5", verbose=0, save_best_only=False)
    genMdelSaver=ModelCheckpoint(filepath="./Kits/weights_testivedot_ext.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=5, mode='auto')
    history = LossHistory()
    learninratescheduler=LearningRateScheduler(klik, verbose=1)
    if filename is not None and updateModel is not 'generator':
        X_train, y_train, numDiffHits = vectorizeCSV(filename, seqLen, sampleMul, forceGen=forceGen)
        if X_train is None:
            return None

    elif bigFile is not None and updateModel is not 'generator':
        #X_train, y_train, numDiffHits = vectorizeCSV(filename, seqLen, sampleMul, forceGen=forceGen, bigFile=bigFile)
        X_train,y_train=resample(np.array(bigFile[0]), np.array(bigFile[1]), n_samples=bigFile[1].shape[0]*sampleMul, replace=True)
        if X_train is None:
            return None
    elif updateModel is not 'generator':
        #print(len(BigX))
        X_train, y_train=resample(np.array(BigX), np.array(BigY), n_samples=len(BigX), replace=True)

    if updateModel is 'generator':
        def myGenerator():
            while True:
                name=np.random.randint(0,len(filename)-1)
                x, y,_ = vectorizeCSV(filename=filename[name+1], seqLen=seqLen, sampleMul=1, generator=20)
                if model_type=='parallel_mgu':
                    X_train2 = x[:, -int(seqLen / dvs[0]):, :]
                    X_train3 = x[:, -int(seqLen / dvs[1]):, :]
                    X_train4 = x[:, -int(seqLen / dvs[2]):, :]
                    X_train5 = x[:, -int(seqLen / dvs[3]):, :]
                    yield([x,X_train2, X_train3,X_train4,X_train5],y)
                else:
                    yield (x, y)

        X_test, y_test,_ = vectorizeCSV(filename=filename[0], seqLen=seqLen, sampleMul=.1)
        X_train=X_test
        tr_gen = myGenerator()

    if model_type=='parallel_mgu' or model_type=='parallel_conc_mgu':
        X_train2 = X_train[:, -int(seqLen / dvs[0]):, :]
        X_train3 = X_train[:, -int(seqLen / dvs[1]):, :]
        X_train4 = X_train[:, -int(seqLen / dvs[2]):, :]
        X_train5 = X_train[:, -int(seqLen / dvs[3]):, :]
        X_train_comp=[X_train,X_train2, X_train3,X_train4, X_train5]
    else:
        X_train_comp=X_train
    if forceGen:
        return X_train[0]
    # model=getModel()
    # model.fit_generator(generator=tr_gen, steps_per_epoch=200, max_queue_size=10,callbacks=[modelsaver, earlystopper],
    #                    workers=8, use_multiprocessing=True, verbose=1)

    with graph.as_default():
        #model=getModel()
        if updateModel=='generator':
            #model.save_weights("./Kits/weights_testivedot_ext.hdf5")
            model.fit_generator(generator=tr_gen,epochs=20, steps_per_epoch=10, max_queue_size=10,callbacks=[genMdelSaver, earlystopper],
                              workers=1, use_multiprocessing=False, verbose=2, validation_data=(X_train_comp, y_test))
            model.load_weights("./Kits/weights_testivedot_{}.hdf5".format(model_type))
            model.save('./Kits/model_{}.hdf5'.format(model_type))
        if updateModel==True:
            model.fit(X_train_comp, y_train, batch_size=int(max([y_train.shape[0]/50,25])), epochs=100,
                  callbacks=[modelsaver, earlystopper, history],# learninratescheduler],
                  validation_split=0.33,
                  verbose=2)
            lastLoss = np.mean(history.losses[-10:])
            model.load_weights("./Kits/weights_testivedot_{}.hdf5".format(model_type))
            model.save('./Kits/model_{}.hdf5'.format(model_type))
        elif updateModel==False:
            model.load_weights("./Kits/weights_testivedot_{}.hdf5".format(model_type))
            model.fit(X_train_comp, y_train, batch_size=int(max([y_train.shape[0]/50,25])), epochs=20,
                      callbacks=[temporarysaver,earlystopper,history],  # learninratescheduler],
                      validation_split=(1/3.),
                      verbose=2)
            #take mean of recent iteration losses for fuzz scaler
            lastLoss=np.mean(history.losses[-10:])
            print(lastLoss)
            model.load_weights("./Kits/temp.hdf5")
        #model.save_weights("./Kits/weights_testivedot2.hdf5")
    return X_train[0],history.hist

def generatePart(data,partLength=123, temp=None, include_seed=False, model_type='parallel_mgu'):

    print(data.shape)
    seed = data
    data = seed

    print('generating new sequence.')

    generated=[]

    #save seed
    if include_seed:
        for i in data:
            try:
                pass
                generated.append(Ichar.get(np.where(i==True)[0][0],0))
            except Exception as e:
                print('gen-init: ', e)
                pass

    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.asarray(a).astype('float64')
        a=a ** (1 / temperature)
        a_sum=a.sum()
        a=a/a_sum
        return np.argmax(np.random.multinomial(1, a, 1))

        #same from keras examples:
        #a = np.log(a) / temperature
        #a = np.exp(a) / np.sum(np.exp(a))
        # choices = range(len(a))
        # return np.random.choice(choices, p=a)


    if temp is -1:
        fuzz = np.max([1.5 - lastLoss, 0.1])
    else:
        fuzz = temp
    print('fuzz factor:',fuzz)
    print(data.shape)
    for i in range(partLength):

        data = data.reshape(1, seqLen, numDiffHits)
        if model_type=='parallel_mgu' or model_type=='parallel_conc_mgu':
            datas=[data, data[:,-int(seqLen/dvs[0]):,:],data[:,-int(seqLen/dvs[1]):,:],data[:,-int(seqLen/dvs[2]):,:],data[:,-int(seqLen/dvs[3]):,:]]
        else :
            datas=[data]

        with graph.as_default():
            pred = model.predict(datas, verbose=0)

        next_index = sample(pred[0],fuzz)
        #next_index = np.argmax(pred)
        next_char = Ichar.get(next_index,0)
        generated.append(next_char)

        next = np.zeros((numDiffHits,), dtype=bool)
        next[next_index] = True
        next = next.reshape(1, 1, next.shape[0])
        data = np.concatenate((data[:, 1:, :], next), axis=1)



    gen = pd.DataFrame(generated, columns=['inst'])
    filename = 'generated.csv'
    #filename='generated.csv'
    gen.to_csv(filename, index=True, header=None, sep='\t')
    print('valmis')
    #return filename
    # change to time and midinotes
    generated = splitrowsanddecode(generated)
    gen = pd.DataFrame(generated, columns=['time', 'inst'])
    #Cut over 30s.
    #maxFrames=300/(1/SAMPLE_RATE*Q_HOP)
    #gen = gen[gen['time'] < maxFrames]
    gen['time'] = frame_to_time(gen['time'], hop_length=HOP_SIZE)
    gen['inst'] = to_midinote(gen['inst'])
    gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
    gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))

    madmom.io.midi.write_midi(gen.values, 'generated_{}.mid'.format(model_type))



#####################################
#Testing from here on end
#make midi from source file
def debug():
    if False:
        generated=pd.read_csv('./midi_data_set/mididata9.csv', header=None, sep="\t", usecols=[1])
        print(generated.head())
        generated = splitrowsanddecode(generated[1])
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
            d = pd.read_csv(i, header=None, sep="\t").values
            data.extend(list(d[:, 1]))
        #d = pd.read_csv(vocab, header=None, sep="\t").values
        #data.extend(list(d[:, 1]))
        #huge=pd.DataFrame(data)
        #huge.to_csv('./huge.csv', index=True, header=None, sep='\t')
        counts = Counter(data)
        #Thow out rare hits
        vals=[x for x, _ in counts.most_common(256)]
        newVocab=list(set(vals))
        newVocab = pd.DataFrame(newVocab)
        filename='newVocab.csv'
        newVocab.to_csv(filename, index=True, header=None, sep='\t')
        return filename



    ##vectorizeCSV('./Kits/Default/takes/testbeat1535385910.271116.csv')
    #
    ##print(takes)
    if False:
        takes = [f for f in os.listdir('./Kits/birchcustom/takes/') if not f.startswith('.')]
        takes.sort()
        data = []
        filename=rebuild_vocabulary('./newVocab.csv', takes)

        buildVocabulary(filename)
        trainingRuns=1
        xss = []
        yss = []
        for j in range(trainingRuns):
            for i in takes[:25]:
                #print(i)
                X,y, _=vectorizeCSV('./Kits/birchcustom/takes/{}'.format(i),seqLen, sampleMul=1)
                if X is None:
                    continue
                xss.append(X)
                yss.append(y)
            Xs = np.concatenate((xss))
            ys = np.concatenate((yss))
            print(Xs.shape, ys.shape)
            train('./Kits/mcd2/takes/{}'.format(i),seqLen=seqLen,sampleMul=4)
            #d = pd.read_csv('./Kits/mcd2/takes/{}'.format(i), header=None, sep="\t").values
            #data.extend(list(d[:, 1]))
            #train(filename=None, seqLen=seqLen, sampleMul=4, bigFile=data, updateModel=True)
            print(len(Xs))
            #train(filename=None,seqLen=seqLen,sampleMul=4, bigFile=Xs, updateModel=True)
        print('training a model from scratch:%0.2f' % (time() - t0))
    #t0=time()
    #print('going big')
    #train('testbeat0.csv', seqLen, sampleMul=1.5, forceGen=False, updateModel='extreme')
    if False:
        t0=time()
        takes = [f for f in os.listdir('./Kits/mcd2/takes/') if not f.startswith('.')]
        filename = rebuild_vocabulary('./newVocab.csv', takes)
        buildVocabulary(filename)
        xss = []
        yss = []
        for i in takes:
            # print(i)
            X, y, _ = vectorizeCSV('./Kits/mcd2/takes/{}'.format(i), seqLen, sampleMul=1)
            if X is None:
                continue
            xss.append(X)
            yss.append(y)
        generatePart(train(filename=None, seqLen=seqLen, sampleMul=1,forceGen=False, bigFile=[np.concatenate((xss)),np.concatenate((yss))] , updateModel=True), temp=1.)
        #for i in takes:
            #train('./Kits/mcd2/takes/{}'.format(i), seqLen, sampleMul=4, forceGen=False, updateModel=True)
        print('training a model from scratch:%0.2f' % (time() - t0))
    #t0=time()
    #generatePart(train('testbeat2.csv', seqLen, sampleMul=1, forceGen=False, updateModel=True),temp=1.)
    #print('run:%0.2f' % (time() - t0))
    files=['../../midi_data_set/mididata0.csv',
           '../../midi_data_set/mididata1.csv',]
           #'../../midi_data_set/mididata2.csv',
           #'../../midi_data_set/mididata3.csv',
           #'../../midi_data_set/mididata4.csv',
           #'../../midi_data_set/mididata5.csv',
           #'../../midi_data_set/mididata6.csv',
           #'../../midi_data_set/mididata7.csv',
           #'../../midi_data_set/mididata8.csv',
           #'../../midi_data_set/mididata9.csv']
    #rebuild_vocabulary(newData=files)
    buildVocabulary('./newVocab.csv')
    new = False
    #model_type= 'time_dist_conv_mgu2'#'time_dist_conv_mgu','conv_mgu','parallel_mgu', 'single_mgu', 'parallel_conc_mgu'
    seqLen=64
    #model = initModel(seqLen, destroy_old=new, model_type=model_type)
    #model.load_weights("./Kits/weights_testivedot_{}.hdf5".format(model_type))

    #generatePart(train('../../midi_data_set/mididata9.csv', seqLen, sampleMul=.1, forceGen=True,
    #                   updateModel=True, model_type=model_type), partLength=666, temp=1., include_seed=True,
    #             model_type=model_type)
    #return True
    sampleMul=1
    all=False
    logs=[]
    #success: 'single_mgu','parallel_mgu'
    for i in ['time_dist_conv_mgu','time_dist_mp_conv_mgu','conv_mgu','single_gru','single_lstm']:
        model_type=i
        print('Starting: {}'.format(i))
        model = initModel(seqLen, destroy_old=True, model_type=model_type)
        t0 = time()
        #train(files,seqLen=seqLen, sampleMul=1.,updateModel='generator', model_type=model_type, forceGen=False)

        if all:
            print('../../midi_data_set/mididata0.csv')
            train('../../midi_data_set/mididata0.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata1.csv')
            train('../../midi_data_set/mididata1.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata2.csv')
            train('../../midi_data_set/mididata2.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata3.csv')
            train('../../midi_data_set/mididata3.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata4.csv')
            train('../../midi_data_set/mididata4.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata5.csv')
            train('../../midi_data_set/mididata5.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata6.csv')
            train('../../midi_data_set/mididata6.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata7.csv')
            train('../../midi_data_set/mididata7.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata8.csv')
            train('../../midi_data_set/mididata8.csv', seqLen, sampleMul=sampleMul, forceGen=False, updateModel=True,model_type=model_type)
            print('../../midi_data_set/mididata9.csv')
        _,log=train('../../midi_data_set/mididata0.csv', seqLen, sampleMul=sampleMul, forceGen=False,
                                   updateModel=True, model_type=model_type)
        logs.append(log)
        print(log)
        seed =train('../../midi_data_set/mididata0.csv', seqLen, sampleMul=.1, forceGen=True,
                           updateModel=True, model_type=model_type)
        generatePart(seed, partLength=666, temp=1., include_seed=True,model_type=model_type)
        print('run:%0.2f' % (time() - t0))
    pickle.dump(logs, open("{}/logs_full.log".format('.'), 'wb'))
    print(logs)
    #print('run:%0.2f' % (time() - t0))
    #print('gen:%0.2f' % (time() - t0))'../../midi_data_set/mididata0.csv'
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

    #
    # # Just the hits: Not working
    # def vectorize(filename, seqLen=32, sampleMul=1., forceGen=False):
    #     data = []
    #     d = pd.read_csv(filename, header=None, sep="\t").values
    #     data.extend(list(d[:, 1]))
    #
    #     def binarize(a):
    #         if a < 0:
    #             return map(int, ['1', '0'] + list(format(a, "0{}b".format(nrOfDrums)))[1:])
    #         else:
    #             return map(int, list(format(a, "0{}b".format(nrOfDrums + 1))))
    #
    #     def binarize2(a):
    #         return [x for x in bin(a)[2:]]
    #
    #     data = [list(binarize(i)) for i in data]
    #     #print(data)
    #     words = []
    #     outchar = []
    #     #print('corpus length:', len(data))
    #     for i in range(0, len(data) - seqLen, 1):
    #         words.append(data[i: i + seqLen])
    #         outchar.append(data[i + seqLen])
    #     #print('nb sequences:', len(words))
    #     X = np.array(words)
    #     y = np.array(outchar)
    #     X, y = resample(np.array(X), np.array(y), n_samples=len(words), replace=True)
    #     return X, y, numDiffHits
    #
    # def label_encode(filename, seqLen=32, sampleMul=1., forceGen=False):
    #     'Returns a DataFrame with encoded columns'
    #     data = []
    #     d = pd.read_csv(filename, header=None, sep="\t").values
    #     data.extend(list(d[:, 1]))
    #     data=pd.Series(data)
    #     factorised = pd.factorize(d[:, 1])[1]
    #     labels = pd.Series(range(len(factorised)), index=factorised)
    #     encoded_col = data.map(labels)
    #     encoded_col[encoded_col.isnull()] = -1
    #     words = []
    #     outchar = []
    #     #print('corpus length:', len(encoded_col))
    #     for i in range(0, len(encoded_col) - seqLen, 1):
    #         words.append(encoded_col[i: i + seqLen])
    #         outchar.append(encoded_col[i + seqLen])
    #     #print('nb sequences:', len(words))
    #     X = np.array(words)
    #     y = np.array(outchar)
    #     X, y = resample(np.array(X), np.array(y), n_samples=len(words)*sampleMul, replace=True)
    #     return X, y, numDiffHits
    # def freq_encode(filename, seqLen=32, sampleMul=1., forceGen=False):
    #     '''Returns a DataFrame with encoded columns'''
    #     data = []
    #     d = pd.read_csv(filename, header=None, sep="\t").values
    #     data.extend(list(d[:, 1]))
    #     data=pd.Series(data)
    #     freqs_cat = data.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
    #     #freqs_cat = data.count()/data.shape[0]
    #     encoded_col = data.map(freqs_cat)
    #     encoded_col[encoded_col.isnull()] = 0
    #     words = []
    #     outchar = []
    #     #print('corpus length:', len(encoded_col))
    #     for i in range(0, len(encoded_col) - seqLen, 1):
    #         words.append(encoded_col[i: i + seqLen])
    #         outchar.append(encoded_col[i + seqLen])
    #     #print('nb sequences:', len(words))
    #     X = np.array(words)
    #     y = np.array(outchar)
    #     X, y = resample(np.array(X), np.array(y), n_samples=len(words) * sampleMul, replace=True)
    #     return X, y, numDiffHits
    #
    # def keras_encode(filename, seqLen=32, sampleMul=1., forceGen=False):
    #     'Returns a DataFrame with encoded columns'
    #     data = []
    #     d = pd.read_csv(filename, header=None, sep="\t").values
    #     data.extend(list(d[:, 1]))
    #     #data=pd.Series(data)
    #     encoded_col = to_categorical(data,numDiffHits*2)
    #     #encoded_col[encoded_col.isnull()] = -1
    #     words = []
    #     outchar = []
    #     #print('corpus length:', len(encoded_col))
    #     for i in range(0, len(encoded_col) - seqLen, 1):
    #         words.append(encoded_col[i: i + seqLen])
    #         outchar.append(encoded_col[i + seqLen])
    #     #print('nb sequences:', len(words))
    #     X = np.array(words)
    #     y = np.array(outchar)
    #     X, y = resample(np.array(X), np.array(y), n_samples=len(words)*sampleMul, replace=True)
    #     return X, y, numDiffHits
if __name__ == "__main__":
    debug()