'''
This module contains sequence learning and generating related functionality
'''

import utils
import constants
from time import time
from random import randint
import pickle
import os
import pandas as pd
import drumsynth
from sklearn.utils import resample, class_weight
from MGU import MGU
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,Callback, TensorBoard
from keras.models import Model, Sequential,load_model
from keras.layers import *#,Dense,Conv1D,Flatten,TimeDistributed,Input,  MaxPooling1D,GlobalAveragePooling1D,GRU, BatchNormalization, GRUCell, Dropout, TimeDistributed, Reshape, LSTM, Activation
from keras.utils import Sequence,to_categorical
from keras import regularizers
from keras.constraints import max_norm
from keras.optimizers import *#nadam
from collections import Counter
import tcn
#Fix seed, comment for random operation
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)
#Force cpu processing
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#sys.setrecursionlimit(10000)
t0 = time()
seqLen=16
#parallel MGU divisors
dvs=[1, 2, 4, 8, 16, 32]
#dvs=[1,4,16,64]
partLength = 0
lastLoss=0
lr=.002
CurrentKitPath='./Kits/'
Ichar = {}
charI = {}

def buildVocabulary(filename=None, hits=None):
    """
    Create a forvard and backward transformation keys for available vocabulary
    :param filename: String, .csv file where vocabulary is extracted from
    :param hits: list, list of possible hits on a drumkit
    :return: None, we use global variables for now (change this later!)
    """
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
    print('vocabulary size: ',len(charI))
    print('vocabulary built')

def getVocabulary():
    global numDiffHits, charI, Ichar, partLength
    return Ichar, charI

global model, graph

def get_sequences(filename, seqLen=64, sampleMul=1., forceGen=False):
    #If you use this you must use Embedding layer and sparse_categorical_crossentropy
    # i.e. model.add(Embedding(numDiffHits+1,numDiffHits, input_length=seqLen))
    data = []
    d = pd.read_csv(filename, header=None, sep="\t").values
    data.extend(list(d[:, 1]))
    dataI=[]
    for i in range(len(data)):
        try:
            dataI.append(charI[data[i]])
        except:
            pass
    words = []
    outchar = []
    for i in range(0, len(dataI) - seqLen, 1):
     words.append(dataI[i: i + seqLen])
     outchar.append(dataI[i + seqLen])
    X = np.array(words)
    y = np.array(outchar)
    samples = np.max([int(len(words) * sampleMul), 33])
    if forceGen is not 'game':
        X, y = resample(np.array(X), np.array(y),n_samples=samples, replace=True, random_state=2)
    return X, y, numDiffHits

def setLr(new_lr):
    global lr
    print('new learning rate:',new_lr)
    lr=new_lr


def initModel(seqLen=16,kitPath=None, destroy_old=False, model_type='single_mgu'):
    global model, CurrentKitPath
    layerSize =16#numDiffHits #Layer size = number of categorical variables
    if kitPath is None:
        filePath='./Kits/'
    else:
        filePath=kitPath
    CurrentKitPath=filePath
    try:
        if destroy_old:
            pickle.dump(0, open(CurrentKitPath + '/initial_epoch.k', 'wb'))
            raise ValueError('old model destroyed!!!')
        else:
            model = load_model('{}model_{}.hdf5'.format(filePath, model_type), custom_objects={'MGU': MGU})
            optr = nadam(lr=0.001)
            model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optr)
    except Exception as e:
        print('Making new model')
        model = Sequential()

        if model_type=='conv_mgu':
            #Conv1D before MGU
            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(Conv1D(32,2, activation='elu', input_shape=(seqLen, numDiffHits)))
            model.add(Dropout(0.5))
            model.add(MGU(layerSize, activation='tanh', return_sequences=False, dropout=0.5, recurrent_dropout=0.5,
                          implementation=1))
            model.add(BatchNormalization())
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type=='autoencoder':
            ###NOT FINISHED DO NOT TRY!!
            in1 = Input(shape=(seqLen,))
            em1 = Embedding(numDiffHits + 1, numDiffHits)(in1)
            encoder=MGU(layerSize, activation='tanh', input_shape=(seqLen, numDiffHits),
                return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1)(em1)
            decoder2 = RepeatVector(seqLen-1)(encoder)
            decoder2 = MGU(layerSize, activation='tanh', input_shape=(seqLen, numDiffHits),
                           return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1)(decoder2)
            bn = BatchNormalization()(decoder2)
            out2 = Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal")(bn)

            #enc1=Dense(640, activation="sigmoid")(in1)
            #enc2=Dense(32, activation="relu")(enc1)
            #enc3=Dense(16, activation="relu")(enc2)
            #drop1=Dropout(0.0)(enc3)
            #dec0=Dense(16, activation="relu")(drop1)
            #dec1=Dense(32, activation="relu")(dec0)
            #dec2=Dense(640, activation="sigmoid")(dec1)
            ##flat=GlobalAveragePooling1D()(dec2)
            #out = Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal")(dec2)
            model=Model(inputs=in1, outputs=[out2])
        elif model_type=='many2many':
            ###NOT FINISHED DO NOT TRY!!
            X=Input(shape=(seqLen,))
            embedding_layer = Embedding(numDiffHits + 1, numDiffHits)
            reshape_layer=Reshape((1, numDiffHits))
            mgu_layer=MGU(128, activation='tanh', return_sequences=False, dropout=0.0, recurrent_dropout=0.0,
                        implementation=1)
            out_layer=Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal")
            output_layers = []
            a0 = Input(shape=(layerSize,), name='a0')
            c0 = Input(shape=(layerSize,), name='c0')
            a = a0
            c = c0
            for i in range(seqLen):
                x = Lambda(lambda x:X[:,i])(X)
                emb1= embedding_layer(x)
                res1 = reshape_layer(emb1)
                mgu1= mgu_layer(res1)
                out1=out_layer(mgu1)
                output_layers.append(out1)
            model = Model(inputs=[X],outputs=output_layers)

        elif model_type=='ATT_TDC_P_mgu':
            drop=0.7
            cdrop=0.7
            sdrop=0.7
            n_kernels=4
            unroll=False
            use_bias=False
            ret_seq=True
            regVal=0.
            regVal2=0.
            attention_on=True
            def get_layer(seqLen, n_kernels, n_win):
                in1 = Input(shape=(seqLen,))
                em1=Embedding(numDiffHits+1, numDiffHits)(in1)
                reshape1=Reshape((1,seqLen, numDiffHits))(em1)
                conv1=TimeDistributed(Conv1D(n_kernels,n_win, activation='elu',kernel_regularizer=regularizers.l2(regVal),
                    activity_regularizer=regularizers.l1(regVal2)), input_shape=(1,)+(seqLen, numDiffHits))(reshape1)

                #bn1=TimeDistributed(BatchNormalization())(conv1)
                sdrop1=TimeDistributed(SpatialDropout1D(sdrop))(conv1)
                flat1=TimeDistributed(Flatten())(sdrop1)
                #drop1=TimeDistributed(Dropout(cdrop))(flat1)
                mgu1=(MGU(layerSize, activation='tanh', return_sequences=ret_seq, dropout=drop, recurrent_dropout=drop,
                              implementation=1, unroll=unroll, use_bias=use_bias))(flat1)
                if attention_on:
                    attention_r = Dense(layerSize, activation='softmax')(mgu1)
                    attention_r = Flatten()(attention_r)
                    # attention_r = Activation('softmax')(attention_r)
                    attention_r = RepeatVector(layerSize)(attention_r)
                    attention_r = Permute([2, 1])(attention_r)
                    mgu1 = Multiply()([mgu1, attention_r])
                return [in1, mgu1]

            layers=[]
            for i in dvs:
                layers.append(get_layer(int(seqLen/i),n_kernels, min(8,int(seqLen/i))))
            #Merging
            layers=np.array(layers)

            merged=Concatenate()(list(layers[:,1]))
            flat = Flatten()(merged)
            bn=BatchNormalization()(flat)
            out=Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal")(bn)
            model = Model(list(layers[:,0]), out)

        elif model_type=='TDC_parallel_mgu':
            drop=0.55 #.55 without attention
            cdrop=0.6
            sdrop=0.6 #.6 without attention
            layerSize=int(4) #4 without attention
            n_kernels=4
            unroll=False
            use_bias=False
            ret_seq=True
            regVal=0.
            regVal2=0.
            attention_on=False
            def get_layer(seqLen, n_kernels, n_win):
                in1 = Input(shape=(seqLen,))
                em1=Embedding(numDiffHits+1, numDiffHits)(in1)
                reshape1=Reshape((1,seqLen, numDiffHits))(em1)
                conv1=TimeDistributed(Conv1D(n_kernels,n_win, activation='elu',kernel_regularizer=regularizers.l2(regVal),
                    activity_regularizer=regularizers.l1(regVal2)), input_shape=(1,)+(seqLen, numDiffHits))(reshape1)

                #bn1=TimeDistributed(BatchNormalization())(conv1)
                sdrop1=TimeDistributed(SpatialDropout1D(sdrop))(conv1)
                flat1=TimeDistributed(Flatten())(sdrop1)
                #drop1=TimeDistributed(Dropout(cdrop))(flat1)
                mgu1=(MGU(layerSize, activation='tanh', return_sequences=ret_seq, dropout=drop, recurrent_dropout=drop,
                              implementation=1, unroll=unroll, use_bias=use_bias))(flat1)
                if attention_on:
                    attention_r = Dense(layerSize, activation='softmax')(mgu1)
                    attention_r = Flatten()(attention_r)
                    # attention_r = Activation('softmax')(attention_r)
                    attention_r = RepeatVector(layerSize)(attention_r)
                    attention_r = Permute([2, 1])(attention_r)
                    mgu1 = Multiply()([mgu1, attention_r])
                return [in1, mgu1]

            layers=[]
            for i in dvs:
                layers.append(get_layer(int(seqLen/i),n_kernels, min(8,int(seqLen/i))))
            #Merging
            layers=np.array(layers)

            merged=Concatenate()(list(layers[:,1]))
            flat = Flatten()(merged)
            bn=BatchNormalization()(flat)
            out=Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal")(bn)
            model = Model(list(layers[:,0]), out)

        elif model_type == 'parallel_mgu':
            drop = 0.7
            unroll = False
            use_bias = False
            ret_seq = False
            def get_layer(seqLen):
                in1 = Input(shape=(seqLen,))
                em1 = Embedding(numDiffHits + 1, numDiffHits)(in1)
                mgu1 = (MGU(layerSize, activation='tanh', return_sequences=ret_seq, dropout=drop, recurrent_dropout=drop,
                            implementation=1, unroll=unroll, use_bias=use_bias))(em1)
                return [in1, mgu1]

            layers = []
            for i in dvs:
                layers.append(get_layer(int(seqLen / i)))
            # Merging
            layers = np.array(layers)

            merged = Add()(list(layers[:, 1]))
            bn = BatchNormalization()(merged)
            out = Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal")(bn)
            model = Model(list(layers[:, 0]), out)
        elif model_type=='time_dist_conv_mgu':
            #TimeDistributed version
            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(Reshape((1,seqLen, numDiffHits), input_shape=(seqLen, numDiffHits)))
            model.add(TimeDistributed(Conv1D(32,2, activation='elu'), input_shape=(1,)+(seqLen, numDiffHits)))
            #model.add(TimeDistributed(MaxPooling1D(3)))
            model.add(TimeDistributed(Flatten()))
            model.add(TimeDistributed(Dropout(0.625)))
            model.add(MGU(layerSize, activation='tanh', return_sequences=False, dropout=0.6, recurrent_dropout=0.6,
                          implementation=1))
            model.add(BatchNormalization())
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type == 'time_dist_mp_conv_mgu':
            # TimeDistributed version
            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(Reshape((1, seqLen, numDiffHits), input_shape=(seqLen, numDiffHits)))
            model.add(TimeDistributed(Conv1D(64, 2, activation='elu'), input_shape=(1,) + (seqLen, numDiffHits)))
            model.add(TimeDistributed(MaxPooling1D(3)))
            model.add(TimeDistributed(Flatten()))
            model.add(TimeDistributed(Dropout(0.4)))
            model.add(MGU(layerSize, activation='elu', return_sequences=False, dropout=0.4, recurrent_dropout=0.4,
                          implementation=1))
            # model.add(BatchNormalization(momentum=0.5))
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type=='attention_mgu':
            att_input=False
            att_mgu=True
            #MGU block
            inputs = Input(shape=(seqLen,))
            emb1 = Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen)(inputs)
            res1=Reshape((1, seqLen, numDiffHits), input_shape=(seqLen, numDiffHits))(emb1)
            tim1=TimeDistributed(Conv1D(32, 2, activation='elu'), input_shape=(1,) + (seqLen, numDiffHits))(res1)
            # model.add(TimeDistributed(MaxPooling1D(3)))
            tim2=TimeDistributed(Flatten())(tim1)
            tim3=TimeDistributed(Dropout(0.625))(tim2)
            if att_input:
                input_dim = int(emb1.shape[2])
                print(emb1.shape)
                attention_i = Permute((2, 1))(emb1)
                attention_i = Reshape((input_dim, seqLen))(attention_i)
                attention_i = Dense(seqLen, activation='tanh')(attention_i)
                attention_i = Activation('softmax')(attention_i)
                attention_i = Permute([2, 1])(attention_i)
                attention_i_out = Multiply()([emb1, attention_i])
                emb1=attention_i_out
            mgu1 = (MGU(layerSize, activation='tanh', return_sequences=True, dropout=0.8, recurrent_dropout=0.8,
                        implementation=1))(tim3)
            if att_mgu:
                #Attention block
                attention_r = Dense(1, activation='softmax')(mgu1)
                attention_r = Flatten()(attention_r)
                #attention_r = Activation('softmax')(attention_r)
                attention_r = RepeatVector(layerSize)(attention_r)
                attention_r = Permute([2, 1])(attention_r)
                mgu1 = Multiply()([mgu1, attention_r])


            #attention_i = Permute([2, 1])(attention_i)

            #Multiply blocks
            flat = Flatten()(mgu1)
            bn = BatchNormalization()(flat)
            output = Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal")(bn)
            model = Model(input=[inputs], output=output)

        elif model_type=='single_mgu':

            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(MGU(layerSize, activation='tanh',input_shape=(seqLen, numDiffHits),  # kernel_initializer='lecun_normal',
                      return_sequences=False, dropout=0.6, recurrent_dropout=0.6,implementation=1))
            model.add(BatchNormalization())
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))

        elif model_type == 'stacked_mgu':
            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(MGU(layerSize, activation='tanh', input_shape=(seqLen, numDiffHits),
                          # kernel_initializer='lecun_normal',
                          return_sequences=True, dropout=0.0, recurrent_dropout=0.0, implementation=1))
            model.add(MGU(layerSize, activation='tanh', input_shape=(seqLen, numDiffHits),
                          # kernel_initializer='lecun_normal',
                          return_sequences=True, dropout=0.0, recurrent_dropout=0.0, implementation=1))
            model.add(MGU(layerSize, activation='tanh', input_shape=(seqLen, numDiffHits),
                          # kernel_initializer='lecun_normal',
                          return_sequences=False, dropout=0., recurrent_dropout=0., implementation=1))
            #model.add(BatchNormalization())
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type=='single_gru':
            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(GRU(layerSize, activation='tanh',input_shape=(seqLen, numDiffHits),  # kernel_initializer='lecun_normal',
                      return_sequences=False, dropout=0.65, recurrent_dropout=0.65,implementation=1))
            model.add(BatchNormalization())
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))
        elif model_type == 'single_lstm':
            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(LSTM(layerSize, activation='tanh', input_shape=(seqLen, numDiffHits),
                          # kernel_initializer='lecun_normal',
                          return_sequences=False, dropout=0.7, recurrent_dropout=0.7, implementation=1))
            model.add(BatchNormalization())
            model.add(Dense(numDiffHits, activation="softmax", kernel_initializer="he_normal"))

        elif model_type=='tcn':
            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(tcn.compiled_tcn(return_sequences=False,
                             num_feat=numDiffHits,
                             num_classes=numDiffHits,
                             nb_filters=32,#64/16 result from 64
                             kernel_size=2,#64/16 result from 8
                             dilations=[2 ** i for i in range(2)],#64/16 result from 3
                             nb_stacks=1,#64/16 result from 2
                             max_len=None,#64/16 result from 128
                             use_skip_connections=True,
                             dropout_rate=0.65))#64/16 result from 0.75
        #the markovian option
        elif model_type=='tcn_1':
            model.add(Embedding(numDiffHits + 1, numDiffHits, input_length=seqLen))
            model.add(tcn.compiled_tcn(return_sequences=False,
                             num_feat=numDiffHits,
                             num_classes=numDiffHits,
                             nb_filters=64,#64/16 result from 64
                             kernel_size=1,#64/16 result from 8
                             dilations=[2 ** i for i in range(3)],#64/16 result from 3
                             nb_stacks=1,#64/16 result from 2
                             max_len=None,#64/16 result from 128
                             use_skip_connections=True,
                             dropout_rate=0.5))#64/16 result from 0.75
        print(model.summary())
        optr = adam(lr=0.001)
        model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'], optimizer=optr)
        print('Saving new model....')
        model.save_weights("{}weights_testivedot_{}.hdf5".format(CurrentKitPath, model_type))
        #model.save('{}model_{}.hdf5'.format(CurrentKitPath,model_type))
    global graph
    graph = tf.get_default_graph()
    #writer = tf.summary.FileWriter('./graphs/', tf.get_default_graph())
    return model

def train(filename=None, seqLen=seqLen, sampleMul=1., forceGen=False, bigFile=None, updateModel=False, model_type='parallel_mgu', return_history=False):
    global lastLoss
    print(model_type)
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

    #Callbacks
    class Init_History(Callback):
        """
        Callback that records events into a `History` object.

        This callback is automatically applied to
        every Keras model. The `History` object
        gets returned by the `fit` method of models.
        """
        def on_train_begin(self, logs=None):
            if not hasattr(self, 'epoch'):
                self.epoch = []
                self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

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

            self.hist.append([logs.get('loss'),logs.get('acc'),logs.get('val_loss'),logs.get('val_acc'),epoch])

    modelsaver = ModelCheckpoint(filepath="{}weights_testivedot_{}.hdf5".format(CurrentKitPath,model_type), verbose=1, save_best_only=True)
    temporarysaver = ModelCheckpoint(filepath="{}temp.hdf5".format(CurrentKitPath), verbose=0, save_best_only=False)
    genMdelSaver=ModelCheckpoint(filepath="{}weights_testivedot_ext.hdf5".format(CurrentKitPath), verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=2, mode='auto')
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    history = LossHistory()
    learninratescheduler=LearningRateScheduler(klik, verbose=1)
    if filename is not None and updateModel is not 'generator':
        if updateModel is 'game':
            X_train, y_train, numDiffHits = get_sequences(filename, seqLen, sampleMul, forceGen='game')
        else:
            X_train, y_train, numDiffHits = get_sequences(filename, seqLen, sampleMul, forceGen=forceGen)
        if X_train is None:
            return None

    if updateModel is 'generator':
        def myGenerator():
            while True:
                name=np.random.randint(0,len(filename)-1)
                x, y,_ = get_sequences(filename=filename[name+1], seqLen=seqLen, sampleMul=1, generator=20)
                if model_type=='parallel_mgu':
                    X_train2 = x[:, -int(seqLen / dvs[0]):]
                    X_train3 = x[:, -int(seqLen / dvs[1]):]
                    X_train4 = x[:, -int(seqLen / dvs[2]):]
                    X_train5 = x[:, -int(seqLen / dvs[3]):]
                    yield([x,X_train2, X_train3,X_train4,X_train5],y)
                else:
                    yield (x, y)

        X_test, y_test,_ = get_sequences(filename=filename[0], seqLen=seqLen, sampleMul=.1)
        X_train=X_test
        tr_gen = myGenerator()

    if model_type in ['parallel_mgu','TDC_parallel_mgu','ATT_TDC_P_mgu']:
        X_train_comp =[]
        for i in dvs:
            X_train_comp.append(X_train[:, -int(seqLen / i):])

    else:
        X_train_comp=X_train
    if forceGen:
        return X_train[0]
    # model=getModel()
    # model.fit_generator(generator=tr_gen, steps_per_epoch=200, max_queue_size=10,callbacks=[modelsaver, earlystopper],
    #                    workers=8, use_multiprocessing=True, verbose=1)
    try:
        initial_epoch=pickle.load(open(CurrentKitPath+ '/initial_epoch.k', 'rb'))
    except:
        initial_epoch=0

    with graph.as_default():

        #Compute weigths
        class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)
#
        if updateModel=='generator':
            #model.save_weights("./Kits/weights_testivedot_ext.hdf5")
            model.fit_generator(generator=tr_gen,epochs=20, steps_per_epoch=10, max_queue_size=10,callbacks=[genMdelSaver, earlystopper],
                              workers=1, use_multiprocessing=False, verbose=2, validation_data=(X_train_comp, y_test))
            model.load_weights("{}weights_testivedot_{}.hdf5".format(CurrentKitPath,model_type))
            model.save('{}Kits/model_{}.hdf5'.format(CurrentKitPath,model_type))
        if updateModel=='game':
            model.fit(X_train_comp, y_train, batch_size=int(max([y_train.shape[0] / 50, 32])),
                      epochs=initial_epoch + 20,
                      callbacks=[modelsaver, history],  # learninratescheduler],earlystopper,
                      #validation_split=0.33,
                      verbose=2, initial_epoch=initial_epoch, class_weight=class_weights, shuffle=False)
            lastLoss = np.mean(history.losses[-10:])
            pickle.dump(initial_epoch + 20, open(CurrentKitPath + '/initial_epoch.k', 'wb'))
            model.save('{}model_{}.hdf5'.format(CurrentKitPath, model_type))


        if updateModel == True:
            # model.load_weights("{}weights_testivedot_{}.hdf5".format(CurrentKitPath,model_type))
            #
            model.fit(X_train_comp, y_train, batch_size=int(max([y_train.shape[0] / 50, 32])),
                      epochs=initial_epoch + 20,
                      callbacks=[modelsaver, earlystopper, history],  # learninratescheduler],earlystopper,
                      validation_split=0.33,
                      verbose=2, initial_epoch=initial_epoch, class_weight=class_weights, shuffle=False)
            lastLoss = np.mean(history.losses[-10:])
            pickle.dump(initial_epoch + 20, open(CurrentKitPath + '/initial_epoch.k', 'wb'))
            model.load_weights("{}weights_testivedot_{}.hdf5".format(CurrentKitPath, model_type))
            model.save('{}model_{}.hdf5'.format(CurrentKitPath, model_type))
            #
        if updateModel=='m2m':
            # a0 = np.zeros((X_train_comp.shape[0],16))
            print(X_train_comp.shape)

            # y_train=np.roll(X_train_comp,-1,1)
            y_comp = []
            y_comp.append(y_train)
            for i in range(seqLen - 1):
                y_comp.append(np.roll(y_comp[-1], 1, 0))

            # y_train=np.swapaxes(y_train, 0,1)
            # y_train = np.expand_dims(y_train, axis=-1)
            # y_train = np.expand_dims(y_train, axis=1)

            #model.load_weights("{}weights_testivedot_{}.hdf5".format(CurrentKitPath,model_type))
            #
            model.fit([X_train_comp],y_comp,batch_size=int(max([y_train.shape[0]/50,32])), epochs=initial_epoch+2000,
                  callbacks=[history],# learninratescheduler],earlystopper,
                  validation_split=0.33,
                  verbose=2, initial_epoch=initial_epoch, shuffle=False)
            lastLoss = np.mean(history.losses[-10:])
            pickle.dump(initial_epoch+20, open(CurrentKitPath+ '/initial_epoch.k', 'wb'))

        elif updateModel==False:
            model.fit(X_train_comp, y_train[0], batch_size=int(max([y_train.shape[0]/50,25])), epochs=20,
                      callbacks=[temporarysaver,earlystopper,history],  # learninratescheduler],
                      validation_split=(0.33),
                      verbose=2, class_weight=class_weights, shuffle=False)
            #take mean of recent iteration losses for fuzz scaler
            lastLoss=np.mean(history.losses[-10:])
            print(lastLoss)

    if return_history:
        return X_train[0],history.hist
    else:
        return X_train[0]#,history.hist

def generatePart(data,partLength=123,seqLen=16, temp=None, include_seed=False, model_type='parallel_mgu'):

    #print(data)
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
        if temperature<=0:
            print ('Use temperatures over freezing >0°')
            #return the most probable element to avoid errors
            return a[0]
        a = np.asarray(a).astype('float64')
        a=a ** (1 / temperature)
        a_sum=a.sum()
        a=a/a_sum

        #The other way from Sutton, R. S. and Barto A. G. Reinforcement Learning: An Introduction.
        #likes lower temperatures :)
        #a = np.log(a)/temperature
        #a = np.exp(a) / np.sum(np.exp(a))

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
    #print(data.shape)
    for i in range(partLength):
        data = data.reshape(1, seqLen)
        datas=[]
        if model_type in ['parallel_mgu', 'TDC_parallel_mgu', 'ATT_TDC_P_mgu']:
            for i in dvs:
                datas.append(data[:,-int(seqLen/i):])
        else :
            datas=[data]
        with graph.as_default():
            pred = model.predict(datas, verbose=0)

        next_index = sample(pred[0],fuzz)
        next_char = Ichar.get(next_index,0)
        generated.append(next_char)
        data=np.concatenate((data[:,1:], [[next_index]]), axis=-1)

    gen = pd.DataFrame(generated, columns=['inst'])
    filename = 'generated.csv'
    gen.to_csv(filename, index=True, header=None, sep='\t')
    print('valmis')
    #return here, remove for testing
    return filename

    # change to time and midinotes
    generated = utils.splitrowsanddecode(generated)
    gen = pd.DataFrame(generated, columns=['time', 'inst'])
    #Cut over 30s.
    #maxFrames=300/(1/SAMPLE_RATE*Q_HOP)
    #gen = gen[gen['time'] < maxFrames]
    gen['time'] = frame_to_time(gen['time'], hop_length=constants.Q_HOP)
    gen['inst'] = to_midinote(gen['inst'])
    gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
    gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))
    madmom.io.midi.write_midi(gen.values, 'generated_{}.mid'.format(model_type))



#####################################
#Testing from here on end do not use unless absolutely necessary.
#make midi from source file
def debug():

    #d=pd.read_csv('../../midi_data_set/mididata8.csv', header=None, sep="\t").values
    #generated=list(d[:, 1])
    #generated = splitrowsanddecode(generated)
    #gen = pd.DataFrame(generated, columns=['time', 'inst'])
    ## Cut over 30s.
    ## maxFrames=300/(1/SAMPLE_RATE*Q_HOP)
    ## gen = gen[gen['time'] < maxFrames]
    #gen['time'] = frame_to_time(gen['time'], hop_length=HOP_SIZE)
    #gen['inst'] = to_midinote(gen['inst'])
    #gen['duration'] = pd.Series(np.full((len(generated)), 0, np.int64))
    #gen['vel'] = pd.Series(np.full((len(generated)), 127, np.int64))
#
    #madmom.io.midi.write_midi(gen.values, 'generated_{}.mid'.format(8))
    #return
    if False:
        generated=pd.read_csv('./midi_data_set/mididata9.csv', header=None, sep="\t", usecols=[1])
        print(generated.head())
        generated = utils.splitrowsanddecode(generated[1])
        gen = pd.DataFrame(generated, columns=['time', 'inst'])

        gen['time'] = utils.frame_to_time(gen['time'], hop_length=utils.Q_HOP)

        gen['inst'] = utils.to_midinote(gen['inst'])
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
    seqLen=64
    from keras.utils import plot_model
    if True:
        logs=[]
        times=[]
        #
        model_type=['TDC_parallel_mgu', 'time_dist_conv_mgu','parallel_mgu','stacked_mgu','conv_mgu','single_mgu',
                    'single_gru', 'single_lstm', 'tcn','ATT_TDC_P_mgu']
        model_type = ['ATT_TDC_P_mgu']
        buildVocabulary(hits=utils.get_possible_notes([0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 13]))
        for j in model_type:
            log=[]
            model = initModel(seqLen=seqLen, destroy_old=True, model_type=j)

            takes = ['./Kits/MDC_Stack/takes/{}'.format(f) for f in os.listdir('./Kits/MDC_Stack/takes/') if not f.startswith('.')]
            takes.sort()
            takes2 = ['./Kits/timedist/takes/{}'.format(f) for f in os.listdir('./Kits/timedist/takes/') if not f.startswith('.')]
            takes2.sort()
            data = []

            trainingRuns=1
            k=0
            t0=time()
            for i in takes2:
                print(i)

                t1 = time()
                seed, history=train(i,seqLen=seqLen,sampleMul=1,model_type=j,updateModel=True, return_history=True)

                file = generatePart(seed, partLength=333,seqLen=seqLen, temp=.666, include_seed=False, model_type=j)
                print('roundtrip time:%0.4f' % (time() - t1))
                #drumsynth.createWav(i, 'orig_temp_{}.wav'.format(j), addCountInAndCountOut=False,
                #                    deltaTempo=1,
                #                    countTempo=1)
                drumsynth.createWav(file, 'gen_temp_{}_{}.wav'.format(j,'0'), addCountInAndCountOut=False,
                                    deltaTempo=1,
                                    countTempo=1)
                #drumsynth.createWav(i, 'gen_temp_{}_{}.wav'.format('orig', k), addCountInAndCountOut=False,
                #                    deltaTempo=1,
                #                    countTempo=1)
                log.extend(history[:][0])
                log.extend(history[:][-1])
                k+=1

            for i in takes:
                print(i)
                seed, history=train(i,seqLen=seqLen,sampleMul=1,model_type=j,updateModel=True, return_history=True)
                file = generatePart(seed, partLength=333,seqLen=seqLen, temp=.666, include_seed=False, model_type=j)
                drumsynth.createWav(file, 'gen_temp_{}_{}.wav'.format(j, '0'), addCountInAndCountOut=False,
                                    deltaTempo=1,
                                    countTempo=1)
                #drumsynth.createWav(i, 'gen_temp_{}_{}.wav'.format('orig', k), addCountInAndCountOut=False,
                #                    deltaTempo=1,
                #                    countTempo=1)
                log.extend(history[:][0])
                log.extend(history[:][-1])
                k+=1

            logs.append(log)
            times.append(time() - t0)
            print('training a model from scratch:%0.2f' % (time() - t0))

    pickle.dump(logs, open("{}/logs_full_folder_complete_att64_8.log".format('.'), 'wb'))
    print('times')
    print(times)
    return
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
           '../../midi_data_set/mididata1.csv',
           '../../midi_data_set/mididata2.csv',
           '../../midi_data_set/mididata3.csv',
           '../../midi_data_set/mididata4.csv',
           '../../midi_data_set/mididata5.csv',
           '../../midi_data_set/mididata6.csv',
           '../../midi_data_set/mididata7.csv',
           '../../midi_data_set/mididata8.csv',
           '../../midi_data_set/mididata9.csv']
    rebuild_vocabulary(newData=files)
    buildVocabulary(hits=get_possible_notes([0,1,2,3,5,8,9,10,11,12]))
    #buildVocabulary('./newVocab.csv')
    new = False
    #model_type= 'time_dist_conv_mgu2'#'time_dist_conv_mgu','conv_mgu','parallel_mgu', 'single_mgu', 'parallel_conc_mgu'

    #model = initModel(seqLen, destroy_old=new, model_type=model_type)
    #model.load_weights("./Kits/weights_testivedot_{}.hdf5".format(model_type))

    #generatePart(train('../../midi_data_set/mididata9.csv', seqLen, sampleMul=.1, forceGen=True,
    #                   updateModel=True, model_type=model_type), partLength=666, temp=1., include_seed=True,
    #             model_type=model_type)
    #return True
    seqLen = 64
    sampleMul=1
    all=False
    logs=[]
    model_type = ['TDC_parallel_mgu', 'time_dist_conv_mgu', 'parallel_mgu', 'stacked_mgu', 'conv_mgu', 'single_mgu',
                  'single_gru', 'single_lstm', 'tcn']
    for i in model_type:
        print('Starting: {}'.format(i))
        t0 = time()
        log = []
        model = initModel(seqLen=seqLen, destroy_old=True, model_type=i)
        if all:
            for j in files:
                print(j)
                seed, history = train(j, seqLen=seqLen, sampleMul=1, model_type=i, updateModel=True,
                                      return_history=True)
                file = generatePart(seed, partLength=333, seqLen=seqLen, temp=1., include_seed=False, model_type=i)
                drumsynth.createWav(file, 'gen_temp_{}.wav'.format(i), addCountInAndCountOut=False,
                                    deltaTempo=1,
                                    countTempo=1)
                log.extend(history[:][0])
                log.extend(history[:][-1])
        logs.append(log)
        times.append(time() - t0)
        print('training a model from scratch:%0.2f' % (time() - t0))
    pickle.dump(logs, open("{}/logs_full_folder_complete_bigfiles_all16.log".format('.'), 'wb'))
    print('times')


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
# def vectorizeCSV(filename, seqLen=32, sampleMul=1., forceGen=False, bigFile=None, generator=False):
#     global numDiffHits, charI, Ichar, partLength
#     if filename is not None:
#         #Init with pauses before drumming starts
#         data = list(np.full(seqLen,-32))
#         d = pd.read_csv(filename, header=None, sep="\t").values
#         if bigFile is 'extreme':
#             target =randint(0, d[-1,0]-200)
#             data.extend(list(d[target:target + 200, 1]))
#         else:
#             data.extend(list(d[:, 1]))
#
#     if forceGen:
#         data=data[-180:]
#         #return
#     words = []
#     outchar = []
#     if generator:
#         for i in range(generator):
#             loc=np.random.randint(0, len(data) - seqLen)
#             words.append(data[loc: loc + seqLen])
#             outchar.append(data[loc + seqLen])
#     else:
#         for i in range(0, len(data) - seqLen, 1):
#             words.append(data[i: i + seqLen])
#             outchar.append(data[i + seqLen])
#
#     if len(words)<1:
#         return None, None, None
#
#     X = np.zeros((len(words), seqLen, numDiffHits), dtype=np.bool)
#     y = np.zeros((len(words), numDiffHits), dtype=np.bool)
#
#     for i, word in enumerate(words):
#         for t, char in enumerate(word):
#             # If we find a hit not in vocabulary of #numDiffHits we simply omit that
#             # Pitääkö järjestää??
#             try:
#                 X[i, t, charI[char]] = 1
#             except:
#                 #print(char)
#                 pass
#         try:
#             y[i, charI[outchar[i]]] = 1
#         except:
#             pass
#     if generator:
#         return np.array(X), np.array(y), numDiffHits
#
#     samples=np.max([int(len(words)*sampleMul),33])
#     X, y = resample(np.array(X), np.array(y), n_samples=samples, replace=True, random_state=2)
#     return X, y, numDiffHits

if __name__ == "__main__":
    debug()