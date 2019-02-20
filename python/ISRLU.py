from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf


class ISRLU(Activation):

    def __init__(self, activation, **kwargs):
        super(ISRLU, self).__init__(activation, **kwargs)
        self.__name__ = 'isrlu'

def isrlu(x, alpha=3):
    #return K.switch(x >= 0,x, x*((1+alpha*(x**2))**-.5))
    x=tf.where(x > 0,x, x*((1+alpha*x**2)**-.5))
    return x
get_custom_objects().update({'isrlu': ISRLU(isrlu)})


class ISRU(Activation):

    def __init__(self, activation, **kwargs):
        super(ISRU, self).__init__(activation, **kwargs)
        self.__name__ = 'isru'

def isru(x, alpha=3):
    #return K.switch(x >= 0,x, x*((1+alpha*(x**2))**-.5))
    return  x*((1+alpha*x**2)**-.5)
get_custom_objects().update({'isru': ISRLU(isru)})