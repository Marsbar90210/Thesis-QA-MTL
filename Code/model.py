from data import _seed
import numpy as np
np.random.seed(_seed)
import tensorflow as tf
tf.set_random_seed(_seed)
import sys, tempfile, string, os
from keras.models import load_model

# Model wrapper
class model:
    def __init__(self,
                 keras_model=None,
                 fitf=None,
                 testf=None,
                 predictf=None,
                 savef=None,
                 restoref=None):
        self.__keras_model = keras_model
        self.__fit_func = fitf
        self.__test_func = testf
        self.__predict_func = predictf
        self.__save_func = savef
        self.__restore_func = restoref
        self.__restore_points = []
    
    def fit(self, x, y, e):
        if self.__keras_model != None:
            return self.__keras_model.train_on_batch(x, y)
        # No keras model - use tf
        return self.__fit_func(x, y, e)
        
    def test(self, x, y):
        if self.__keras_model != None:
            return self.__keras_model.test_on_batch(x, y)
        # No keras model - use tf
        return self.__test_func(x, y)
    
    def predict(self, x):
        if self.__keras_model != None:
            res = self.__keras_model.predict_on_batch(x)
        else:
            # No keras model - use tf
            res = self.__predict_func(x)
        return res
    
    def save(self):
        fp = tempfile.gettempdir() + '/' + ''.join(np.random.choice(list(string.ascii_letters + string.digits), 20))
        if self.__save_func:
            self.__restore_points.append(self.__save_func(fp))
        else:
            # No tensorflow function - use keras
            fp = fp + '.h5'
            self.__keras_model.save(fp)
            self.__restore_points.append(fp)
    
    def restore(self, ri):
        if self.__restore_func:
            self.__restore_func(self.__restore_points[-ri])
            return True
        else:
            self.__keras_model = load_model(self.__restore_points[-ri])
            return True
        return False
    
#    def inputs(self):
#        if self.__keras_model != None:
#            return self.__keras_model.inputs
#        return self.__inputs
#    
#    def outputs(self):
#        if self.__keras_model != None:
#            return self.__keras_model.outputs
#        return self.__outputs
