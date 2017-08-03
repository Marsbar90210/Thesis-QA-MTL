from data import _seed
import numpy as np
np.random.seed(_seed)
import tensorflow as tf
tf.set_random_seed(_seed)
import sys
from keras.layers.embeddings import Embedding
from keras.layers import Dense, merge, Dropout, RepeatVector
from keras.layers import recurrent, Input, Flatten
from keras.models import Model
from keras.initializers import RandomNormal, Orthogonal
from experiment import evaluate_model, handlesysargs
from model import model as modelwrapper

print("STL model based on GRU")

INITIALIZER = RandomNormal(mean=0.0, stddev=0.05, seed=_seed)
RINITIALIZER = Orthogonal(gain=1.0, seed=_seed)
RNN = recurrent.GRU
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100

def compile_model(inputs, repeat):
    (vocab_size, story_maxlen, query_maxlen) = inputs[0]
    
    sentinp = Input((story_maxlen,))
    #sentrnn = Embedding(vocab_size, EMBED_HIDDEN_SIZE,
    #                    input_length=story_maxlen,
    #                    embeddings_initializer=INITIALIZER)(sentinp)
    #sentrnn = Dropout(0.3, seed=_seed)(sentrnn)
    sentrnn = RepeatVector(query_maxlen)(sentinp)
    sentrnn = Flatten()(sentrnn)

    qinput = Input((query_maxlen,))
    #qrnn = Embedding(vocab_size, EMBED_HIDDEN_SIZE,
    #                   input_length=query_maxlen,
    #                   embeddings_initializer=INITIALIZER)(qinput)
    #qrnn = Dropout(0.3, seed=_seed)(qrnn)
    #qrnn = RNN(query_maxlen, return_sequences=False,
    #           kernel_initializer=INITIALIZER,
    #           bias_initializer=INITIALIZER,
    #           recurrent_initializer=RINITIALIZER)(qinput)
    qrnn = RepeatVector(story_maxlen)(qinput)
    qrnn = Flatten()(qrnn)
    
    out = merge([sentrnn, qrnn], mode='sum')
    #out = RNN(EMBED_HIDDEN_SIZE, return_sequences=False,
    #          kernel_initializer=INITIALIZER,
    #          bias_initializer=INITIALIZER,
    #          recurrent_initializer=RINITIALIZER)(out)
    #out = Dropout(0.3, seed=_seed)(out)
    out = Dense(vocab_size,
                kernel_initializer=INITIALIZER,
                activation='softmax')(out)

    model = Model([sentinp, qinput],[out])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return modelwrapper(model)

ver = 'en'
qas = None
ver, qas, repeat = handlesysargs(ver, qas)
evaluate_model(compile_model, ver, qas, repeat=repeat);
