from data import _seed
import numpy as np
np.random.seed(_seed)
import tensorflow as tf
tf.set_random_seed(_seed)
import sys
from keras.layers import Dense, merge, Dropout, RepeatVector, Embedding
from keras.layers import recurrent, Input
from keras.models import Model
from keras.initializers import RandomNormal, Orthogonal
from experiment import evaluate_model, handlesysargs
from model import model as modelwrapper

print("MTL model sharing rnn layer, based on gru")

INITIALIZER = RandomNormal(mean=0.0, stddev=0.05, seed=_seed)
RINITIALIZER = Orthogonal(gain=1.0, seed=_seed)
RNN = recurrent.GRU
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100

def compile_model(inputs, repeat):
    (vocab_size1, story_maxlen1, query_maxlen1) = inputs[0]
    (vocab_size2, story_maxlen2, query_maxlen2) = inputs[1]
    mvocab_size = vocab_size1
    if (vocab_size2 > mvocab_size):
        mvocab_size = vocab_size2
    
    ensinput = Input(
        (story_maxlen1,))
    ensent = Embedding(
        vocab_size1,
        EMBED_HIDDEN_SIZE,
        input_length=story_maxlen1,
        embedding_initializer=INITIALIZER)(ensinput)
    ensent = Dropout(0.3, seed=_seed)(ensent)
    
    enqinput = Input(
        (query_maxlen1,))
    enques = Embedding(
        vocab_size1,
        EMBED_HIDDEN_SIZE,
        input_length=query_maxlen1,
        embedding_initializer=INITIALIZER)(enqinput)
    enques = Dropout(0.3, seed=_seed)(enques)
    enques = RNN(EMBED_HIDDEN_SIZE, return_sequences=False,
                 kernel_initializer=INITIALIZER,
                 recurrent_initializer=RINITIALIZER)(enques)
    enques = RepeatVector(story_maxlen1)(enques)

    hnsinput = Input(
        (story_maxlen2,))
    hnsent = Embedding(vocab_size2,
        EMBED_HIDDEN_SIZE,
        input_length = story_maxlen2)(hnsinput)
    hnsent = Dropout(0.3)(hnsent)

    hnqinput = Input(
        (query_maxlen2,))
    hnques = Embedding(vocab_size2,
        EMBED_HIDDEN_SIZE,
        input_length = query_maxlen2,
        embedding_initializer=INITIALIZER)(hnqinput)
    hnques = Dropout(0.3, seed=_seed)(hnques)
    hnques = RNN(EMBED_HIDDEN_SIZE, return_sequences = False,
                 kernel_initializer=INITIALIZER,
                 recurrent_initializer=RINITIALIZER)(hnques)
    hnques = RepeatVector(story_maxlen2)(hnques)
    
    enres = merge([ensent, enques], mode='sum')
    
    hnres = merge([hnsent, hnques], mode='sum')
    
    srnn = RNN(EMBED_HIDDEN_SIZE, return_sequences=False,
               kernel_initializer=INITIALIZER,
               recurrent_initializer=RINITIALIZER)
    enrnnout = srnn(enres)
    hnrnnout = srnn(hnres)
    
    do = Dropout(0.3, seed=_seed)
    endoout = do(enrnnout)
    hndoout = do(hnrnnout)
    
    enout = Dense(vocab_size1, activation='softmax',
                kernel_initializer=INITIALIZER)(endoout)
    hnout = Dense(vocab_size2, activation='softmax',
                kernel_initializer=INITIALIZER)(hndoout)
    
    model = Model([ensinput, enqinput, hnsinput, hnqinput],
        [enout, hnout])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return modelwrapper(model)

ver = 'en/hn'
qas = None
ver, qas, repeat = handlesysargs(ver, qas)
evaluate_model(compile_model, ver, qas, pad=1, wmethod='oncat', repeat=repeat, stop_early=1);
