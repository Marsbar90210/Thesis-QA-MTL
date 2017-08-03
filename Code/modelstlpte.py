from data import _seed
import numpy as np
np.random.seed(_seed)
import sys
from keras.layers.embeddings import Embedding
from keras.layers import Dense, merge, Dropout, RepeatVector
from keras.layers import recurrent, Input
from keras.models import Model
from keras.initializers import RandomNormal, Orthogonal
import keras.backend as K
from experiment import evaluate_model, handlesysargs
from model import model as modelwrapper
from pretraining import pre_train_embedding

print("STL model based on GRU")

embedding_matrix = None
INITIALIZER = RandomNormal(mean=0.0, stddev=0.05, seed=_seed)
RINITIALIZER = Orthogonal(gain=1.0, seed=_seed)
RNN = recurrent.GRU
EMBED_HIDDEN_SIZE = 30#50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
Emb_train = False

def compile_model(inputs, repeat):
    (vocab_size, story_maxlen, query_maxlen) = inputs[0]
    
    sentinp = Input((story_maxlen,))
    sentrnn = Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                        input_length=story_maxlen,
                        weights=[embedding_matrix],
                        trainable=Emb_train)(sentinp)
    sentrnn = Dropout(0.3, seed=_seed)(sentrnn)
    #sentrnn = RNN(EMBED_HIDDEN_SIZE, return_sequences=False,
    #           kernel_initializer=INITIALIZER,
    #           recurrent_initializer=RINITIALIZER)(sentrnn)

    qinput = Input((query_maxlen,))
    qrnn = Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                       input_length=query_maxlen,
                       weights=[embedding_matrix],
                       trainable=Emb_train)(qinput)
    qrnn = Dropout(0.3, seed=_seed)(qrnn)
    qrnn = RNN(EMBED_HIDDEN_SIZE, return_sequences=False,
               kernel_initializer=INITIALIZER,
               recurrent_initializer=RINITIALIZER)(qrnn)
    qrnn = RepeatVector(story_maxlen)(qrnn)
    
    out = merge([sentrnn, qrnn], mode='sum')
    out = RNN(SENT_HIDDEN_SIZE, return_sequences=False,
              kernel_initializer=INITIALIZER,
              recurrent_initializer=RINITIALIZER)(out)
    out = Dropout(0.3, seed=_seed)(out)
    out = Dense(vocab_size,
                kernel_initializer=INITIALIZER,
                activation='softmax')(out)

    model = Model([sentinp, qinput],[out])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return modelwrapper(model)

ver, qas, repeat = handlesysargs('en', None, 1)
embedding_matrix, word_idx = pre_train_embedding(EMBED_HIDDEN_SIZE, ver, method='external')
word_idx = {w: i+1 for w, i in word_idx.items()}
nil_slot = np.zeros((1,EMBED_HIDDEN_SIZE), float)
embedding_matrix = np.append(nil_slot, embedding_matrix, axis=0)

def wmethod():
    global word_idx
    return word_idx

evaluate_model(compile_model, ver, qas, pad=1, wmethod=wmethod, repeat=repeat, stop_early=0);
