from data import _seed
import numpy as np
np.random.seed(_seed)
import sys
from keras.layers import Dense, merge, Dropout, RepeatVector, Embedding
from keras.layers import recurrent, Input, Activation
from keras.models import Model
from keras.initializers import RandomNormal, Orthogonal
from experiment import evaluate_model, handlesysargs
from model import model as modelwrapper

print("MTL model sharing everything but output, based on gru")

INITIALIZER = RandomNormal(mean=0.0, stddev=0.05, seed=_seed)
RINITIALIZER = Orthogonal(gain=1.0, seed=_seed)
RNN = recurrent.GRU
EMBED_HIDDEN_SIZE = 30#50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100

def compile_model(inputs, repeat):
    (vocab_size, story_maxlen, query_maxlen) = inputs[0]
    #(vocab_size2, story_maxlen2, query_maxlen2) = inputs[1]
    #mvocab_size = vocab_size1
    #if (vocab_size2 > mvocab_size):
    #    mvocab_size = vocab_size2
    
    ensinput = Input(
        (story_maxlen,))
    sentemb = Embedding(
        vocab_size,
        EMBED_HIDDEN_SIZE,
        input_length=story_maxlen,
        embeddings_initializer=INITIALIZER)
    ensent = sentemb(ensinput)
    sentdrop = Dropout(0.3, seed=_seed)
    ensent = sentdrop(ensent)
    
    enqinput = Input(
        (query_maxlen,))
    quesemb = Embedding(
        vocab_size,
        EMBED_HIDDEN_SIZE,
        input_length=query_maxlen,
        embeddings_initializer=INITIALIZER)
    enques = quesemb(enqinput)
    quesdrop = Dropout(0.3, seed=_seed)
    enques = quesdrop(enques)
    quesrnn = RNN(EMBED_HIDDEN_SIZE, return_sequences=False,
                  kernel_initializer=INITIALIZER,
                  recurrent_initializer=RINITIALIZER)
    enques = quesrnn(enques)
    quesrv = RepeatVector(story_maxlen)
    enques = quesrv(enques)

    hnsinput = Input(
        (story_maxlen,))
    hnsent = sentemb(hnsinput)
    hnsent = sentdrop(hnsent)

    hnqinput = Input(
        (query_maxlen,))
    hnques = quesemb(hnqinput)
    hnques = quesdrop(hnques)
    hnques = quesrnn(hnques)
    hnques = quesrv(hnques)
    
    enres = merge([ensent, enques], mode='sum')
    
    hnres = merge([hnsent, hnques], mode='sum')
    
    srnn = RNN(SENT_HIDDEN_SIZE, return_sequences=False,
               kernel_initializer=INITIALIZER,
               recurrent_initializer=RINITIALIZER)
    enrnnout = srnn(enres)
    hnrnnout = srnn(hnres)
    
    do = Dropout(0.3, seed=_seed)
    endoout = do(enrnnout)
    hndoout = do(hnrnnout)
    
    
    dense = Dense(vocab_size, activation=None,
                kernel_initializer=INITIALIZER)
    enout = dense(endoout)
    hnout = dense(hndoout)
    
    enout = Activation('softmax')(enout)
    hnout = Activation('softmax')(hnout)
    
    model = Model([ensinput, enqinput, hnsinput, hnqinput],
        [enout, hnout])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return modelwrapper(model)

ver = 'en/hn'
qas = None
ver, qas, repeat = handlesysargs(ver, qas);
evaluate_model(compile_model, ver, qas, pad=1, wmethod='concat', repeat=repeat, stop_early=0);
