import sys
import math
from keras.layers.embeddings import Embedding
from keras.layers import Dense, merge, Dropout, RepeatVector, Reshape
from keras.layers import recurrent, Input, Activation, Lambda
from keras.models import Model
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import tensorflow as tf
from experiment import evaluate_model, handlesysargs

print("STL model based on end to end memory with 3 hops")

EMBED_HIDDEN_SIZE = 20
HOPS = 3
INITIAL_LRATE = 0.01
DROP = 0.5
EPOCHS_DROP = 25.0
INIT_WEIGHT = 'uniform'
W_CONSTRAINT = None#maxnorm(40)

def step_decay(epoch):
	lrate = INITIAL_LRATE * math.pow(DROP, math.floor((1+epoch)/EPOCHS_DROP))
	return lrate

def row_wise_dot(tensors):
    m = tensors[0]
    u = tensors[1]
    u_temp = tf.transpose(tf.expand_dims(u, -1), [0,2,1])
    return tf.reduce_sum(m * u_temp, 2)

def row_wise_cos(tensors):
    m = tensors[0]
    u = tensors[1]
    unorm = tf.sqrt(tf.reduce_sum(tf.square(u), 1))
    mnorm = tf.sqrt(tf.reduce_sum(tf.square(m), 2))
    norm = tf.multiply(tf.transpose(mnorm), unorm)
    u_temp = tf.transpose(tf.expand_dims(u, -1), [0,2,1])
    return tf.divide(tf.reduce_sum(m * u_temp, 2), mnorm)

def compile_model(inputs, repeat):
    (vocab_size, max_sent, sent_maxlen, query_maxlen) = inputs[0]
    story_input = Input((max_sent,sent_maxlen))
    query_input = Input((sent_maxlen,))
    
    # H = Dense(EMBED_HIDDEN_SIZE)
    
    embedBlayer = Embedding(vocab_size,
        EMBED_HIDDEN_SIZE,
        input_length = sent_maxlen,
        init = INIT_WEIGHT,
        W_constraint = W_CONSTRAINT)
    
    embeddingB = embedBlayer(query_input)
    #u = Lambda(lambda x: K.sum(x, axis=2), output_shape=lambda s: (s[0], s[1]))(embeddingB)
    u = [Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBED_HIDDEN_SIZE,))(embeddingB)]
    embedlayer = None
    
    for hop in range(HOPS):
        embeddingA = Embedding(vocab_size,
            EMBED_HIDDEN_SIZE,
            input_length = sent_maxlen,
            init = INIT_WEIGHT,
            W_constraint = W_CONSTRAINT)(story_input)
        # if hop == 0:
            # embeddingA = embedBlayer(story_input)
        # else:
            # embeddingA = embedlayer(story_input)
        #ms = Lambda(lambda x: K.sum(x, axis=3), output_shape=lambda s: (s[0], s[1], s[2]))(embeddingA)
        ms = Lambda(lambda x: K.sum(x, axis=2), output_shape=(max_sent,EMBED_HIDDEN_SIZE))(embeddingA)
        
        dotproduct = merge([ms, u[-1]], mode=row_wise_dot, output_shape=(max_sent,))
        #dotproduct = merge([ms, u], mode=row_wise_cos, output_shape=(max_sent,))
        prob = Activation('softmax')(dotproduct)
        
        embedlayer = Embedding(vocab_size,
            EMBED_HIDDEN_SIZE,
            input_length=sent_maxlen,
            init = INIT_WEIGHT,
            W_constraint = W_CONSTRAINT)
        embeddingC = embedlayer(story_input)
        c = Lambda(lambda x: K.sum(x, axis=2), output_shape=(max_sent, EMBED_HIDDEN_SIZE))(embeddingC)
        c_temp = Lambda(lambda x: tf.transpose(x, [0,2,1]), output_shape=(EMBED_HIDDEN_SIZE, max_sent))(c)
        
        o = merge([c_temp, prob], mode=row_wise_dot, output_shape=(EMBED_HIDDEN_SIZE,))
        newu = merge([u[-1], o], mode='sum', output_shape=(EMBED_HIDDEN_SIZE,))
        # u.append(H(newu))
        u.append(newu)
        
    # Applying w matrix
    dl = Dense(vocab_size, input_dim=(EMBED_HIDDEN_SIZE,))(u[-1])
    
    # Using last C as W per adjacent weight tying
    # func = lambda x:tf.matmul(x, tf.transpose(embedlayer.get_weights()[0], [1,0]))
    # dl = Lambda(func)(newu)
    
    pred = Activation('softmax')(dl)
    # pred = Dense(vocab_size, activation='softmax')(dl)
    
    model = Model(input=[story_input, query_input], output=[pred])
    
    opt = Adam(lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-08,
               decay=0.0)
    
    # opt = SGD(lr=0.0,
              # momentum=0.0,
              # decay=0.0,
              # nesterov=False)
    
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, [LearningRateScheduler(step_decay)]
    # return model, None

ver = 'en'
qas = None
repeat = 1;
ver, qas = handlesysargs(ver, qas, repeat)
evaluate_model(compile_model, ver, qas, pad=1, wmethod=None, flatten=0, repeat=repeat);
