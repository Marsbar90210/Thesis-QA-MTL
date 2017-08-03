import sys
import math
from keras.layers.embeddings import Embedding
from keras.layers import Dense, merge, Dropout, RepeatVector
from keras.layers import recurrent, Input, Activation, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import tensorflow as tf
from experiment import evaluate_model, handlesysargs
from model import model as modelwrapper

print("STL model based on end to end memory")

def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 25.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

EMBED_HIDDEN_SIZE = 20

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
    
    embeddingA = Embedding(vocab_size,
        EMBED_HIDDEN_SIZE,
        input_length=sent_maxlen)(story_input)
    #ms = Lambda(lambda x: K.sum(x, axis=3), output_shape=lambda s: (s[0], s[1], s[2]))(embeddingA)
    ms = Lambda(lambda x: K.sum(x, axis=2), output_shape=(max_sent,EMBED_HIDDEN_SIZE))(embeddingA)
    embeddingB = Embedding(vocab_size,
        EMBED_HIDDEN_SIZE,
        input_length = sent_maxlen)(query_input)
    #u = Lambda(lambda x: K.sum(x, axis=2), output_shape=lambda s: (s[0], s[1]))(embeddingB)
    u = Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBED_HIDDEN_SIZE,))(embeddingB)
    
    dotproduct = merge([ms, u], mode=row_wise_dot, output_shape=(max_sent,))
    #dotproduct = merge([ms, u], mode=row_wise_cos, output_shape=(max_sent,))
    prob = Activation('softmax')(dotproduct)
    
    embeddingC = Embedding(vocab_size,
        EMBED_HIDDEN_SIZE,
        input_length=sent_maxlen)(story_input)
    c = Lambda(lambda x: K.sum(x, axis=2), output_shape=(max_sent, EMBED_HIDDEN_SIZE))(embeddingC)
    c_temp = Lambda(lambda x: tf.transpose(x, [0,2,1]), output_shape=(EMBED_HIDDEN_SIZE, max_sent))(c)
	
    o = merge([c_temp, prob], mode=row_wise_dot, output_shape=(EMBED_HIDDEN_SIZE,))
    sumou = merge([u, o], mode='sum')
    
    dl = Dense(vocab_size, input_dim=(EMBED_HIDDEN_SIZE,))(sumou)
    
    # pred = Dense(vocab_size, activation='softmax')(dl)
    pred = Activation('softmax')(dl)
    
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
                  loss='categorical_crossentropy')
    # return model, [LearningRateScheduler(step_decay)]
    return modelwrapper(model)

ver = 'en'
qas = None
ver, qas, repeat = handlesysargs(ver, qas)
evaluate_model(compile_model, ver, qas, pad=1, wmethod=None, flatten=0, repeat=repeat, E2E=0);
