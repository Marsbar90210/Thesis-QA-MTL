import sys
import math
from keras.layers.embeddings import Embedding
from keras.layers import Dense, merge, Dropout, RepeatVector, Reshape
from keras.layers import recurrent, Input, Activation, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import tensorflow as tf
from experiment import evaluate_model, handlesysargs

print("MTL model based on end to end memory with 3 hops sharing all")

EMBED_HIDDEN_SIZE = 20
HOPS = 3
INITIAL_LRATE = 0.01
DROP = 0.5
EPOCHS_DROP = 25.0
INIT_WEIGHT = 'uniform'

def share(layer, inputs):
    return [layer(input) for input in inputs];

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
    (vocab_size1, max_sent1, sent_maxlen1, query_maxlen1) = inputs[0]
    (vocab_size2, max_sent2, sent_maxlen2, query_maxlen2) = inputs[1]
    mvocab_size = vocab_size1
    if (mvocab_size < vocab_size2):
        mvocab_size = vocab_size2
    story_input1 = Input((max_sent1,sent_maxlen1))
    story_input2 = Input((max_sent2,sent_maxlen2))
    query_input1 = Input((sent_maxlen1,))
    query_input2 = Input((sent_maxlen2,))
    
    # H = Dense(EMBED_HIDDEN_SIZE)
    
    embedBlayer = Embedding(vocab_size1,
        EMBED_HIDDEN_SIZE,
        input_length = sent_maxlen1,
        init = INIT_WEIGHT)
    
    embeddingB = share(embedBlayer, [query_input1, query_input2])
    #u = Lambda(lambda x: K.sum(x, axis=2), output_shape=lambda s: (s[0], s[1]))(embeddingB)
    u = [share(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBED_HIDDEN_SIZE,)),embeddingB)]
    embedlayer = None
    
    for hop in range(HOPS):
        embeddingA = share(Embedding(vocab_size1,
            EMBED_HIDDEN_SIZE,
            input_length=sent_maxlen1,
            init = INIT_WEIGHT), [story_input1, story_input2])
        # if hop == 0:
            # embeddingA = share(embedBlayer, [story_input1, story_input2])
        # else:
            # embeddingA = share(embedlayer, [story_input1, story_input2])
        #ms = Lambda(lambda x: K.sum(x, axis=3), output_shape=lambda s: (s[0], s[1], s[2]))(embeddingA)
        ms = share(Lambda(lambda x: K.sum(x, axis=2), output_shape=(max_sent1,EMBED_HIDDEN_SIZE)),embeddingA)
        
        dotproduct = [merge([ms[0], u[-1][0]], mode=row_wise_dot, output_shape=(max_sent1,)),
            merge([ms[1], u[-1][1]], mode=row_wise_dot, output_shape=(max_sent2,))]
        #dotproduct = merge([ms, u], mode=row_wise_cos, output_shape=(max_sent,))
        prob = share(Activation('softmax'), dotproduct)
        
        embedlayer = Embedding(vocab_size1,
            EMBED_HIDDEN_SIZE,
            input_length=sent_maxlen1,
            init = INIT_WEIGHT)
        embeddingC = share(embedlayer, [story_input1, story_input2])
        c = share(Lambda(lambda x: K.sum(x, axis=2), output_shape=(max_sent1, EMBED_HIDDEN_SIZE)),embeddingC)
        c_temp = share(Lambda(lambda x: tf.transpose(x, [0,2,1]), output_shape=(EMBED_HIDDEN_SIZE, max_sent1)),c)
        
        os = [merge([c_temp[0], prob[0]], mode=row_wise_dot, output_shape=(EMBED_HIDDEN_SIZE,)),
            merge([c_temp[1], prob[1]], mode=row_wise_dot, output_shape=(EMBED_HIDDEN_SIZE,))]
        newus = [merge([u[-1][0], os[0]], mode='sum', output_shape=(EMBED_HIDDEN_SIZE,)),
            merge([u[-1][1], os[1]], mode='sum', output_shape=(EMBED_HIDDEN_SIZE,))]
        # u.append(H(newu))
        u.append(newus)
        
    # Applying w matrix
    dls = share(Dense(vocab_size1, input_dim=(EMBED_HIDDEN_SIZE,)),(u[-1]))
    
    # Using last C as W per adjacent weight tying
    # func = lambda x:tf.matmul(x, tf.transpose(embedlayer.get_weights()[0], [1,0]))
    # dls = share(Lambda(func, name='out'), u[-1])
    
    preds = [Activation('softmax')(dls[0]),
        Activation('softmax')(dls[1])]
    
    
    model = Model(input=[story_input1,
        query_input1,
        story_input2,
        query_input2], output=preds)
    
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
    # return model, [LearningRateScheduler(step_decay)]
    return model, None

ver = 'en/hn'
qas = None
ver, qas, repeat = handlesysargs(ver, qas)
evaluate_model(compile_model, ver, qas, pad=1, wmethod='concat', flatten=0, repeat=repeat, limit=500);
