import sys
import math
from keras import backend as K
import tensorflow as tf
import numpy as np
from experiment import evaluate_model, handlesysargs
from model import model
from pretraining import pre_train_embedding

print("Re-implementation of MemN2N model in tf with external pretrained embedding")

def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 25.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    t = tf.convert_to_tensor(t, name="t")
    s = tf.shape(t)[1]
    z = tf.zeros(tf.stack([1, s]))
    return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    t = tf.convert_to_tensor(t, name="t")
    gn = tf.random_normal(tf.shape(t), stddev=stddev)
    return tf.add(t, gn, name=name)

embedding_matrix = None
EMBEDDING_SIZE = 20
HOPS = 3
MAX_GRAD_NORM = 40
NONLIN = None
#INIT = tf.random_normal_initializer(stddev=0.1)
INIT = lambda x: embedding_matrix
ENCODING = position_encoding
LRF = step_decay
WS = 0
EMB_TRAIN = True

COUNTER = 0

def compile_model(inputs, repeat):
    global COUNTER
    COUNTER += 1
    vocab_size, max_sent, sen_maxlen, query_maxlen = inputs[0]
    tf.reset_default_graph()
    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
    
        # Inputs and variables
        stories = tf.placeholder(tf.int32, [None, max_sent, sen_maxlen])
        queries = tf.placeholder(tf.int32, [None, sen_maxlen])
        answers = tf.placeholder(tf.int32, [None, vocab_size])
        lr = tf.placeholder(tf.float32, [])
        
        nil_word_slot = tf.zeros([1, EMBEDDING_SIZE])
        A = tf.concat(axis=0, values=[ nil_word_slot, INIT([vocab_size-1, EMBEDDING_SIZE]) ])
        VC = tf.concat(axis=0, values=[ nil_word_slot, INIT([vocab_size-1, EMBEDDING_SIZE]) ])
        
        A_1 = tf.Variable(A, name="A", trainable=EMB_TRAIN)

        C = []

        for hopn in range(HOPS):
            with tf.variable_scope('hop_{}'.format(hopn)):
                C.append(tf.Variable(VC, name="C", trainable=EMB_TRAIN))

        # Dont use projection for layerwise weight sharing
        # H = tf.Variable(_INIT([_EMBEDDING_SIZE, EMBEDDING_SIZE]), name="H")

        # Use final C as replacement for W
        if not WS:
            W = tf.Variable(tf.transpose(VC, [1,0]), name="W")

        nil_vars = set([A_1.name] + [x.name for x in C])
        
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

        encoding = tf.constant(ENCODING(sen_maxlen, EMBEDDING_SIZE), name="encoding")
        
        # Actual model
        # Use A_1 for the question embedding as per Adjacent Weight Sharing
        q_emb = tf.nn.embedding_lookup(A_1, queries)
        u_0 = tf.reduce_sum(q_emb * encoding, 1)
        u = [u_0]

        for hopn in range(HOPS):
            if hopn == 0:
                m_emb_A = tf.nn.embedding_lookup(A_1, stories)
                m_A = tf.reduce_sum(m_emb_A * encoding, 2)

            else:
                with tf.variable_scope('hop_{}'.format(hopn - 1)):
                    m_emb_A = tf.nn.embedding_lookup(C[hopn - 1], stories)
                    m_A = tf.reduce_sum(m_emb_A * encoding, 2)

            # hack to get around no reduce_dot
            u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
            dotted = tf.reduce_sum(m_A * u_temp, 2)

            # Calculate probabilities
            probs = tf.nn.softmax(dotted)

            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            with tf.variable_scope('hop_{}'.format(hopn)):
                m_emb_C = tf.nn.embedding_lookup(C[hopn], stories)
            m_C = tf.reduce_sum(m_emb_C * encoding, 2)

            c_temp = tf.transpose(m_C, [0, 2, 1])
            o_k = tf.reduce_sum(c_temp * probs_temp, 2)

            # Dont use projection layer for adj weight sharing
            # u_k = tf.matmul(u[-1], H) + o_k

            u_k = u[-1] + o_k

            # nonlinearity
            if NONLIN:
                u_k = NONLIN(u_k)

            u.append(u_k)

        # Use last C for output (transposed)
        with tf.variable_scope('hop_{}'.format(HOPS)):
            if WS:
                logits = tf.matmul(u_k, tf.transpose(C[-1], [1,0]))
            else:
                logits = tf.matmul(u_k, W)
        
        # Loss functions and training
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum

        # gradient pipeline
        grads_and_vars = opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, MAX_GRAD_NORM), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = opt.apply_gradients(nil_grads_and_vars, name="train_op")

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        modelsaver = tf.train.Saver()
        
        # Bind to model wrapper
        def fitf(trainX, trainY, t):
            feed_dict = {stories: trainX[0],
                         queries: trainX[1],
                         answers: trainY[0],
                         lr: LRF(t)}
            return sess.run(train_op, feed_dict=feed_dict)

        def testf(trainX, trainY):
            feed_dict = {stories: trainX[0],
                         queries: trainX[1],
                         answers: trainY[0]}
            return sess.run(loss_op, feed_dict=feed_dict)

        def predictf(trainX):
            feed_dict = {stories: trainX[0],
                        queries: trainX[1]}
            return sess.run(logits, feed_dict=feed_dict)
        
        def savef(filepath):
            return modelsaver.save(sess, filepath)
        
        def restoref(filepath):
            modelsaver.restore(sess, filepath)

    return model(fitf=fitf,
                 testf=testf,
                 predictf=predictf,
                 savef=savef,
                 restoref=restoref)


ver, qas, repeat = handlesysargs('en', None, 10)
embedding_matrix, word_idx = pre_train_embedding(EMBEDDING_SIZE, ver, method='external')

def wmethod():
    global word_idx
    return word_idx

evaluate_model(compile_model, ver, qas, pad=1, wmethod=wmethod, flatten=0, repeat=repeat, E2E=1);
