from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, merge, Permute, Dropout, Input
from keras.layers import LSTM, GRU
from keras.models import Model
from keras.optimizers import Adam, SGD
from experiment import evaluate_model, handlesysargs
import sys

EMBED_HIDDEN_SIZE = 20

def compile_model(inputs, repeat):
    (vocab_size, story_maxlen, query_maxlen) = inputs[0]3
    story_input = Input((story_maxlen,))
    query_input = Input((query_maxlen,))
    
    # embed the input sequence into a sequence of vectors
    input_encoder_m = Embedding(input_dim=vocab_size,
                                  output_dim=EMBED_HIDDEN_SIZE,
                                  input_length=story_maxlen,
                                  dropout=0.3)(story_input)
    # input_encoder_m.add(Dropout(0.3)) 
    # output: (samples, story_maxlen, embedding_dim)
    # embed the question into a sequence of vectors
    question_encoder = Embedding(input_dim=vocab_size,
                                   output_dim=EMBED_HIDDEN_SIZE,
                                   input_length=query_maxlen,
                                   dropout=0.3)(query_input)
    # question_encoder.add(Dropout(0.3))
    # output: (samples, query_maxlen, embedding_dim)
    # compute a 'match' between input sequence elements (which are vectors)
    # and the question vector sequence
    match = merge([input_encoder_m, question_encoder],
                    mode='dot',
                    dot_axes=[2, 2])
    match = Activation('softmax')(match)
    # output: (samples, story_maxlen, query_maxlen)
    # embed the input into a single vector with size = story_maxlen:
    input_encoder_c = Embedding(input_dim=vocab_size,
                                  output_dim=query_maxlen,
                                  input_length=story_maxlen,
                                  dropout=0.3)(story_input)
    # input_encoder_c.add(Dropout(0.3))
    # output: (samples, story_maxlen, query_maxlen)
    # sum the match vector with the input vector:
    response = merge([match, input_encoder_c], mode='sum')
    # output: (samples, story_maxlen, query_maxlen)
    response = Permute((2, 1))(response)  # output: (samples, query_maxlen, story_maxlen)

    # concatenate the match vector with the question vector,
    # and do logistic regression on top
    answer = merge([response, question_encoder], mode='concat', concat_axis=-1)
    # the original paper uses a matrix multiplication for this reduction step.
    # we choose to use a RNN instead.
    answer = LSTM(32)(answer)
    # one regularization layer -- more would probably be needed.
    answer = Dropout(0.3)(answer)
    answer = Dense(vocab_size)(answer)
    # we output a probability distribution over the vocabulary
    answer = Activation('softmax')(answer)

    model = Model(input=[story_input, query_input], output=[answer])
    
    opt = Adam(lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-08,
               decay=0.0)
    
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, None

ver = 'en'
qas = None
ver, qas, repeat = handlesysargs(ver, qas)
evaluate_model(compile_model, ver, qas, pad=1, wmethod=None, flatten=1, repeat=repeat);
