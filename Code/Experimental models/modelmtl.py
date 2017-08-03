import sys
from keras.layers import Dense, merge, Dropout, RepeatVector, Embedding
from keras.layers import recurrent, Input
from keras.models import Model
from experiment import evaluate_model, handlesysargs

print("MTL model sharing rnn layer, based on lstm")

RNN = recurrent.LSTM
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
        input_length=story_maxlen1)(ensinput)
    ensent = Dropout(0.3)(ensent)
    
    enqinput = Input(
        (query_maxlen1,))
    enques = Embedding(
        vocab_size1,
        EMBED_HIDDEN_SIZE,
        input_length=query_maxlen1)(enqinput)
    enques = Dropout(0.3)(enques)
    enques = RNN(EMBED_HIDDEN_SIZE, return_sequences=False)(enques)
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
        input_length = query_maxlen2)(hnqinput)
    hnques = Dropout(0.3)(hnques)
    hnques = RNN(EMBED_HIDDEN_SIZE, return_sequences = False)(hnques)
    hnques = RepeatVector(story_maxlen2)(hnques)
    
    enres = merge([ensent, enques], mode='sum')
    
    hnres = merge([hnsent, hnques], mode='sum')
    
    srnn = RNN(EMBED_HIDDEN_SIZE, return_sequences=False)
    enrnnout = srnn(enres)
    hnrnnout = srnn(hnres)
    
    do = Dropout(0.3)
    endoout = do(enrnnout)
    hndoout = do(hnrnnout)
    
    enout = Dense(vocab_size1, activation='softmax')(endoout)
    hnout = Dense(vocab_size2, activation='softmax')(hndoout)
    
    model = Model([ensinput, enqinput, hnsinput, hnqinput],
        [enout, hnout])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, None

ver = 'en/hn'
qas = None
ver, qas, repeat = handlesysargs(ver, qas)
evaluate_model(compile_model, ver, qas, wmethod='concat', repeat=repeat);
