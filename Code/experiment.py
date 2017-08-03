from data import _seed, get_challenges
import os, sys, time
from timeit import default_timer as timer
from math import floor, ceil
import numpy as np
np.random.seed(_seed)
from sklearn import model_selection, metrics
import tensorflow as tf
tf.set_random_seed(_seed)

from model import model as modelwrapper

VERBOSE = 0

BATCH_SIZE = 32
EPOCHS = 100#50
VALIDATION_SPLIT = 0.10#0.05
EVALUATION_INTERVAL = 10
PRE_TRAIN_LIMIT = 200

ALLQA = ['qa1',
    'qa2',
    'qa3',
    'qa4',
    'qa5',
    'qa6',
    'qa7',
    'qa8',
    'qa9',
    'qa10',
    'qa11',
    'qa12',
    'qa13',
    'qa14',
    'qa15',
    'qa16',
    'qa17',
    'qa18',
    'qa19',
    'qa20']

def pphistory(history):
    for key in sorted(history.keys()):
        print('Train {}: max={:.4f} mean={:.4f} min={:.4f} final={:.4f}'.format(
            key,
            max(history[key]),
            np.mean(history[key]),
            min(history[key]),
            history[key][-1]))
    print('')

def ppresult(loss, acc):
    if (isinstance(loss, list)):
        print('Total test loss = {:.4f}'.format(sum(loss)))
    else:
        loss = [loss]
    for i, (loss, acc) in enumerate(zip(loss, acc)):
        if hasattr(loss, "__len__"):
                loss = float(loss)
        print('Test loss / test accuracy for output {} = {:.4f} / {:.4f}'.format(i, loss, acc))

def handlesysargs(ver, qas, repeat=1):
    if (len(sys.argv) > 1):
        ver = sys.argv[1]
    if (len(sys.argv) > 2):
        if (sys.argv[2].lower() == 'none'):
            qas = None
        else:
            qas = sys.argv[2].split(',')
            qas = [temp.split('.') for temp in qas]
    if (len(sys.argv) > 3):
        repeat = int(sys.argv[3]);
    return ver, qas, repeat

def mes_acc(pred, labels):
    if len(labels) < 2:
        pred = [pred]
    return [metrics.accuracy_score(np.argmax(p, axis=1), np.argmax(l, axis=1))
        for p, l in zip(pred, labels)]

def update_history(history, data, name):
    if not isinstance(data, list):
        history[name] = history.get(name, [])
        history[name].append(data)
    else:
        for i, d in enumerate(data):
            n = '{} {}'.format(name, i)
            history[n] = history.get(n, [])
            if hasattr(d, "__len__"):
                d = float(d)
            history[n].append(d)
    return history

def training(model, trainX, trainY, valX, valY, batches, history={}, stop_early=1):
    best_val_acc = [0];
    for t in range(1, EPOCHS+1):

        np.random.shuffle(batches)
        for start, end in batches:
            x = [da[start:end] for da in trainX]
            y = [da[start:end] for da in trainY]
            model.fit(x, y, t)

        if t % EVALUATION_INTERVAL == 0:
            if stop_early:
                model.save()
            
            train_loss = model.test(trainX, trainY)
            train_pred = model.predict(trainX)
            train_acc = mes_acc(train_pred, trainY)
            
            val_loss = model.test(valX, valY)
            val_pred = model.predict(valX)
            val_acc = mes_acc(val_pred, valY)
            
            if stop_early:
                if val_acc[0] >= best_val_acc[0]:
                    best_val_acc = val_acc
                else:
                    best_t = t-EVALUATION_INTERVAL
                    if model.restore(1):
                        print("Stopped early in iteration", t)
                        print("Returning iteration", best_t)
                        return model, history
            
            history = update_history(history, train_loss, 'loss')
            history = update_history(history, train_acc, 'acc')
            history = update_history(history, val_loss, 'val loss')
            history = update_history(history, val_acc, 'val acc')
            
            if VERBOSE:
                print("Epoch {} stat data:".format(t))
                print("Stat loss", train_loss)
                print("Stat acc", train_acc)
                print("Stat val loss", val_loss)
                print("Stat val acc", val_acc)
    return model, history

def evaluate_model(compile_model, ver,
                   qass=None,
                   pad=1,
                   wmethod=None,
                   flatten=1,
                   repeat=1,
                   word=1,
                   limit=None,
                   E2E=0,
                   stop_early=1):
    print("Running model for {}, {}".format(ver, time.strftime("%d/%m/%Y")));
    print("Parameters pad={}, wmethod={}, flatten={}, repeat={}, word={}, limit={}, stop_early={}".
        format(pad, wmethod, flatten, repeat, word, limit, stop_early))
    if (not qass):
        qass = ALLQA
    if (not any(isinstance(i, list) for i in qass)):
        qass = [[temp] for temp in qass]
    for qas in qass:
        trainX = [];
        trainY = [];
        testX = [];
        testY = [];
        inputs = [];
        for challenges, qa in get_challenges(ver, qas, pad, wmethod, flatten, word, limit, E2E):
            for challenge in challenges:
                (X, Xq, Y), (tX, tXq, tY), input = challenge
                trainX.append(X)
                trainX.append(Xq)
                testX.append(tX)
                testX.append(tXq)
                trainY.append(Y)
                testY.append(tY)
                inputs.append(input)
        
        temp = model_selection.train_test_split(*trainX,
                                         *trainY,
                                         test_size=VALIDATION_SPLIT)
        n_in = len(trainX)
        trainT = temp[0::2]
        valT = temp[1::2]
        trainX = trainT[0:n_in]
        valX = valT[0:n_in]
        trainY = trainT[n_in:]
        valY = valT[n_in:]
        
        n_train = len(trainX[0])
        batch_size = BATCH_SIZE
        
        batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
        batches = [(start, end) for start, end in batches]
        
        
        if (len(qa) == 1):
            qa = qa[0]
        
        for i in range(repeat):
            model = compile_model(inputs, len(qas))
            
            if (not model):
                return
            if (not isinstance(model, modelwrapper)):
                print("This model is deprcated and has to be rewritten to use the new model wrapper")
                return
            
            print('')
            print('Training {} iter {}'.format(qa, i+1))
            
            start = timer();
            model, history = training(model, trainX, trainY, valX, valY, batches, stop_early=stop_early)
            end = timer();
            print('Training took {:.4f} sec'.format(end-start))
            pphistory(history)
                
            test_loss = model.test(testX, testY)
            test_pred = model.predict(testX)
            test_acc = mes_acc(test_pred, testY)
            ppresult(test_loss, test_acc)
            print('', flush=True)
