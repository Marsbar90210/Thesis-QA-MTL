from __future__ import print_function
_seed = None # Use None to disable reproducibility
from functools import reduce
import re
import tarfile
import random
random.seed(_seed)
from itertools import chain

import numpy as np
np.random.seed(_seed) # for reproducibility

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False, E2E=0):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        # if E2E:
        line = line.decode('utf-8').strip()
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            
            if E2E:
                a = [a]
            
            substory = None
            
            if E2E:
                # remove question marks
                if q[-1] == "?":
                    q = q[:-1]
            
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            
            data.append((substory, q, a))
            story.append('')
        else:
            sent = line
            sent = tokenize(sent)
            
            if E2E:
                if sent[-1] == ".":
                    sent = sent[:-1]
            
            story.append(sent)
    return data

def ransubset(data, limit):
    if limit and len(data) > limit:
        return [data[i] for i in sorted(random.sample(range(len(data)), limit))]
    else:
        return data

def get_stories(f, only_supporting=False, max_length=None, flatten=1, limit=None, E2E=0):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting, E2E=E2E)
    func = lambda x:x
    if (flatten):
        func = lambda data: reduce(lambda x, y: x + y, data)
    data = [(func(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    data = ransubset(data, limit)
    return data

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen, word=1):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x, xq = [], []
        if word:
            x = [word_idx.get(w, 0) for w in story]
            xq = [word_idx.get(w, 0) for w in query]
        else:
            func = lambda x, y: x + y
            x = [ord(c) for c in reduce(func, story)]
            xq = [ord(c) for c in reduce(func, query)]
        
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx.get(answer,0)] = 1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

def vectorize_ufstories(data, word_idx, max_sent, sent_maxlen, query_maxlen, word=1):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x, xq = [], []
        if word:
            x = [[word_idx.get(w,0) for w in sent] for sent in story]
            xq = [word_idx.get(w,0) for w in query]
        else:
            func = lambda x, y: x + y
            x = [[ord(c) for c in reduce(func, sent)] for sent in story]
            xq = [ord(c) for c in reduce(func, query)]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx.get(answer,0)] = 1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    X = pad_sequences([pad_sequences(story, maxlen=sent_maxlen).tolist() for story in X], maxlen=max_sent);
    return X, pad_sequences(Xq, maxlen=sent_maxlen), np.array(Y)

def vectorize_data(data, word_idx, max_sent, sent_maxlen, query_maxlen, word=1):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sent_maxlen - len(sentence))
            ss.append([word_idx.get(w,0) for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:max_sent][::-1]

        # Make the last word of each sentence the time 'word' which 
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - max_sent - i + len(ss)

        # pad to max_sent
        lm = max(0, max_sent - len(ss))
        for _ in range(lm):
            ss.append([0] * sent_maxlen)

        lq = max(0, sent_maxlen - len(query))
        q = [word_idx.get(w,0) for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx.get(a,0)] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)   

TAR = None

def init_tar():
    global TAR
    if (not TAR):
	    try:
	        path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
	    except:
	        print('Error downloading dataset, please download it manually:\n'
		      '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
		      '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
	        raise
	    TAR = tarfile.open(path)

def read_challenge(challenge, flatten=1, limit=None, E2E=0):
    init_tar()
    train = get_stories(TAR.extractfile(challenge.format('train')), flatten=flatten, limit=limit, E2E=E2E)
    test = get_stories(TAR.extractfile(challenge.format('test')), flatten=flatten, limit=limit, E2E=E2E)
    
    return train, test

def calc_challenge(train, test, flatten=1, word=1, E2E=0, memory_size=20):
    if E2E:
        data = train + test

        temp = [set(reduce(lambda x, y:x + y, s) + q + a) for s, q, a in data]
        vocab = sorted(reduce(lambda x, y: x | y, temp))
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

        max_story_size = max(map(len, (s for s, _, _ in data)))
        mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
        sentence_size = max(map(len, reduce(lambda x, y:x + y, (s for s, _, _ in data))))
        query_size = max(map(len, (q for _, q, _ in data)))
        memory_size = min(memory_size, max_story_size)

        # Add time words/indexes
        for i in range(memory_size):
            word_idx['time{}'.format(i+1)] = 'time{}'.format(i+1)

        vocab_size = len(word_idx) + 1 # +1 for nil word
        sentence_size = max(query_size, sentence_size) # for the position
        sentence_size += 1  # +1 for time words
        
        return word_idx, vocab_size, memory_size, sentence_size, sentence_size

    data = []
    if (flatten):
        data = (set(story + q + [answer]) for story, q, answer in train + test)
    else:
        data = (set(reduce(lambda x, y:x + y, story) + q + [answer]) for story, q, answer in train + test)
    
    vocab = sorted(reduce(lambda x, y: x | y, data))
    # Reserve 0 for masking via pad_sequences
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    
    func = lambda x:reduce(lambda x, y:x + y, x)
    vocab_size = len(word_idx) + 1
    if word:
        func = lambda x:x

    if (flatten):
        story_maxlen = max(map(len, func((x for x, _, _ in train + test))))
        query_maxlen = max(map(len, func((x for _, x, _ in train + test))))
        return word_idx, vocab_size, story_maxlen, query_maxlen
    else:
        max_sent = max(map(len, (x for x, _, _ in train + test)))
        local_maxlen = lambda x: max(map(len, map(func, x)))
        sent_maxlen = max(map(local_maxlen, (x for x, _, _ in train + test)))
        query_maxlen = max(map(len, func((x for _, x, _ in train + test))))
        return word_idx, vocab_size, max_sent, sent_maxlen, query_maxlen

def get_challenge(qa, ver='en', wmethod=None, flatten=1, word=1, limit=None, E2E=0, memory_size=20):
    challenge = 'tasks_1-20_v1-2/{}/{}'.format(ver, qas_to_challenges[qa])
    train, test = read_challenge(challenge, flatten=flatten, limit=limit, E2E=E2E)
    calc = calc_challenge(train, test, flatten=flatten, word=word, E2E=E2E, memory_size=memory_size)
    if callable(wmethod):
        widx = wmethod()
        calc = (widx, len(widx)+1, *calc[2:])
    if E2E:
        word_idx, vocab_size, max_sent, sent_maxlen, query_maxlen = calc
        X, Xq, Y = vectorize_data(train, word_idx, max_sent, sent_maxlen, query_maxlen, word=word)
        tX, tXq, tY = vectorize_data(test, word_idx, max_sent, sent_maxlen, query_maxlen, word=word)
        return ((X, Xq, Y),(tX, tXq, tY), (vocab_size, max_sent, sent_maxlen, query_maxlen))
    if (flatten):
        word_idx, vocab_size, story_maxlen, query_maxlen = calc
        X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen, word=word)
        tX, tXq, tY = vectorize_stories(test, word_idx, story_maxlen, query_maxlen, word=word)
        return ((X, Xq, Y),(tX, tXq, tY), (vocab_size, story_maxlen, query_maxlen))
    else:
        word_idx, vocab_size, max_sent, sent_maxlen, query_maxlen = calc
        X, Xq, Y = vectorize_ufstories(train, word_idx, max_sent, sent_maxlen, query_maxlen, word=word)
        tX, tXq, tY = vectorize_ufstories(test, word_idx, max_sent, sent_maxlen, query_maxlen, word=word)
        return ((X, Xq, Y),(tX, tXq, tY), (vocab_size, max_sent, sent_maxlen, query_maxlen))

def get_mtlchallenge(challenges, norm, wimethod, flatten=1, word=1, limit=None):
    data = [(read_challenge(c, flatten, limit=limit)) for c in challenges]
    calcs = [(calc_challenge(train, test, flatten, word=word)) for train, test in data]
    if (norm):
        if(flatten):
            smm = 0
            qmm = 0
            for _, _, sm, qm in calcs:
                if(sm > smm):
                    smm = sm
                if(qm > qmm):
                    qmm = qm
            calcs = [(t1, t2, smm, qmm) for t1, t2, _, _ in calcs]
        else:
            msm = 0
            smm = 0
            qmm = 0
            for _, _, ms, sm, qm in calcs:
                if(ms > msm):
                    msm = ms
                if(sm > smm):
                    smm = sm
                if(qm > qmm):
                    qmm = qm
            calcs = [(t1, t2, ms, smm, qmm) for t1, t2, _, _, _ in calcs]
    if (wimethod == 'concat'):
        if (flatten):
            wordidx = {}
            for widx in [widx for widx, _, _, _ in calcs]:
                wordidx.update(widx);
            calcs = [(wordidx, len(wordidx)+1, t1, t2) for _, _, t1, t2 in calcs]
        else:
            wordidx = {}
            for widx in [widx for widx, _, _, _, _ in calcs]:
                wordidx.update(widx);
            calcs = [(wordidx, len(wordidx)+1, t1, t2, t3) for _, _, t1, t2, t3 in calcs]
    if (callable(wimethod)):
        widx = wimethod()
        calcs = [(widx, len(widx)+1, *calc[2:]) for calc in calcs]
    for dat, calc in zip(data, calcs):
        train, test = dat
        if (flatten):
            word_idx, vocab_size, story_maxlen, query_maxlen = calc
            X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen, word=word)
            tX, tXq, tY = vectorize_stories(test, word_idx, story_maxlen, query_maxlen, word=word)
            yield ((X, Xq, Y),(tX, tXq, tY), (vocab_size, story_maxlen, query_maxlen))
        else:
            word_idx, vocab_size, max_sent, sent_maxlen, query_maxlen = calc
            X, Xq, Y = vectorize_ufstories(train, word_idx, max_sent,
                sent_maxlen, query_maxlen, word=word)
            tX, tXq, tY = vectorize_ufstories(test, word_idx, max_sent,
                sent_maxlen, query_maxlen, word=word)
            yield ((X, Xq, Y),(tX, tXq, tY),
                (vocab_size, max_sent, sent_maxlen, query_maxlen))

qas_to_challenges = {
    'qa1': 'qa1_single-supporting-fact_{}.txt',
    'qa2': 'qa2_two-supporting-facts_{}.txt',
    'qa3': 'qa3_three-supporting-facts_{}.txt',
    'qa4': 'qa4_two-arg-relations_{}.txt',
    'qa5': 'qa5_three-arg-relations_{}.txt',
    'qa6': 'qa6_yes-no-questions_{}.txt',
    'qa7': 'qa7_counting_{}.txt',
    'qa8': 'qa8_lists-sets_{}.txt',
    'qa9': 'qa9_simple-negation_{}.txt',
    'qa10': 'qa10_indefinite-knowledge_{}.txt',
    'qa11': 'qa11_basic-coreference_{}.txt',
    'qa12': 'qa12_conjunction_{}.txt',
    'qa13': 'qa13_compound-coreference_{}.txt',
    'qa14': 'qa14_time-reasoning_{}.txt',
    'qa15': 'qa15_basic-deduction_{}.txt',
    'qa16': 'qa16_basic-induction_{}.txt',
    'qa17': 'qa17_positional-reasoning_{}.txt',
    'qa18': 'qa18_size-reasoning_{}.txt',
    'qa19': 'qa19_path-finding_{}.txt',
    'qa20': 'qa20_agents-motivations_{}.txt'
}

def get_challenges(ver, qas, pad=1, wmethod=None, flatten=1, word=1, limit=None, E2E=0):
    ver = ver.lower();
    vers = ver.split('/');
    if (len(qas) == 1 and len(vers) == 1):
        for qa in qas:
            qa = qa.lower();
            yield ([get_challenge(qa, ver, wmethod, flatten, word, limit, E2E)], qa)
    else:
        challenges = []
        for qa in qas:
            qa = qa.lower();
            challenges = challenges + ['tasks_1-20_v1-2/{}/{}'
                                    .format(ver, qas_to_challenges[qa])
                                        for ver in vers]
        yield (get_mtlchallenge(challenges, pad, wmethod, flatten, word, limit), qas)
