from data import read_challenge, qas_to_challenges, calc_challenge, ransubset, _seed
import numpy as np
np.random.seed(_seed)
import tensorflow as tf
tf.set_random_seed(_seed)
from functools import reduce
import math, os
from os import path
from timeit import default_timer as timer
from transliterate import transliterate, de_transliterate
from polyglot.mapping import Embedding
from difflib import SequenceMatcher

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

def load_ding_dic(path):
    with open(path, encoding='utf-8') as fh:
        data = {}
        for line in fh:
            parts = line.split('::')
            enparts = parts[0].split(' ')
            hiparts = parts[1].split(',')
            hiparts = list(map(lambda x: x.strip().split(' ')[0].strip(), hiparts))
            enparts = list(map(str.strip, enparts))
            temp = data.get(enparts[0], {'trans': [], 'mean': [], 'tag': []})
            temp['trans'] += hiparts
            temp['mean'].append(enparts[2])
            temp['tag'].append(enparts[1])
            data[enparts[0]] = temp
        return data

def ran_translate(dic, sour):
    opts = []
    p = [1]
    if (sour in dic):
        opts = list(map(de_transliterate, dic[sour]))
    k = len(opts)
    if not k == 0:
        p = [1/2] + [1/(2*k)]*k
    return np.random.choice([sour] + opts, replace=False, p=p)

def prep_dic(word_idx):
    dic = load_ding_dic(FILE_DIR + 'en-hi-enwiktionary.txt')
    dic = {k: dic[k] for k in dic if k in word_idx}
    for k in dic:
        dic[k] = [t for t in dic[k]['trans'] if t in word_idx]
    return dic

FILE_DIR = './../data/'

def get_raw_data(ver, qas, limit=None):
    vers = ver.split('/')
    train_acc, test_acc = [], []

    for ver in vers:
        challenge = 'tasks_1-20_v1-2/{}/'.format(ver)

        for qa in qas:
            text = challenge + qas_to_challenges[qa]
            train, test = read_challenge(text, flatten=0, E2E=0)
            train_acc += train
            test_acc += test
        
    return train_acc, test_acc

def get_word_idx(ver):
    train_acc, test_acc = get_raw_data(ver, ALLQA)
    word_idx, _, _, _, _ = calc_challenge(train_acc, test_acc, flatten=0, E2E=0)
    
    return word_idx

widxfilename = FILE_DIR + "babiword_idx_{}.txt"

def dump_word_idx_to_file(word_idx,  ver):
    ver = ver.replace('/', '_')
    fname = widxfilename.format(ver)
    with open(fname, "w") as file:
        print(repr(word_idx), file=file)

def word_idx_from_file(ver):
    ver = ver.replace('/', '_')
    fname = widxfilename.format(ver)
    if os.path.exists(fname):
        with open(fname, "r") as file:
            fdata = file.read()
            word_idx = eval(fdata)
            return word_idx
    else:
        return None

def get_and_dump_word_idx(ver):
    word_idx = word_idx_from_file(ver)
    if not word_idx:
        word_idx = get_word_idx(ver)
        dump_word_idx_to_file(word_idx, ver)
    return word_idx

def get_context_data(word_idx, ver, qas, dic=None, limit=None):
    train_acc, test_acc = get_raw_data(ver, qas)
    data = train_acc + test_acc
    data = reduce(lambda x, y: x + y, [s + [q] for s, q, a in data])
    if dic:
        data = [[ran_translate(dic, w.strip()) for w in sent] for sent in data]
    func = lambda x: list(zip(x[1:], x[0:-1])) + list(zip(x[0:-1], x[1:]))
    data = reduce(lambda x, y: x + y, map(func, data))
    data = list(set(data))
    np.random.shuffle(data)
    data = ransubset(data, limit)
    data = list(zip(*data))
    data[0] = list(map(lambda x:word_idx.get(x,0), data[0]))
    data[1] = list(map(lambda x:[word_idx.get(x,0)], data[1]))
    
    return data

filename = FILE_DIR + "babicontext_data_{}_{}.txt"

def dump_context_to_file(data, ver, qa):
    ver = ver.replace('/', '_')
    fname = filename.format(ver, qa)
    with open(fname, "w") as file:
        print(repr(data), file=file)

def data_from_context_file(ver, qa):
    ver = ver.replace('/', '_')
    fname = filename.format(ver, qa)
    if os.path.exists(fname):
        with open(fname, "r") as file:
            fdata = file.read()
            data = eval(fdata)
            return data
    else:
        return None

def get_and_dump_data(word_idx, ver, qas):
    data = [[],[]]
    dic = None
    if '/' in ver:
        dic = prep_dic(word_idx)
    for qa in qas:
        temp = data_from_context_file(ver, qa)
        if not temp:
            temp = get_context_data(word_idx, ver, [qa], dic=dic)
            dump_context_to_file(temp, ver, qa)
        data = [d + t for d, t in zip(data, temp)]
    return data

def skip_gram_pre_train_emb(shape, data):
    sess = tf.Session()
    with sess.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        
        embeddings = tf.Variable(
            tf.random_uniform(shape))
        nce_weights = tf.Variable(
            tf.truncated_normal(shape,stddev=1.0 / math.sqrt(shape[1])))
        nce_biases = tf.Variable(tf.zeros(shape[0]))
        
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=64,
                     num_classes=shape[0]))
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
        
        sess.run(tf.global_variables_initializer())
        
        for batch_input, batch_output in data:
            feed_dict = {train_inputs: batch_input, train_labels: batch_output}
            _, _, embedding_matrix = sess.run([optimizer, loss, embeddings], feed_dict=feed_dict)
        
        return embedding_matrix

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

class transdict(dict):
    _translit = None
    def __init__(self, *args):
        self._translit = {}
        dict.__init__(self, *args)
    
    def find_matching_key(self, key):
        tkey = transliterate(key).strip()
        best = None
        bestr = 0
        for pkey in sorted(self.keys()):
            cr = similar(tkey, pkey)
            if (cr > bestr):
                best = pkey
                bestr = cr
        return best, cr

    def __getitem__(self, key):
        okey = key
        cr = -1
        if (not key in self):
            if (not key in self._translit):
                self._translit[key], cr = self.find_matching_key(key)
            key = self._translit[key]
        #print(okey, transliterate(okey), key, cr)
        return dict.__getitem__(self, key)
    
    def get(self, key, *args):
        okey = key;
        cr = -1
        if (not key in self):
            if (not key in self._translit):
                self._translit[key], cr = self.find_matching_key(key)
            key = self._translit[key]
        #print(okey, transliterate(okey), key, cr)
        return dict.get(self, key, *args)

class transdict_stl(dict):
    _translit = None
    def __init__(self, *args):
        self._translit = {}
        dict.__init__(self, *args)
    
    def find_matching_key(self, key):
        tkey = transliterate(key).strip()
        best = None
        bestr = 0
        for pkey in sorted(self.keys()):
            cr = similar(tkey, pkey)
            if (cr > bestr):
                best = pkey
                bestr = cr
        return best, cr
    
    def __getitem__(self, key):
        if (not key in _translit):
            self._translit[key], cr = self.find_matching_key(key)
        return dict.__getitem__(self, self._translit[key])
    
    def get(self, key, *args):
        if (not key in self._translit):
            self._translit[key], cr = self.find_matching_key(key)
        return dict.get(self, self._translit[key], *args)
    
def read_projected_file(pname):
    emb = {}
    with open(pname, 'r') as f:
        for line in f.readlines():
            data = line.split(' ')
            word = data[0]
            data = data[1:]
            vec = list(map(float,data))
            emb[word] = vec
    return emb

def external_wrapper_function(ver):
    if '/' in ver:
        func = lambda x:x
        vers = list(map(lambda x: x.replace('hn', 'hi'), ver.split('/')))
        home = path.expanduser('~')
        gemb = {}
        for v in vers:
            emb = read_projected_file(path.join(home, 'Dokumenter/git/crossling-embed/crosslingual-cca-master/data/out_%s_projected.txt' % v))
            gemb.update(emb)
        words = []
        vecs = []
        for k, v in sorted(emb.items(), key=lambda x: x[1]):
            words.append(k)
            vecs.append(v)
        widx = {w: i for i, w in enumerate(words)}
        return vecs, transdict(widx)
    else:
        return external_polygot_embedding(ver)

def external_polygot_embedding(ver):
    ver = ver.replace('hn', 'hi')
    home = path.expanduser('~')
    emb = Embedding.load(path.join(home, 'polyglot_data/embeddings2/%s/embeddings_pkl.tar.bz2' % ver))
    word_idx = {w: i for i, w in enumerate(emb.words)}
    if (ver == 'hi'):
        word_idx = transdict_stl(word_idx)
    embedding = emb.vectors
    return embedding, word_idx

def projection_polyglot_embedding(ver):
    ver = ver.replace('hn', 'hi');
    home = path.expanduser('~');
    emb = read_projected_file(path.join(home, 'Dokumenter/git/crossling-embed/crosslingual-cca-master/data/out_%s_projected.txt' % ver))
    words = []
    vecs = []
    for k, v in emb.items():
        words.append(k)
        vecs.append(v)
    widx = {w: i for i, w in enumerate(words)}
    if (ver == 'hi'):
        widx = transdict(widx)
    return vecs, widx

def pre_train_embedding(embedding_size, ver, method='skip_gram', limit=None):
    if (method == 'projected'):
        if ('/' in ver):
            method = 'external'
        embedding, widx = projection_polyglot_embedding(ver)
        embedding = np.array([vec[:embedding_size] for vec in embedding])
        return embedding, widx
    if (method == 'external'):
        # using polygot external embeddings
        embedding, widx = external_wrapper_function(ver)
        embedding = np.array([vec[:embedding_size] for vec in embedding])
        return embedding, widx
    if (method == 'skip_gram'):
        word_idx = get_and_dump_word_idx(ver)
        data = get_and_dump_data(word_idx, ver, ALLQA)
        batches = zip(np.array_split(data[0], 32), np.array_split(data[1], 32))
        shape = [len(word_idx)+1, embedding_size]
        embedding =  skip_gram_pre_train_emb(shape, batches)
        return embedding[1:], word_idx
