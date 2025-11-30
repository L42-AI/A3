import wget, os, gzip, pickle, random, re, sys, importlib, tqdm, math, os, gzip, re

from tqdm import trange
from collections import Counter
from .c_types import Seq, X, Y, Dataset, I2W, W2I
from typing import Literal

import torch

import random
from random import choice

IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'
WP_DATA = 'http://mattmahoney.net/dc/enwik8.gz'

PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'

SENT = '_s'
TOY = {
    '_s': ['_s _adv', '_np _vp', '_np _vp _prep _np', '_np _vp ( _prep _np )', '_np _vp _con _s','_np _vp ( _con _s )'],
    '_adv': ['briefly', 'quickly', 'impatiently'],
    '_np': ['a _noun', 'the _noun', 'a _adj _noun', 'the _adj _noun'],
    '_prep': ['on', 'with', 'to', 'for', 'at'],
    '_con': ['while', 'but'],
    '_noun': ['mouse', 'bunny', 'cat', 'dog', 'man', 'woman', 'person', 'bear', 'koala', 'judge', 'businessman',
        'businesswoman', 'lawyer', 'teacher', 'engineer'],
    '_vp': ['walked', 'walks', 'ran', 'runs', 'goes', 'went', 'hiked'],
    '_adj': ['short', 'quick', 'busy', 'nice', 'gorgeous', 'spectacular', 'reluctant', 'systematic', 'willowy', 'engaged', 'synthetic']
}

def gen_sentence(sent=SENT, g=TOY):

    symb = '_[a-z]*'

    while True:

        match = re.search(symb, sent)
        if match is None:
            return sent

        s = match.span()
        sent = sent[:s[0]] + random.choice(g[sent[s[0]:s[1]]]) + sent[s[1]:]

def load_toy(ntrain=100_000, ntest=20_000, to_torch=True):
    """
    Generates language from a toy grammar.
    :param ntrain:
    :param ntest:
    :return:
    """

    train, test = '', ''
    while len(train) < ntrain:
        train += gen_sentence() + ' . '
    while len(train) < ntrain:
        test += gen_sentence() + ' . '

    ctr = Counter(train + test)
    i2t = [PAD, START, END, UNK] + [t for t, _ in ctr.most_common()]
    t2i = { w : i for  i, w in enumerate(i2t)}

    train = [t2i[t] for t in train]
    test  = [t2i[t] for t in test]
    
    if to_torch:
        return (t(train), t(test)), (i2t, t2i) # Torch vectors (this takes a few seconds)

    return (train, test), (i2t, t2i)

def t(blist):
    return torch.tensor([int(b) for b in blist], dtype=torch.uint8)

def load_wp(fname='enwik8.gz', split=(90, 5, 5), to_torch=True):
    """
    Load the enwik8 dataset from the Hutter challenge as a list or vector of bytes.
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """

    if not os.path.exists(fname):
        # If it doesn't exist, download it
        print('Downloading')
        wget.download(WP_DATA, out=fname)
        
    with gzip.open(fname, 'r') if fname.endswith('.gz') else open(fname, 'rb') as file:
        all = file.read()

        split = tuple(s/sum(split) for s in split)
        split = tuple(int(s * len(all)) for s in split)

        train, val, test = all[:split[0]], all[split[0]:split[0]+split[1]], all[split[0]+split[1]:]

        if to_torch:
            return t(train), t(val), t(test) # Torch vectors (this takes a few seconds)

        return train, val, test # Plain python lists


def load_xor(ntrain=25_000, ntest=25_000, seed=0) -> tuple[Dataset, Dataset, tuple[I2W, W2I], Literal[2]]:

    random.seed(seed)

    i2w = [PAD, START, END, UNK, 'true', 'false'] #
    w2i = {w : i for i, w in enumerate(i2w)}

    dataset, labels = [], []
    for _ in range(ntrain + ntest):
        sentence = [
            choice((i2w[4], i2w[5])),
            choice((i2w[4], i2w[5]))
        ]

        f1, f2 = (sentence[0] == i2w[4]), (sentence[1] == i2w[4]) # true: very/great false: not/terrible
        # -- these words are the only meaningful features
        label = 0 if f1 != f2 else 1

        dataset.append([w2i[word] for word in sentence])
        labels.append(label)

    return \
        (dataset[:ntrain], labels[:ntrain]), \
        (dataset[ntrain:], labels[ntrain:]), \
        (i2w, w2i), 2

def load_imdb_synth(ntrain=25_000, ntest=25_000, seed=0) -> tuple[Dataset, Dataset, tuple[I2W, W2I], Literal[2]]:
    """
    Synthetic IMDb dataset
    :param seed:
    :param voc:
    :return:
    """

    random.seed(seed)

    adjectives = ['classic', 'silent', 'modern', 'vintage', 'independent', 'foreign', 'animated', 'documentary',
    'epic', 'dramatic', 'romantic', 'comic', 'thrilling', 'mysterious', 'gritty', 'stylized', 'iconic', 'acclaimed',
    'popular', 'forgettable', 'unreleased', 'awardwinning', 'blockbuster', 'lowbudget', 'highbudget', 'experimental',
    'mainstream', 'cult', 'notable', 'original']
    nouns = ['movie', 'film', 'motion-picture', 'feature', 'featurette', 'picture', 'flick', 'cinema', 'screenplay',
    'blockbuster', 'talkie', 'silent', 'biopic', 'short', 'docudrama', 'documentary', 'animation', 'cartoon',
    'anime', 'telefilm', 'miniseries', 'drama', 'comedy', 'thriller', 'western', 'musical', 'noir']
    verbs = ['was', 'is', 'became', 'becomes', 'seemed', 'seems']

    i2w = [PAD, START, END, UNK, 'this', 'not', 'very', 'great','terrible'] + verbs + adjectives + nouns
    w2i = {w : i for i, w in enumerate(i2w)}

    dataset, labels = [], []
    for _ in range(ntrain + ntest):
        sentence = [
            i2w[4], # this
            choice(adjectives), # old
            choice(nouns), # movie
            choice(verbs), # was
            choice((i2w[5], i2w[6])),
            choice((i2w[7], i2w[8]))
        ]

        f1, f2 = (sentence[4] == i2w[6]), (sentence[5] == i2w[7]) # true: very/great false: not/terrible
        # -- these words are the only meaningful features
        label = 0 if f1 != f2 else 1

        dataset.append([w2i[word] for word in sentence])
        labels.append(label)

    return \
        (dataset[:ntrain], labels[:ntrain]), \
        (dataset[ntrain:], labels[ntrain:]), \
        (i2w, w2i), 2


def load_imdb(final=False, val=5000, seed=0, voc=None, char=False) -> tuple[Dataset, Dataset, tuple[I2W, W2I], Literal[2]]:

    cst = 'char' if char else 'word'

    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set( random.sample(range(len(sequences['train'])), k=val) )
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2

def load_wikipedia():
    pass # todo