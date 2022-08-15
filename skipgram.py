import time
import numpy as np
from gensim.models import Word2Vec
from sklean.cluester import Kmeans
from utils.filesystem import save_pickle, load_pickle

SOS_TOKEN = '<SOS>'
PAD_TOKEN = '<PAD>'
EOS_TOKEN = '<EOS>'
TOKENS = [SOS_TOKEN, PAD_TOKEN, EOS_TOKEN]

class Vocab:
    def __init__(self, data):
        w2i, i2w, i2w_infreq, w2w_infreq, c2w_infreq = train_embeddings(config, data)
        self.w2i = w2i
        self.i2w = i2w
        self.i2w_infreq = i2w_infreq
        self.w2w_infreq = w2w_infreq
        self.w2w_infreq = w2s_infreq
        self.c2w_infreq = c2w_infreq

        self.size = len(self.w2i)

    def get_size(self):
        return self.size
    def get_effective_size(self):
        return len(self.w2i)

def train_embeddings(config, data):
    start = time.time()
    print("Training and clustering embeddings...", end = " ")
    embed_size = 64
    embed_window = 3
    mask_freq = 2
    use_mask = True

    i2w_infreq = None
    w2w_infreq = None
    c2w_infreq = None
    start_idx = len(TOKENS)

    if use_mask:
    else:
        data = [s.split(" ") for s in data.fragments]
    w2i = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}

    w2v = Word2Vec(
            data,
            vector_size = embed_size,
            window = embed_window,
            min_count = 1,
            negative = 5,
            workers = 20,
            epochs = 10,
            sg = 1)

