import time
import numpy as np
from collections import defaultdict

from gensim.models import Word2Vec

# from sklearn.cluester import Kmeans
# from utils.filesystem import save_pickle, load_pickle

SOS_TOKEN = '<SOS>'
PAD_TOKEN = '<PAD>'
EOS_TOKEN = '<EOS>'
TOKENS = [SOS_TOKEN, PAD_TOKEN, EOS_TOKEN]


class Vocab:
    def __init__(self, data):
        w2i, i2w, w2v, i2w_infreq, w2w_infreq, c2w_infreq = train_embeddings(data)
        self.w2i = w2i
        self.i2w = i2w
        self.w2v = w2v
        self.i2w_infreq = i2w_infreq
        self.w2w_infreq = w2w_infreq
        self.c2w_infreq = c2w_infreq

        self.size = len(self.w2i)

    def get_size(self):
        return self.size

    def get_effective_size(self):
        return len(self.w2i)


def calculate_frequencies(sentences):
    '''
    Function to return the frequency of each word (fragment) in all sentences (molecules)
    :param sentences: pd.series [N]
    :return: w2f: dict [number of unique words]
    '''
    w2f = defaultdict(int)

    for sentence in sentences:
        for word in sentence:
            w2f[word] += 1

    return w2f


def train_embeddings(data):
    '''
    Function to train the fragment embeddings
    :param data: pd.DataFrame [N x Number of Features]
    :return: w2i: dict, i2w: dict, i2w_infreq, w2w_infreq, c2w_infreq
    '''
    start = time.time()
    print("Training and clustering embeddings...", end=" ")
    embed_size = 64
    embed_window = 3
    mask_freq = 2
    use_mask = True

    i2w_infreq = None
    w2w_infreq = None
    c2w_infreq = None
    start_idx = len(TOKENS)

    if use_mask:
        sentences = [s.split(" ") for s in data.fragments]

        # get word embedding
        w2v = Word2Vec(
            sentences,
            vector_size=embed_size,
            window=embed_window,
            min_count=1,
            negative=5,
            workers=20,
            epochs=10,
            sg=1)

        vocab = w2v.wv.key_to_index

        # get the dictionary of the frequency of each word
        w2f = calculate_frequencies(sentences)

        # get the dictionary of word : index and vice versa from the vocabulary
        w2i = {k: v for (k, v) in vocab.items()}
        i2w = {v: k for (k, v) in w2i.items()}

        # get infrequent indices into a list
        infreq = [w2i[w] for (w, freq) in w2f.items() if freq <= mask_freq]

        # generate a dictionary of infrequent words according to the count
        i2w_infreq = {}
        for inf in infreq:
            word = i2w[inf]
            i2w_infreq[inf] = f"cluster{w2f[word]}_{word.count('*')}"

        # generate a dictionary of infrequent words and masked word instructions
        w2w_infreq = {i2w[k]: v for (k, v) in i2w_infreq.items()}

        # generate a dictionary of clusters as a list of words that have the same frequency
        c2w_infreq = defaultdict(list)
        for word, cluster_name in w2w_infreq.items():
            c2w_infreq[cluster_name].append(word)

        # substitute infrequent words with cluster words
        data = []
        for sentence in sentences:
            sentence_sub = []
            for word in sentence:
                if word in w2w_infreq:
                    # substitute the actual word with the cluster instruction string
                    word = w2w_infreq[word]
                sentence_sub.append(word)
            data.append(sentence_sub)
    else:
        sentences = [s.split(" ") for s in data.fragments]
    w2i = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}

    w2v = Word2Vec(
        sentences,
        vector_size=embed_size,
        window=embed_window,
        min_count=1,
        negative=5,
        workers=20,
        epochs=10,
        sg=1)
    vocab = w2v.wv.key_to_index
    w2i.update({k: v + start_idx for (k, v) in vocab.items()})
    i2w = {v: k for (k, v) in w2i.items()}
    end = time.time() - start
    elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
    print(f'Done. Time elapsed: {elapsed}.')
    return w2i, i2w, w2v, i2w_infreq, w2w_infreq, c2w_infreq