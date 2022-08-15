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
        data