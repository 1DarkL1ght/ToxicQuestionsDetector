import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import torch
import sklearn

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torcheval.metrics import MulticlassAccuracy
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from tqdm.notebook import tqdm


nltk.download('stopwords')
nltk.download('popular')
nltk.download('punkt')
nltk.download('punkt_tab')

warnings.filterwarnings('ignore')

from detector.model import LSTM_Model

import __main__
setattr(__main__, "LSTM_Model", LSTM_Model)


class Detector:
    def __init__(self, model_path='model.pt', vocab_path='vocab.txt'):
        self.backbone = torch.load(model_path, weights_only=False)
        self.backbone.eval()
        self.device = 'cpu'
        self.vocab_path = vocab_path


    def set_device(self, device: str) -> None:
        try:
            self.device = device
            self.backbone.to(self.device)
        except:
            raise ValueError(f'No device called {self.device}')
        
    def _preprocess(self, text:str, maxlen=30) -> torch.Tensor:
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()

        def lemmatize_sentence(tokens: list) -> list:
            lemmatized_tokens = []
            for token in tokens:
                lemmatized_tokens.append(lemmatizer.lemmatize(token))
            return lemmatized_tokens

        def remove_stopwords(tokens: list) -> list:
            return [token for token in tokens if token not in stop_words]

        def read_vocab(vocab_path: str) -> dict:
            word_to_idx = {}
            with open(vocab_path, 'r', encoding='utf8') as vocab:
                for line in vocab.readlines():
                    word, idx = line.split()
                    word_to_idx[word] = int(idx)
            vocab_size = len(word_to_idx)
            return word_to_idx, vocab_size
        
        def text_to_idx(tokens: list, word_to_idx) -> list:
            return [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
        

        word_to_idx, vocab_size = read_vocab(self.vocab_path)

        text = text.lower()
        tokens = tokenizer.tokenize(text)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_sentence(tokens)
        indices = torch.tensor(text_to_idx(tokens, word_to_idx), dtype=torch.long).unsqueeze(0)
        return indices


    def __call__(self, text: str):
        model_input = self._preprocess(text)
        return nn.Sigmoid()(self.backbone(model_input.to(self.device)))
