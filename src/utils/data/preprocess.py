import os
import re
import sys
import h5py
import math
import json
import torch
import gensim
import random
import itertools
import unicodedata
import numpy as np
import pandas as pd
from itertools import zip_longest
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import matplotlib.pyplot as plt


def process_raw(config):
    """
    convert raw datafiles from different datasets:
        how2 => machine translation
        daily-dialog => dialog generation
    to one desired format
    """
    if config['dataset'] == 'sst1':
        train_data, val_data, test_data = process_sst1()
    elif config['dataset'] == '20ng':
        train_data, test_data = process_20ng()

    # Save the processed files
    with open('{}/{}/train.txt'.format(config['data_dir'], config['dataset']), 'w') as f:
        f.write(train_data)
    # with open('{}/{}/val.txt'.format(config['data_dir'], config['dataset']), 'w') as f:
    #    f.write(val_data)
    with open('{}/{}/test.txt'.format(config['data_dir'], config['dataset']), 'w') as f:
        f.write(test_data)


def process_sst1():
    with open('data/raw/sst1/datasetSentences.txt') as f:
        dataSentences = f.readlines()
    with open('data/raw/sst1/datasetSplit.txt') as f:
        dataSplit = f.readlines()
    with open('data/raw/sst1/sentiment_labels.txt') as f:
        sentimentLabels = f.readlines()

    train_data, val_data, test_data = '', '', ''
    split_dict = {}
    for s in dataSplit[1:]:
        s = s.strip()
        pid, split_label = s.split(',')
        split_dict[pid] = int(split_label)

    sentiment_dict = {}
    for s in sentimentLabels[1:]:
        s = s.strip()
        pid, sentiment_value = s.split('|')
        sentiment_dict[pid] = float(sentiment_value)

    for s in dataSentences[1:]:
        s = s.strip()
        pid, phrase = s.split('\t')
        label = get_sentiment_label(sentiment_dict[pid])
        mode = split_dict[pid]
        if mode == 1:
            train_data += '{}\t{}\n'.format(phrase, label)
        elif mode == 2:
            test_data += '{}\t{}\n'.format(phrase, label)
        elif mode == 3:
            val_data += '{}\t{}\n'.format(phrase, label)
    return train_data, val_data, test_data


def process_20ng():
    with open('data/raw/20ng/20ng-train-all-terms.txt') as f:
        data_att1 = f.readlines()
    with open('data/raw/20ng/20ng-test-all-terms.txt') as f:
        data_att2 = f.readlines()

    train_data, test_data = '', ''

    for s in data_att1:
        s = s.strip()
        category, news = s.split('\t')
        cat = get_category(category)
        if cat is None:
            continue
        train_data += '{}\t{}\n'.format(news, cat)

    for s in data_att2:
        s = s.strip()
        category, news = s.split('\t')
        cat = get_category(category)
        if cat is None:
            continue
        test_data += '{}\t{}\n'.format(news, cat)
    return train_data, test_data


def get_sentiment_label(sentiment_val):
    if 0 <= sentiment_val <= 0.2:
        return 0
    elif 0.2 < sentiment_val <= 0.4:
        return 1
    elif 0.4 < sentiment_val <= 0.6:
        return 2
    elif 0.6 < sentiment_val <= 0.8:
        return 3
    elif 0.8 < sentiment_val <= 1:
        return 4


def get_category(category_type):
    if 'comp' in category_type:
        return 0
    elif 'politics' in category_type:
        return 1
    elif 'rec' in category_type:
        return 2
    elif 'religion' in category_type:
        return 3
    else:
        return None


# Vocab and file-reading part
class Vocabulary(object):
    """Vocabulary class"""
    def __init__(self):
        super(Vocabulary, self).__init__()
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4  # count the special tokens above

    def add_sentence(self, sentence):
        for word in sentence.strip().split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.word2count[word] = 1
            self.index2word[self.size] = word
            self.size += 1
        else:
            self.word2count[word] += 1

    def sentence2index(self, sentence):
        indexes = []
        for w in sentence.split():
            try:
                indexes.append(self.word2index[w])
            except KeyError as e:  # handle OOV
                indexes.append(self.word2index['<UNK>'])
        return indexes

    def index2sentence(self, indexes):
        return [self.index2word[i] for i in indexes]


def build_vocab(config):
    all_pairs = read_pairs('train', config)
    all_pairs += read_pairs('test', config)
    vocab = Vocabulary()
    for pair in all_pairs:
        vocab.add_sentence(pair[0])
    print('Vocab size: {}'.format(vocab.size))
    np.save(config['vocab_path'], vocab, allow_pickle=True)
    return vocab


def read_pairs(mode, config):
    """
    Reads src-target sentence pairs given a mode
    """
    # if mode == 'train' / 'test'
    with open('data/processed/{}/{}.txt'.format(config['dataset'], mode), 'r') as f:
        dataset = f.readlines()

    pairs = []
    for s in dataset:
        s = s.strip()
        phrase, label = s.split('\t')
        phrase = ' '.join(phrase.split()[:config['MAX_LENGTH']])
        pairs.append((normalize_string(phrase), int(label)))
    return filter_pairs(pairs, config['MAX_LENGTH'])


def normalize_string(x):
    """Lower-case, strip and remove non-letter characters
    ==============
    Params:
    ==============
    x (Str): the string to normalize
    """
    x = unicode_to_ascii(x.lower().strip())
    x = re.sub(r'([.!?])', r'\1', x)
    x = re.sub(r'[^a-zA-Z.!?]+', r' ', x)
    return x


def unicode_to_ascii(x):
    return ''.join(
        c for c in unicodedata.normalize('NFD', x)
        if unicodedata.category(c) != 'Mn')


def filter_pairs(pairs, max_len):
    """
    Filter pairs with either of the sentence > max_len tokens
    ==============
    Params:
    ==============
    pairs (list of tuples): each tuple is a src-target sentence pair
    max_len (Int): Max allowable sentence length
    """
    return [pair for pair in pairs if (len(pair[0].split()) <= max_len)]


# Embeddings part
def generate_word_embeddings(vocab, config):
    # Load original (raw) embeddings
    ftype = 'bin'

    # Train w2v models if not already trained
    train_w2v_model(config)

    src_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
        'data/raw/english_w2v.bin', binary=True)

    # Create filtered embeddings
    # Initialize filtered embedding matrix
    combined_embeddings = np.zeros((vocab.size, config['embedding_dim']))
    for index, word in vocab.index2word.items():
        try:  # random normal for special and OOV tokens
            if index <= 4:
                combined_embeddings[index] = \
                    np.random.normal(size=(config['embedding_dim'], ))
                continue  # use continue to avoid extra `else` block
            combined_embeddings[index] = src_embeddings[word]
        except KeyError as e:
            combined_embeddings[index] = \
                np.random.normal(size=(config['embedding_dim'], ))

    with h5py.File(config['filtered_emb_path'], 'w') as f:
        f.create_dataset('data', data=combined_embeddings, dtype='f')
    return torch.from_numpy(combined_embeddings).float()


def train_w2v_model(config):
    all_pairs = read_pairs('train', config)
    random.shuffle(all_pairs)
    src_sentences = []
    for pair in all_pairs:
        src_sentences.append(pair[0].split())

    src_w2v = gensim.models.Word2Vec(src_sentences, size=300,
                                     min_count=1, iter=50)
    src_w2v.wv.save_word2vec_format('data/raw/english_w2v.bin', binary=True)


def load_word_embeddings(config):
    with h5py.File(config['filtered_emb_path'], 'r') as f:
        return torch.from_numpy(np.array(f['data'])).float()


def btmcd(vocab, sentences):
    """
    batch_to_model_compatible_data
    Returns padded source and target index sequences
    ==================
    Parameters:
    ==================
    vocab (Vocabulary object): Vocabulary built from the dataset
    pairs (list of tuples): The source and target sentence pairs
    """
    pad_token = vocab.word2index['<PAD>']
    eos_token = vocab.word2index['<EOS>']
    src_indexes, src_lens = [], []
    for sent in sentences:
        src_indexes.append(vocab.sentence2index(sent) + [eos_token])
        src_lens.append(len(sent.split()))

    # pad the batches
    src_indexes = pad_indexes(src_indexes, value=pad_token)
    src_lens = torch.tensor(src_lens)
    return src_indexes, src_lens


def pad_indexes(indexes_batch, value):
    """
    Returns a padded tensor of shape (max_seq_len, batch_size) where
    max_seq_len is the sequence with max length in indexes_batch and
    batch_size is the number of elements in indexes_batch
    ==================
    Parameters:
    ==================
    indexes_batch (list of list): the batch of indexes to pad
    value (int): the value with which to pad the indexes batch
    """
    return torch.tensor(list(zip_longest(*indexes_batch, fillvalue=value)))


def plot_confusion_matrix(targets, predictions, classes,
                          epoch, model_code, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    cm = confusion_matrix(targets, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not os.path.exists('reports/figures/{}'.format(model_code)):
        os.mkdir('reports/figures/{}'.format(model_code))
    plt.savefig('reports/figures/{}/cm_{}'.format(model_code, epoch))
    plt.close()


def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance
