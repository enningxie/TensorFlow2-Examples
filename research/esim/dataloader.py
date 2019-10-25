# coding=utf-8
import os
import pandas as pd
import tensorflow as tf
import numpy as np


class DataLoader(object):

    # 加载字典
    def load_char_vocab(self):
        path = os.path.join(os.path.dirname(__file__), '../data/vocab.txt')
        vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
        word2idx = {word: index for index, word in enumerate(vocab)}
        idx2word = {index: word for index, word in enumerate(vocab)}
        return word2idx, idx2word

    # 字->index
    def char_index(self, p_sentences, h_sentences, max_len, pad_mode='post'):
        word2idx, idx2word = self.load_char_vocab()

        p_list, h_list = [], []
        for p_sentence, h_sentence in zip(p_sentences, h_sentences):
            p = [word2idx[word.lower()] for word in p_sentence if
                 len(word.strip()) > 0 and word.lower() in word2idx.keys()]
            h = [word2idx[word.lower()] for word in h_sentence if
                 len(word.strip()) > 0 and word.lower() in word2idx.keys()]

            p_list.append(p)
            h_list.append(h)

        p_list = tf.keras.preprocessing.sequence.pad_sequences(p_list, maxlen=max_len, padding=pad_mode,
                                                               truncating=pad_mode)
        h_list = tf.keras.preprocessing.sequence.pad_sequences(h_list, maxlen=max_len, padding=pad_mode,
                                                               truncating=pad_mode)

        return p_list, h_list

    def load_char_data(self, data_path, mx_len, data_size=None, duplicate=False):
        path = os.path.join(os.path.dirname(__file__), '../' + data_path)
        df = pd.read_csv(path)
        p = df['sentence1'].values[0:data_size]
        h = df['sentence2'].values[0:data_size]
        label = df['label'].values[0:data_size]

        p_c_index, h_c_index = self.char_index(p, h, mx_len)

        # duplication
        if duplicate:
            p_c_index_ = np.vstack((p_c_index, h_c_index))
            h_c_index_ = np.vstack((h_c_index, p_c_index))
            label_ = np.hstack((label, label))
            return (p_c_index_, h_c_index_), label_
        else:
            return (p_c_index, h_c_index), label

