# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import logging
import argparse
import datetime

import numpy as np
import theano
import theano.tensor as T
import lasagne

from lstm import LSTMLayer
from tlstm1 import TLSTM1Layer
from tlstm2 import TLSTM2Layer
from tlstm3 import TLSTM3Layer
from plstm import PLSTMLayer, PLSTMTimeGate
from tgate import OutGate, TimeGate
import ConfigParser
import pickle

config = ConfigParser.ConfigParser()
config.read('vmain.cfg')
BASE_DIR = config.get('data_path', 'BASE_DIR')
USER_RECORD_PATH = config.get('data_path', 'USER_RECORD_PATH')
DELTA_TIME_PATH = config.get('data_path', 'DELTA_TIME_PATH')
ACC_TIME_PATH = config.get('data_path', 'ACC_TIME_PATH')

# 下面的两个INDEX好像并没有用到,因为前期的处理已经将ITEM转化成INDEX
INDEX2WORD_PATH = config.get('data_path', 'INDEX2WORD_PATH')
WORD2INDEX_PATH = config.get('data_path', 'WORD2INDEX_PATH')


def softmax(x):
    # return softmax on x
    x = np.array(x)
    x /= np.max(x)
    e_x = np.exp(x)
    out = e_x / e_x.sum()
    return out


def sigmoid(x):
    # return sigmoid on x
    x = np.array(x)
    out = 1. / (1 + np.exp(-x))
    return out


def load_data(data_attr):
    # max_len = 0,# Max length of setence
    # vocab_size = 0, # vocabulary size
    # debug=False, # return a small set if True
    # val_num=100,  # number of validation sample
    # with_time=False, # return time information
    # with_delta_time=False # return delta time if True else if with_time == True return time
    # return: two dictionary
    # train = {'x':..., 'y':..., 't':...}
    # test = {'x':..., 'y':..., 't':...}
    # v:未做注释,因为还不知道是什么意思
    max_len = data_attr.get('max_len', 10000)
    vocab_size = data_attr.get('vocab_size', 20000)
    debug = data_attr.get('debug', False)
    with_time = data_attr.get('with_time', False)
    with_delta_time = data_attr.get('with_delta_time', False)
    data_source = data_attr.get('source', 'game')

    logging.info('Load data using max_len {}, vocab_size {}'.format(max_len, vocab_size))

    # v:可能是如果时间with_time和with_delta_time有一个是true,就表示为true
    with_time = with_time or with_delta_time

    '''v
    # data_source 默认 game
    # prefix : tr_或者te_
    item_seq = [,,,,]
    :return sentences = [item_seq1,item_seq2]
    '''

    def load_file(data_source, prefix, debug=False):
        sentences = []
        user_record_path = os.path.join(BASE_DIR, data_source, prefix + USER_RECORD_PATH)

        # 这里好像的意思是将文件中所有的用户的数据加到一个矩阵中[[]]
        if os.path.exists(user_record_path):
            with open(user_record_path, 'r') as f:
                count = 0
                for line in f:
                    userid, item_seq = line.strip().split(',')
                    item_seq = [int(x) for x in item_seq.split(' ')]
                    # 将用户的全部序列叠加
                    sentences.append(item_seq)
                    count += 1
                    # use a small subset if debug on
                    # 如果是在debug模式,则用50条序列测试
                    if debug and count == 50:
                        break
        else:
            logging.info('{} not exists!'.format(user_record_path))
            exit()

        time_seq = None
        time_file_path = os.path.join(BASE_DIR, data_source, prefix + ACC_TIME_PATH)
        if with_delta_time:
            time_file_path = os.path.join(BASE_DIR, data_source, prefix + DELTA_TIME_PATH)

        if with_time and os.path.exists(time_file_path):
            time_seq = []
            with open(time_file_path, 'r') as f:
                count = 0
                for line in f:
                    userid, delta = line.strip().split(',')
                    delta = [float(x) for x in delta.split(' ')]
                    if len(delta) != len(sentences[count]):
                        logging.info('Data conflict at line {}, delete'.format(count))
                        del sentences[count]
                        continue
                    time_seq.append(delta)
                    count += 1
                    if debug and count == 50:
                        break
        elif with_time:
            logging.info('Time record not found')
            exit()

        return sentences, time_seq

    '''v
    vocab_size:默认为20000,
    单条用户物品长度(one-hot)不超过vocab_size
    '''

    def remove_large_word(sentences, vocab_size, time_seq=None):
        # remove the word which is larger than max
        if time_seq is not None:
            sents_ret = []
            dt_ret = []
            pre_time = 0
            # check whether the word is in vocabulary list.
            # if not, we should add the delta time to the next word
            for sent, delta_time in zip(sentences, time_seq):
                _sent = []
                _dt = []
                for word, delta in zip(sent, delta_time):
                    if word < vocab_size:
                        _sent.append(word)
                        _dt.append(pre_time + delta)
                        pre_time = 0
                    else:
                        pre_time += delta
                assert (len(_sent) == len(_dt))
                sents_ret.append(_sent)
                dt_ret.append(_dt)
            return sents_ret, dt_ret
        else:
            return [filter(lambda word: word < vocab_size, sent) for sent in sentences], None

    '''v
    max_len:默认值10000
    '''

    def cut_sentences(sentences, max_len, time_seq=None):
        # remove the sentences: len < 2 and len > max_len
        dt_ret = None
        if max_len:
            sents_ret = [sent[:max_len] for sent in sentences]
            if time_seq is not None:
                dt_ret = [delta_time[:max_len] for delta_time in time_seq]
        else:
            sents_ret = sentences
            if time_seq is not None:
                dt_ret = time_seq

        return sents_ret, dt_ret

    '''v
    最后测试数据是否合理
    但是我觉得完全没有比必要
    '''

    def check(sentences, time_seq=None):
        # show the data statics
        logging.info('-------------------------------')
        max_len = 0
        total = 0
        lengths = []
        if time_seq is not None:
            for delta_time, sent in zip(time_seq, sentences):
                assert (len(delta_time) == len(sent))
        for sent in sentences:
            length = len(sent)
            lengths.append(length)
            total += length
            max_len = max_len if max_len > length else length
        if len(lengths) > 0:
            logging.info('-  Sentence number: {}'.format(len(sentences)))
            logging.info('-  Max setence length {}'.format(max_len))
            logging.info('-  average sentence length {}'.format(total * 1. / len(lengths)))
            logging.info('-  90% length {}'.format(sorted(lengths)[int(len(lengths) * 0.9)]))
        logging.info('------------------------------')

    '''
    single_y:true 表示只取最后一个作为测试
    single_y:false 表示每一个对应的y都是x+1
    '''

    def generate_x_y(sentences, time_seq=None, single_y=True):
        if single_y:
            x = [sent[:-1] for sent in sentences]
            y = [sent[-1] for sent in sentences]
        else:
            x = [sent[:-1] for sent in sentences]
            y = [sent[1:] for sent in sentences]
        t = None
        if time_seq is not None:
            t = [delta_time[:-1] for delta_time in time_seq]

        return x, y, t

    train_data, train_time_seq = load_file(data_source, 'tr_', debug)
    test_data, test_time_seq = load_file(data_source, 'te_', debug)

    logging.info('Remove large word')
    if vocab_size:
        train_data, train_time_seq = remove_large_word(train_data, vocab_size, train_time_seq)
        test_data, test_time_seq = remove_large_word(test_data, vocab_size, test_time_seq)

    # remove data which is too short
    train_data = filter(lambda sent: len(sent) > 1, train_data)
    # We need test data has more history informatiion
    test_data = filter(lambda sent: len(sent) > 2, test_data)
    if with_time:
        train_time_seq = filter(lambda delta_time: len(delta_time) > 1, train_time_seq)
        test_time_seq = filter(lambda delta_time: len(delta_time) > 2, test_time_seq)

    logging.info('cut sentences')
    train_data, train_time_seq = cut_sentences(train_data, max_len, train_time_seq)
    test_data, test_time_seq = cut_sentences(test_data, max_len, test_time_seq)
    check(train_data)

    '''v
    xtr=[[],[]]
    ytr=[[],[]]
    xte=[[],[]]
    yte=[]
    '''
    xtr, ytr, ttr = generate_x_y(train_data, train_time_seq, single_y=False)
    xte, yte, tte = generate_x_y(test_data, test_time_seq)

    logging.info('Train data:{}'.format(len(xtr)))
    logging.info('Test data:{}'.format(len(xte)))

    train = {'x': xtr, 'y': ytr}
    test = {'x': xte, 'y': yte}
    if with_time:
        train['t'] = ttr
        test['t'] = tte
    # 将训练集和测试集最后以集合的形式输出
    return train, test


def prepare_data(data, vocab_size, one_hot=False, sigmoid_on=False):
    '''
    convert list of data into numpy.array
    padding 0
    generate mask
    '''
    x_origin = data['x']
    y_origin = data['y']
    t_origin = data.get('t', None)
    ndim = 1 if not one_hot else vocab_size

    lengths_x = [len(s) for s in x_origin]
    n_samples = len(x_origin)  # test:20
    max_len = np.max(lengths_x)  # test:5122

    x = np.zeros((n_samples, max_len)).astype('int32')
    t = np.zeros((n_samples, max_len)).astype('float')
    mask = np.zeros((n_samples, max_len)).astype('float')
    for idx, sent in enumerate(x_origin):
        x[idx, :lengths_x[idx]] = sent
        mask[idx, :lengths_x[idx]] = 1.
        if t_origin is not None:
            tmp_t = t_origin[idx]
            if sigmoid_on:
                tmp_t = sigmoid(tmp_t)
            t[idx, :int(np.sum(mask[idx]))] = tmp_t

    if type(y_origin[0]) is list:
        # train
        y = np.zeros((n_samples, max_len)).astype('int32')
        lengths_y = [len(s) for s in y_origin]
        for idx, sent in enumerate(y_origin):
            y[idx, :lengths_y[idx]] = sent
    else:
        # test
        y = np.array(y_origin).astype('int32')

    # v:将x的物品序列转化为one-hot序列
    # 原来的x:[[1 9 5 ...],[...]]
    # 转化后的x:[[[1 0 0 0 ..],[0 0 0 0 ...]...],[[...],[...]]]
    if one_hot:
        # n_samples:案例数量    max_len:物品序列长度   vocab_size:物品one-hot向量的长度
        one_hot_x = np.zeros((n_samples, max_len, vocab_size)).astype('int32')
        for i in range(n_samples):
            for j in range(max_len):
                one_hot_x[i, j, x[i, j]] = 1
        x = one_hot_x
    else:
        x = x.reshape(x.shape[0], x.shape[1], ndim)

    # mask:[[1 1 1 1 1...0 0 0 0 0],[1 1 1 ... 0 0]] 1的个数表示物品的长度
    # lengths_x:[1519 1596 ...] 每一个数字表示用户的序列长度
    # y:next game id的list [0 0 0 1 0 ...] 0为英雄联盟
    ret = {'x': x, 'y': y, 'mask': mask, 'lengths': lengths_x}
    if t_origin is not None:
        ret['t'] = t

    return ret


def save_model(filename, suffix, model, log=None, announce=True, log_only=False):
    # Build filename
    filename = '{}_{}'.format(filename, suffix)
    # Store in separate directory
    filename = os.path.join('./models/', filename)
    # Inform user
    if announce:
        logging.info('Saving to: {}'.format(filename))
    # Generate parameter filename and dump
    param_filename = '%s.params' % (filename)
    if not log_only:
        # Acquire Data
        data = lasagne.layers.get_all_param_values(model)
        with open(param_filename, 'w') as f:
            pickle.dump(data, f)
    # Generate log filename and dump
    if log is not None:
        log_filename = '%s.log' % (filename)
        with open(log_filename, 'w') as f:
            pickle.dump(log, f)
