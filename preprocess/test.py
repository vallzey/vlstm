# encoding:utf-8

from __future__ import print_function
import pandas as pd
import pickle
import os

BASE_DIR = 'data'
DATA_SOURCE = 'game'
user_detail_path = os.path.join(BASE_DIR, DATA_SOURCE, 'than1000USERDETAIL.csv')


def test_game_lsit():
    df = pd.read_csv(user_detail_path, header=None,
                     names=['timestamp', 'user_id', 'game_name', 'game_type', 'duration'])

    sorted_series = df.groupby(['game_name']).size().sort_values(ascending=False)
    index2game = sorted_series.keys().tolist()
    print(sorted_series)
    print('Most common game is "%s":%d' % (index2game[0], sorted_series[0]))


def test_user_list():
    count = 0
    df = pd.read_csv(user_detail_path, header=None,
                     names=['timestamp', 'user_id', 'game_name', 'game_type', 'duration'])
    user_group = df.groupby(['user_id'])
    for user_id, length in user_group.size().sort_values().iteritems():
        if count % 10 == 0:
            print("=====count %d======" % count)
        count += 1
        print('%s %d' % (user_id, length))
        user_data = user_group.get_group(user_id).sort_values(by='timestamp')
        game_seq = user_data['game_name']
        time_seq = user_data['timestamp']
        game_seq = game_seq[game_seq.notnull()]
        time_seq = time_seq[time_seq.notnull()]
        print(game_seq + time_seq)


def test_time_list():
    df = pd.read_csv(user_detail_path, header=None,
                     names=['timestamp', 'user_id', 'game_name', 'game_type', 'duration'])
    user_group = df.groupby(['user_id'])
    user_data = user_group.get_group(272096).sort_values(by='timestamp')
    time_seq = user_data['timestamp']
    time_seq = time_seq[time_seq.notnull()]
    delta_time = pd.to_datetime(time_seq).diff(-1).astype('timedelta64[s]') * -1
    delta_time = delta_time.tolist()
    # -1表示序列的最后一个值
    delta_time[-1] = 0
    time_accumulate = [0]
    for delta in delta_time[:-1]:
        next_time = time_accumulate[-1] + delta
        time_accumulate.append(next_time)

    print(time_accumulate)


test_time_list()
