# encoding:utf-8
from __future__ import print_function
import pandas as pd
import pickle
import os

BASE_DIR = 'data'
DATA_SOURCE = 'game'
user_detail_path = os.path.join(BASE_DIR, DATA_SOURCE, 't500_1000USERDETAIL.csv')
user_item_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item.lst')
user_item_delta_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-delta-time.lst')
user_item_accumulate_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-accumulate-time.lst')
index2game_path = os.path.join(BASE_DIR, DATA_SOURCE, 'index2item')
game2index_path = os.path.join(BASE_DIR, DATA_SOURCE, 'item2index')
# 如果存在,直接读取
if os.path.exists(index2game_path) and os.path.exists(user_item_record):
    index2game = pickle.load(open(index2game_path, 'rb'))
    print('Total game %d' % len(index2game))
    user_item_list = pd.read_csv(user_item_record,header=None)
    print('Total user %d' % len(user_item_list))
    exit(0)

def generate_data():
    out_ui = open(user_item_record, 'w')
    out_uidt = open(user_item_delta_time_record, 'w')
    out_uiat = open(user_item_accumulate_time_record, 'w')

    df = pd.read_csv(user_detail_path,
                     error_bad_lines=False,
                     header=None,
                     names=['timestamp', 'user_id', 'game_name', 'game_type', 'duration'])

    # 如果存在,直接读取
    if os.path.exists(index2game_path) and os.path.exists(game2index_path):
        index2game = pickle.load(open(index2game_path, 'rb'))
        game2index = pickle.load(open(game2index_path, 'rb'))
        print('Total game %d' % len(index2game))
    else:
        print('Build index2game')
        # 将数据按照game分类,在通过记录数量排序
        sorted_series = df.groupby(['game_name']).size().sort_values(ascending=False)
        index2game = sorted_series.keys().tolist()
        print('Most common game is "%s":%d' % (index2game[0], sorted_series[0]))
        print('build game2index')
        # 反向构造game2index的dict
        game2index = dict((v, i) for i, v in enumerate(index2game))
        pickle.dump(index2game, open(index2game_path, 'wb'))
        pickle.dump(game2index, open(game2index_path, 'wb'))

    print('start loop')

    count = 0
    user_group = df.groupby(['user_id'])
    for user_id, length in user_group.size().sort_values().iteritems():
        if count % 10 == 0:
            print("=====count %d======" % count)
        count += 1
        print('%s %d' % (user_id, length))
        # 对没有的用户的游戏事件,通过时间排序
        user_data = user_group.get_group(user_id).sort_values(by='timestamp')
        # 取出游戏名称的序列和时间
        game_seq = user_data['game_name']
        time_seq = user_data['timestamp']
        # 将其中的空值去除
        game_seq = game_seq[game_seq.notnull()]
        time_seq = time_seq[time_seq.notnull()]
        # diff(-1)表示与下一个时间做对比, *-1是因为这个之是一个负数
        delta_time = pd.to_datetime(time_seq).diff(-1).astype('timedelta64[s]') * -1
        # apply()表示沿序列应用函数,这里表示将game_seq中的game_name换成game_id
        game_seq = game_seq.apply(lambda x: game2index[x] if pd.notnull(x) else -1).tolist()
        delta_time = delta_time.tolist()
        delta_time[-1] = 0

        # 用于计算累计量
        time_accumulate = [0]
        for delta in delta_time[:-1]:
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)

        out_ui.write(str(user_id) + ',')
        out_ui.write(' '.join(str(x) for x in game_seq) + '\n')
        out_uidt.write(str(user_id) + ',')
        out_uidt.write(' '.join(str(x) for x in delta_time) + '\n')
        out_uiat.write(str(user_id) + ',')
        out_uiat.write(' '.join(str(x) for x in time_accumulate) + '\n')

    out_ui.close()
    out_uidt.close()
    out_uiat.close()


if __name__ == '__main__':
    generate_data()
