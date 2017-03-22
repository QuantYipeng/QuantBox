import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import tushare as ts
import pandas as pd
from scipy.stats.stats import spearmanr
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tqdm import tqdm


def _pre_process(m):
    # function: pre-process m using Z-score=(x-mu)/std for every column in m
    # return: None
    for c in range(np.shape(m)[1]):
        mean = np.mean(m[:, c])
        std = np.std(m[:, c])
        # if std = 0 use sigmoid
        if std == 0:
            for r in range(np.shape(m)[0]):
                m[r, c] = 1.0 / (1 + np.exp(-float(m[r, c])))
        # else use z-score
        else:
            for r in range(np.shape(m)[0]):
                m[r, c] = (m[r, c] - mean) / std
    return


def _add_label(hist_data_with_label_sample, change):
    label_size = 0

    if change < -0.05:
        hist_data_with_label_sample.append(1)
    else:
        hist_data_with_label_sample.append(0)
    label_size += 1

    if -0.05 <= change < 0:
        hist_data_with_label_sample.append(1)
    else:
        hist_data_with_label_sample.append(0)
    label_size += 1

    if 0 <= change < 0.05:
        hist_data_with_label_sample.append(1)
    else:
        hist_data_with_label_sample.append(0)
    label_size += 1

    if 0.05 < change:
        hist_data_with_label_sample.append(1)
    else:
        hist_data_with_label_sample.append(0)
    label_size += 1

    return label_size


def _is_in_time_interval(start, target, end):
    def _t2s(_t):
        try:
            _h, _m, _s = _t.strip().split(':')
            second = int(_h) * 3600 + int(_m) * 60 + int(_s)
        except:
            second = 0
        return second

    sec_start = _t2s(start)  # change str to seconds
    sec_target = _t2s(target)
    sec_end = _t2s(end)
    if sec_start <= sec_target <= sec_end:
        return True
    else:
        return False


def _get_big_deal_statistic(code='300403', date='2017-03-17', vol=0):
    warnings.filterwarnings("ignore")

    def _change2float(change):
        if change == '--':
            return 0
        else:
            return float(change)

    results = []
    df = ts.get_tick_data(code, date)
    # for total
    for deal_type in ['买盘', '卖盘', '中性盘']:
        # Count
        results.append(len([row['volume']
                            for index, row in df.iterrows()
                            if row['type'] == deal_type
                            and row['volume'] >= vol]))
        # E(volume)
        results.append(np.nan_to_num(np.mean([row['volume']
                                              for index, row in df.iterrows()
                                              if row['type'] == deal_type
                                              and row['volume'] >= vol])))
        # E(change)
        results.append(np.nan_to_num(np.mean([_change2float(row['change'])
                                              for index, row in df.iterrows()
                                              if row['type'] == deal_type
                                              and row['volume'] >= vol])))

    # for each time interval
    for time_interval in [('09:20:00', '10:05:00'),
                          ('10:05:00', '10:35:00'),
                          ('10:35:00', '11:05:00'),
                          ('11:05:00', '11:35:00'),
                          ('12:55:00', '13:35:00'),
                          ('13:35:00', '14:05:00'),
                          ('14:05:00', '14:35:00'),
                          ('14:35:00', '15:05:00')]:
        for deal_type in ['买盘', '卖盘', '中性盘']:
            v_c = [(row['volume'], _change2float(row['change']))
                   for index, row in df.iterrows()
                   if _is_in_time_interval(time_interval[0], row['time'], time_interval[1])
                   and row['type'] == deal_type
                   and row['volume'] >= vol]
            # Count
            results.append(len(v_c))
            # E(volume)
            results.append(np.nan_to_num(np.mean([x[0] for x in v_c])))
            # E(dPrice)
            results.append(np.nan_to_num(np.mean([x[1] for x in v_c])))

    return results


def _get_data_for_predict(target='300403',
                          correlations=10,
                          days=500,
                          data_length=1,
                          recent_days=10,
                          hist_file='hist0316.pkl'):
    # parameters:
    # n = how many correlated assets will be used, n asset with biggest corr, and n asset with smallest corr
    # l = how many history days used in prediction
    # days = how many calendar days of history data should we download
    #   not calendar when using data_file, maybe short that calendar days

    # get data
    with open(hist_file, 'rb') as f:
        content = pickle.load(f)  # read file and build object

    # get hist=[target, related assets]
    hist = []
    codes_content = list(content.keys())
    if target in codes_content:
        codes_content.remove(target)
    codes_adjusted = [target] + codes_content
    for code in codes_adjusted:
        hist.append(content[code][-days:])

    # match & drop
    temp_hist = [hist[0]]
    for i in tqdm(range(len(hist)), desc='[Matching And Dropping '+target+']'):
        if i == 0:
            continue
        # left join with target dates, then fill NaN with last before_data
        temp_hist_b = pd.merge(hist[0].loc[:, ['date']], hist[i],
                               on='date', how='left').fillna(method='pad')
        # drop the hist with NaN at beginning
        if temp_hist_b.isnull().any().any():
            continue
        temp_hist.append(temp_hist_b)
    hist = temp_hist

    # get corr (n largest correlations and n lowest correlation)
    corr = []
    for i in range(len(hist)):
        corr.append(spearmanr(hist[0]['close'].values, hist[i]['close'].values)[0])
    corr_dict = dict(zip(range(len(hist))[1:], corr[1:]))  # ignore itself
    corr_dict_sorted = np.array(sorted(corr_dict.items(), key=lambda item: item[1]))  # from low to large
    corr_index_bottom = corr_dict_sorted[-correlations:, 0].tolist()
    corr_index_top = corr_dict_sorted[:correlations, 0].tolist()
    corr_index = [corr_index_top, corr_index_bottom]
    corr_index = [y for x in corr_index for y in x]
    temp_hist = [hist[0]]
    for i in corr_index:
        temp_hist.append(hist[int(i)])
    hist = temp_hist

    # get sample data and label
    hist_data_with_label = []
    label_size = 0
    # for each sample
    for i in tqdm(range(len(hist[0]) - (data_length + 1)), desc='[Preparing Samples '+target+ ']'):

        hist_data_with_label.append([])
        # add data
        for j in range(data_length):
            # weekday at t+1
            hist_data_with_label[i].append(
                datetime.datetime.strptime(hist[0]['date'].values[i + j + 1], "%Y-%m-%d").weekday())
            # past prices
            for h in hist:
                # changes at t+1 with close at t
                hist_data_with_label[i].append(
                    ((h['high'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                hist_data_with_label[i].append(
                    ((h['close'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                hist_data_with_label[i].append(
                    ((h['open'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                hist_data_with_label[i].append(
                    ((h['low'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                # the yesterday's close
                hist_data_with_label[i].append(h['close'].values[i + j])
                # volume at t+1
                hist_data_with_label[i].append(h['volume'].values[i + j + 1])

        # add label
        change = ((hist[0]['close'].values[i + data_length + 1] - hist[0]['close'].values[i + data_length])
                  / hist[0]['close'].values[i + data_length])
        label_size = _add_label(hist_data_with_label[i], change)

    print('[Last Trading Day]', end=' ')
    print(hist[0].iloc[-1, 0])

    # get current
    recent_data = []
    recent_returns = []
    for i in range(len(hist[0]) - data_length)[-recent_days:]:
        # label day: i + l + 1
        # last sample beginning: i = len(hist[0]) - l - 1
        # last label(when in last sample): len(hist[0]) (unknown)
        recent_data.append([])
        for j in range(data_length):
            # weekday at t+1
            recent_data[-1].append(
                datetime.datetime.strptime(hist[0]['date'].values[i + j + 1], "%Y-%m-%d").weekday())

            # past prices
            for h in hist:
                # changes at t+1 with close at t
                recent_data[-1].append(
                    ((h['high'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                recent_data[-1].append(
                    ((h['close'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                recent_data[-1].append(
                    ((h['open'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                recent_data[-1].append(
                    ((h['low'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                # the yesterday's close
                recent_data[-1].append(h['close'].values[i + j])
                # volume at t+1
                recent_data[-1].append(h['volume'].values[i + j + 1])

        if i == len(hist[0]) - data_length - 1:
            continue
        change = ((hist[0]['close'].values[i + data_length + 1] - hist[0]['close'].values[i + data_length])
                  / hist[0]['close'].values[i + data_length])
        # add return
        recent_returns.append(change)

    # adjust
    hist_data_with_label = np.array(hist_data_with_label)
    data_size = np.shape(hist_data_with_label)[1] - label_size

    train_data = hist_data_with_label[:, :data_size]
    train_label = hist_data_with_label[:, data_size:]

    recent_data = np.array(recent_data)
    recent_label = hist_data_with_label[-(recent_days - 1):, data_size:]

    # normalize
    _pre_process(train_data)
    _pre_process(recent_data)

    # recent_data: recent_days
    # recent_label: recent_days-1
    # recent_returns: recent_days -1
    return train_data, train_label, recent_data, recent_label, recent_returns, data_size, label_size


def dl_predict(target='300403',
               nb_of_correlations=10,
               days_for_statistic=500,
               data_length=15,
               recent_days=20,
               hist_file='hist0322.pkl',
               show_figure=True):
    # parameters:
    # n = how many correlated assets will be used, n asset with biggest corr, and n asset with smallest corr
    # length = how many history days used in prediction
    # info_size = how many factors have been included in each history day
    # warnings.filterwarnings("ignore")

    # get data
    train_data, train_label, recent_data, recent_label, recent_returns, data_size, label_size = \
        _get_data_for_predict(target, nb_of_correlations, days_for_statistic, data_length, recent_days, hist_file)

    # initiate the model
    model = Sequential()
    model.add(Dense(output_dim=128, input_dim=data_size, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=64, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=32, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=label_size, init='uniform'))
    model.add(Activation('softmax'))

    # Multilayer Perceptron (MLP) for multi-class softmax classification
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    # categorical_crossentropy: Calculates the cross-entropy value for multiclass classification problems.
    #   Note: Expects a binary class matrix instead of a vector of scalar classes.
    # RMSprop: This optimizer is usually a good choice for recurrent neural networks.

    # train the model
    model.fit(train_data, train_label, nb_epoch=64, batch_size=20, verbose=0)
    # batch_size: use full batch size

    # predict
    recent_predict = model.predict(recent_data)

    # get the errors
    type_1 = []  # incorrect rejection
    type_2 = []  # incorrect accept
    err = []
    # for each label
    for i in range(np.shape(recent_predict)[1]):
        x1 = np.mean([x for x in (recent_label[:, i] - recent_predict[:-1, i]) if x > 0])
        x2 = np.mean([x for x in (recent_label[:, i] - recent_predict[:-1, i]) if x < 0])
        type_1.append(x1)
        type_2.append(x2)
        err.append(x1 + abs(x2))
    type_1 = np.nan_to_num(type_1)
    type_2 = np.nan_to_num(type_2)
    err = np.nan_to_num(err)

    if show_figure:
        fig = plt.figure(figsize=(15, 6))
        mapping = {0: '(~, -5%)', 1: '[-5%, 0%)', 2: '[0%, 5%)', 3: '[5%, ~)'}
        for i in range(np.shape(recent_predict)[1]):
            ax = fig.add_subplot((np.shape(recent_predict)[1] + 1), 1, (i + 1))
            ax.grid(True)
            x = np.linspace(1, len(recent_predict), len(recent_predict))
            plt.bar(x[:-1], recent_label[:, i], alpha=0.5, color='r')
            plt.bar(x[:-1], recent_predict[:-1, i], alpha=0.5, color='b')
            plt.bar(x[-1], recent_predict[-1, i], alpha=1, color='k')
            plt.title(mapping[i])
        ax = fig.add_subplot((np.shape(recent_predict)[1] + 1), 1, (np.shape(recent_predict)[1] + 1))
        ax.grid(True)
        x = np.linspace(1, len(recent_predict), len(recent_predict))
        plt.bar(x, recent_returns + [0], alpha=0.5, color='k')
        plt.title('returns')
        plt.show()

    expected_movement = - 0.075 * recent_predict[-1, 0] \
                        - 0.025 * recent_predict[-1, 1] \
                        + 0.025 * recent_predict[-1, 2] \
                        + 0.075 * recent_predict[-1, 3]

    return expected_movement, np.mean(type_1), np.mean(type_2), np.mean(err)
