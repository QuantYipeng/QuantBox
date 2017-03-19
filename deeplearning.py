import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import tushare as ts
from scipy.stats.stats import pearsonr
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tqdm import tqdm


def _pre_process(m):
    # function: pre-process m using Z-score=(x-mu)/std for every column in m
    # return: None
    for c in range(np.shape(m)[1]):
        mean = np.mean(m[:, c])
        std = np.std(m[:, c])
        for r in range(np.shape(m)[0]):
            m[r, c] = (m[r, c] - mean) / std
    return


def _is_in_time_interval(start, target, end):
    def _t2s(_t):
        _h, _m, _s = _t.strip().split(':')
        return int(_h) * 3600 + int(_m) * 60 + int(_s)

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


def _get_data_for_back_test(target='300403', correlations=10, days=200, l=1, data_file='hist0316.pkl'):
    # parameters:
    # n = how many correlated assets will be used, n asset with biggest corr, and n asset with smallest corr
    # l = how many history days used in prediction
    # days = how many calendar days of history data should we download
    #   not calendar when using data_file, maybe short that calendar days

    # get data
    fn = data_file
    with open(fn, 'rb') as f:
        content = pickle.load(f)  # read file and build object

    # get [target, related assets]
    hist = []
    pool = list(content.keys())
    if target in pool:
        pool.remove(target)
    code = [target] + pool
    for c in code:
        try:
            hist.append(content[c][-days:])
        except:
            print('cannot get data of ' + c)

    # drop the unmatched data point
    def _match_trim_b(hist_a, hist_b):
        # return:
        # hist_a, hist_b, is_too_many_a_dropped (If True -> drop hist_b)

        to_be_drop_a = []
        if len(hist_a.index) - len(hist_b.index) != 0:
            for _i in range(len(hist_a.index)):
                exist = 0
                for _j in range(len(hist_b.index)):
                    if hist_a.index[_i] != hist_b.index[_j]:
                        exist = 0
                    else:
                        exist = 1
                        break
                if exist == 0:
                    to_be_drop_a.append(hist_a.index[_i])
        nb_drop_a = len(to_be_drop_a)
        if nb_drop_a > 0:
            return hist_a, hist_b, True
        hist_a = hist_a.drop(to_be_drop_a, axis=0)

        to_be_drop_b = []
        if len(hist_b.index) - len(hist_a.index) != 0:
            for _i in range(len(hist_b.index)):
                exist = 0
                for _j in range(len(hist_a.index)):
                    if hist_b.index[_i] != hist_a.index[_j]:
                        exist = 0
                    else:
                        exist = 1
                        break
                if exist == 0:
                    to_be_drop_b.append(hist_b.index[_i])
        hist_b = hist_b.drop(to_be_drop_b, axis=0)

        return hist_a, hist_b, False

    # match & drop
    drop_list = []
    for i in tqdm(range(len(hist)), desc=('[Matching & Dropping for ' + target + ']')):
        if i == 0:
            continue
        hist[0], hist[i], drop_b = _match_trim_b(hist[0], hist[i])
        # if we too drop too many hist_a elements, then we will drop hist_b
        if drop_b:
            drop_list.append(i)

    #
    drop_list.reverse()
    for d in drop_list:
        hist.pop(d)

    # get corr
    corr = []
    for i in range(len(hist)):
        corr.append(pearsonr(hist[0]['close'].values, hist[i]['close'].values)[0])
    # ignore itself
    corr_dict = dict(zip(range(len(hist))[1:], corr[1:]))

    # sort by values: sorted(corr_dict.items(), key=lambda item: item[1])
    # corr_index is the N(correlations) asset's index with the biggest correlation (exclude itself)
    # and the N/2 asset's index with the lowest correlation
    corr_dict_sorted = np.array(sorted(corr_dict.items(), key=lambda item: item[1]))
    corr_index_bottom = corr_dict_sorted[-correlations:, 0].tolist()
    corr_index_top = corr_dict_sorted[:correlations, 0].tolist()
    corr_index = [corr_index_top, corr_index_bottom]
    corr_index = [y for x in corr_index for y in x]

    # select the correlated N assets
    temp_hist = [hist[0]]
    for i in corr_index:
        temp_hist.append(hist[int(i)])
    hist = temp_hist

    data = []
    returns = []
    # for each sample
    for i in tqdm(range(len(hist[0]) - (l + 1)), desc='Preparing Sample Data'):

        data.append([])
        # for each data
        for j in range(l):
            # weekday at t+1
            data[i].append(
                datetime.datetime.strptime(hist[0]['date'].values[i + j + 1], "%Y-%m-%d").weekday())

            # past prices
            for h in hist:
                # changes at t+1 with close at t
                data[i].append(
                    ((h['high'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                data[i].append(
                    ((h['close'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                data[i].append(
                    ((h['open'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                data[i].append(
                    ((h['low'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                # the yesterday's close
                data[i].append(h['close'].values[i + j])
                # volume at t+1
                data[i].append(h['volume'].values[i + j + 1])

            # big deals
            for h in hist:
                big_deal_result = _get_big_deal_statistic(code=h['code'].values[0],
                                                          date=h['date'].values[i + j + 1],
                                                          vol=0)
                for b in big_deal_result:
                    data[i].append(b)

        # add label
        change = ((hist[0]['close'].values[i + l + 1] - hist[0]['close'].values[i + l])
                  / hist[0]['close'].values[i + l])

        if change > 0:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < 0:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.005:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.005:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.01:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.01:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.03:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.03:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.09:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.09:
            data[i].append(1)
        else:
            data[i].append(0)

        returns.append(change)

    return data, returns


def _get_data_for_predict(target='300403', correlations=10, days=200, l=1, hist_file='hist0316.pkl'):
    # parameters:
    # n = how many correlated assets will be used, n asset with biggest corr, and n asset with smallest corr
    # l = how many history days used in prediction
    # days = how many calendar days of history data should we download
    #   not calendar when using data_file, maybe short that calendar days

    # get data
    with open(hist_file, 'rb') as f:
        content = pickle.load(f)  # read file and build object

    # get [target, related assets]
    hist = []
    pool = list(content.keys())
    if target in pool:
        pool.remove(target)
    code = [target] + pool
    for c in code:
        try:
            hist.append(content[c][-days:])
        except:
            print('cannot get data of ' + c)

    # drop the unmatched data point
    def _match_trim_b(hist_a, hist_b):
        # return:
        # hist_a, hist_b, is_too_many_a_dropped (If True -> drop hist_b)

        to_be_drop_a = []
        if len(hist_a.index) - len(hist_b.index) != 0:
            for _i in range(len(hist_a.index)):
                exist = 0
                for _j in range(len(hist_b.index)):
                    if hist_a.index[_i] != hist_b.index[_j]:
                        exist = 0
                    else:
                        exist = 1
                        break
                if exist == 0:
                    to_be_drop_a.append(hist_a.index[_i])
        nb_drop_a = len(to_be_drop_a)
        if nb_drop_a > 0:
            return hist_a, hist_b, True
        hist_a = hist_a.drop(to_be_drop_a, axis=0)

        to_be_drop_b = []
        if len(hist_b.index) - len(hist_a.index) != 0:
            for _i in range(len(hist_b.index)):
                exist = 0
                for _j in range(len(hist_a.index)):
                    if hist_b.index[_i] != hist_a.index[_j]:
                        exist = 0
                    else:
                        exist = 1
                        break
                if exist == 0:
                    to_be_drop_b.append(hist_b.index[_i])
        hist_b = hist_b.drop(to_be_drop_b, axis=0)

        return hist_a, hist_b, False

    # match & drop
    drop_list = []
    for i in tqdm(range(len(hist)), desc=('[Matching & Dropping for ' + target + ']')):
        if i == 0:
            continue
        hist[0], hist[i], drop_b = _match_trim_b(hist[0], hist[i])
        # if we too drop too many hist_a elements, then we will drop hist_b
        if drop_b:
            drop_list.append(i)
    drop_list.reverse()
    for d in drop_list:
        hist.pop(d)

    # get corr
    corr = []
    for i in range(len(hist)):
        corr.append(pearsonr(hist[0]['close'].values, hist[i]['close'].values)[0])
    # ignore itself
    corr_dict = dict(zip(range(len(hist))[1:], corr[1:]))

    # sort by values: sorted(corr_dict.items(), key=lambda item: item[1])
    # corr_index is the N(correlations) asset's index with the biggest correlation (exclude itself)
    # and the N/2 asset's index with the lowest correlation
    corr_dict_sorted = np.array(sorted(corr_dict.items(), key=lambda item: item[1]))
    corr_index_bottom = corr_dict_sorted[-correlations:, 0].tolist()
    corr_index_top = corr_dict_sorted[:correlations, 0].tolist()
    corr_index = [corr_index_top, corr_index_bottom]
    corr_index = [y for x in corr_index for y in x]

    # select the correlated N assets
    temp_hist = [hist[0]]
    for i in corr_index:
        temp_hist.append(hist[int(i)])
    hist = temp_hist

    data = []
    # for each sample
    for i in range(len(hist[0]) - (l + 1)):

        data.append([])
        # for each data
        for j in range(l):
            # weekday at t+1
            data[i].append(
                datetime.datetime.strptime(hist[0]['date'].values[i + j + 1], "%Y-%m-%d").weekday())
            # past prices
            for h in hist:
                # changes at t+1 with close at t
                data[i].append(
                    ((h['high'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                data[i].append(
                    ((h['close'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                data[i].append(
                    ((h['open'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                data[i].append(
                    ((h['low'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                # the yesterday's close
                data[i].append(h['close'].values[i + j])
                # volume at t+1
                data[i].append(h['volume'].values[i + j + 1])

        # add label
        change = ((hist[0]['close'].values[i + l + 1] - hist[0]['close'].values[i + l])
                  / hist[0]['close'].values[i + l])

        if change > 0:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < 0:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.005:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.005:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.01:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.01:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.03:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.03:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.09:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.09:
            data[i].append(1)
        else:
            data[i].append(0)

    # get current
    current = []
    for i in range(len(hist[0]) - l)[-10:]:
        # last day: i = len(hist[0]) - (l + 1)
        current.append([])
        for j in range(l):
            # weekday at t+1
            current[-1].append(
                datetime.datetime.strptime(hist[0]['date'].values[i + j + 1], "%Y-%m-%d").weekday())

            # past prices
            for h in hist:
                # changes at t+1 with close at t
                current[-1].append(
                    ((h['high'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                current[-1].append(
                    ((h['close'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                current[-1].append(
                    ((h['open'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                current[-1].append(
                    ((h['low'].values[i + j + 1] - h['close'].values[i + j]) / h['close'].values[i + j]))
                # the yesterday's close
                current[-1].append(h['close'].values[i + j])
                # volume at t+1
                current[-1].append(h['volume'].values[i + j + 1])
    return data, current


def dl_back_test(target='300403',
                 correlations=10,
                 days=200,
                 length=15,
                 label_size=10,
                 test_ratio=0.9,
                 hist_file='hist0316.pkl',
                 show_figure=False):
    """
    do back test
    parameters:
    n = how many correlated assets will be used, n asset with biggest corr, and n asset with smallest corr
    length = how many history days used in prediction
    info_size = how many factors have been included in each history day
    """
    warnings.filterwarnings("ignore")

    # get data
    data, returns = _get_data_for_back_test(target, correlations, days, length, hist_file)

    data = np.array(data)
    i_data = int(len(data) * test_ratio)
    i_label = np.shape(data)[1] - label_size
    output_dimension = np.shape(data)[1] - i_label
    train_data = data[:i_data, :i_label]
    train_label = data[:i_data, i_label:]
    test_data = data[i_data:, :i_label]
    test_label = data[i_data:, i_label:]
    test_returns = returns[i_data:]

    # normalize
    _pre_process(train_data)
    _pre_process(test_data)

    # initiate the model
    model = Sequential()
    model.add(Dense(output_dim=128, input_dim=i_label, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=64, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=32, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=output_dimension, init='uniform'))
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
    predict = model.predict(test_data)
    true = test_label

    type_1 = []  # incorrect rejection
    type_2 = []  # incorrect accept
    err = []
    for i in range(np.shape(predict)[1]):
        x1 = np.mean([x for x in (true[:, i] - predict[:, i]) if x > 0])
        x2 = np.mean([x for x in (true[:, i] - predict[:, i]) if x < 0])
        type_1.append(x1)
        type_2.append(x2)
        err.append(x1 + abs(x2))

    print('[Error Type 1] ', end='')
    for i in type_1:
        print('%0.2f  ' % i, end='')
    print(' ')

    print('[Error Type 2] ', end='')
    for i in type_2:
        print('%0.2f  ' % i, end='')
    print(' ')

    print('[Error Sum]    ', end='')
    for i in err:
        print('%0.2f  ' % i, end='')
    print(' ')

    if show_figure:
        fig = plt.figure(figsize=(15, 4))
        mapping = {0: '> 0 %', 1: '< 0 %', 2: '> 0.5 %', 3: '< -0.5 %',
                   4: '> 1 %', 5: '< -1 %', 6: '> 3 %', 7: '< -3 %',
                   8: '> 9 %', 9: '< -9 %'}
        for i in range(np.shape(predict)[1]):
            ax = fig.add_subplot(1, np.shape(predict)[1], (i + 1))
            ax.grid(True)
            x = np.linspace(1, len(predict), len(predict))
            plt.bar(x, true[:, i], alpha=0.5, color='r')
            plt.bar(x, predict[:, i], alpha=0.5, color='g')
            plt.plot(x, [x * 10 for x in test_returns], alpha=0.5, color='k')
            plt.title(mapping[i])
        plt.show()

    return {'type_1': type_1, 'type_2': type_2, 'err': err}


def dl_predict(target='300403',
               correlations=10,
               days=500,
               length=15,
               label_size=10,
               hist_file='hist0317.pkl',
               show_figure=False):
    # parameters:
    # n = how many correlated assets will be used, n asset with biggest corr, and n asset with smallest corr
    # length = how many history days used in prediction
    # info_size = how many factors have been included in each history day
    warnings.filterwarnings("ignore")

    # get data
    data, current = _get_data_for_predict(target, correlations, days, length, hist_file)
    data = np.array(data)
    current = np.array(current)

    i_label = np.shape(data)[1] - label_size
    output_dimension = np.shape(data)[1] - i_label
    train_data = data[:, :i_label]
    train_label = data[:, i_label:]

    # get true label (the last true label is the predicted label showing as 0)
    a = data[-9:, i_label:].tolist()
    b = np.zeros(label_size).tolist()  # predicted label
    a.append(b)
    true = np.array(a)

    # normalize
    _pre_process(train_data)
    _pre_process(current)

    # initiate the model
    model = Sequential()
    model.add(Dense(output_dim=128, input_dim=i_label, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=64, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=32, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=output_dimension, init='uniform'))
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
    predict = model.predict(current)

    # get the errors
    type_1 = []  # incorrect rejection
    type_2 = []  # incorrect accept
    err = []
    for i in range(np.shape(predict)[1]):
        x1 = np.mean([x for x in (true[:-1, i] - predict[:-1, i]) if x > 0])
        x2 = np.mean([x for x in (true[:-1, i] - predict[:-1, i]) if x < 0])
        type_1.append(x1)
        type_2.append(x2)
        err.append(x1 + abs(x2))
    type_1 = np.nan_to_num(type_1)
    type_2 = np.nan_to_num(type_2)
    err = np.nan_to_num(err)

    if show_figure:
        fig = plt.figure(figsize=(15, 4))
        mapping = {0: '> 0 %', 1: '< 0 %', 2: '> 0.5 %', 3: '< -0.5 %',
                   4: '> 1 %', 5: '< -1 %', 6: '> 3 %', 7: '< -3 %',
                   8: '> 9 %', 9: '< -9 %'}
        for i in range(np.shape(predict)[1]):
            ax = fig.add_subplot(1, np.shape(predict)[1], (i + 1))
            ax.grid(True)
            x = np.linspace(1, len(predict), len(predict))
            plt.bar(x, true[:, i], alpha=0.5, color='r')
            plt.bar(x[:-1], predict[:-1, i], alpha=0.5, color='b')
            plt.bar(x[-1], predict[-1, i], alpha=1, color='k')
            plt.title(mapping[i])
        plt.show()

    expected_movement = + 0.0025 * predict[-1, 0] \
                        - 0.0025 * predict[-1, 1] \
                        + 0.0075 * predict[-1, 2] \
                        - 0.0075 * predict[-1, 3] \
                        + 0.015 * predict[-1, 4] \
                        - 0.015 * predict[-1, 5] \
                        + 0.06 * predict[-1, 6] \
                        - 0.06 * predict[-1, 7] \
                        + 0.095 * predict[-1, 8] \
                        - 0.095 * predict[-1, 9]
    return expected_movement, np.mean(type_1), np.mean(type_2), np.mean(err)


def get_best_parameters(target='300403',
                        correlations=(1, 2, 3, 5, 10),
                        days=(60, 90, 120, 200),
                        length=(2, 3, 4, 5, 7, 10, 15), hist_file='hist0316.pkl'):
    """
    get the best parameters for specific stock
    return: parameters = {'target': target, 'correlations': 0, 'days': 0, 'length': 0}
    """
    warnings.filterwarnings("ignore")

    record = []
    for i in correlations:
        for j in days:
            for k in length:
                print('[correlations: ' + str(i) + ' days: ' + str(j) + ' length: ' + str(k) + ']')
                result = dl_back_test(target=target,
                                      correlations=i,
                                      days=j,
                                      length=k,
                                      label_size=10,
                                      test_ratio=0.7,
                                      hist_file=hist_file,
                                      show_figure=False)
                record.append({'correlations': i, 'days': j, 'length': k, 'result': result})
    minimum = 2
    parameters = {'target': target, 'correlations': 0, 'days': 0, 'length': 0}
    for i in tqdm(range(len(record)), desc='[Finding the Best Parameters]'):
        err = record[i]['result']['err'][0]
        if err < minimum:
            minimum = err
            parameters['correlations'] = record[i]['correlations']
            parameters['days'] = record[i]['days']
            parameters['length'] = record[i]['length']
    return parameters
