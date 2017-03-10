import datetime
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats.stats import pearsonr
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation


def download_data(file_name='data0309.pkl', days=365):
    # using get_k_hist to download

    # get stock names
    stock_info = ts.get_stock_basics()

    # set date
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime(
        '%Y-%m-%d')

    # calculate the expected returns
    code = []
    data = []
    count = 0
    for i in stock_info.index:
        count += 1
        try:
            # get data
            hist = ts.get_k_data(i, start=one_year_before, end=today)  # (from past to now)
            code.append(i)
            data.append(hist)
            print('[Downloading Stocks]Process:  %0.2f %%' % (100.0 * count / len(stock_info)))
        except:
            continue
    count = 0
    indices = ['000001', '399001', '399006']
    for i in indices:
        if i in code:
            continue
        count += 1
        try:
            # get data
            hist = ts.get_k_data(i, start=one_year_before, end=today)  # (from past to now)
            code.append(i)
            data.append(hist)
            print('[Downloading Indices]Process:  %0.2f %%' % (100.0 * count / len(indices)))
        except:
            continue

    # write into files
    content = dict(zip(code, data))

    fn = file_name
    with open(fn, 'w') as f:  # open file with write-mode
        pickle.dump(content, f)  # serialize and save object
    return


def get_data(target='300403', pool=['300403', '000001'], n=10, days=200, l=1, data_file=''):
    # parameters:
    # n = how many correlated assets will be used, n asset with biggest corr, and n asset with smallest corr
    # l = how many history days used in prediction
    # days = how many calendar days of history data should we download
    #   not calendar when using data_file, maybe short that calendar days

    # get [target, related assets]
    if target in pool:
        pool.remove(target)
    code = [target] + pool

    #
    def is_index(_code='000001'):
        indices = ['000001', '399001', '399006']
        for index in indices:
            if index == _code:
                return True
        return False

    # get data
    if len(data_file):
        fn = data_file
        with open(fn, 'r') as f:
            content = pickle.load(f)  # read file and build object
            hist = []
            for c in code:
                try:
                    hist.append(content[c][-days:])
                except:
                    print('cannot get data of ' + c)
    else:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        hist = []
        count = 0
        for c in code:
            try:
                count += 1
                print('Process:  %0.2f %%' % (100.0 * count / len(code)))
                if is_index(c):
                    hist.append(ts.get_k_data(c, index=True, start=one_year_before, end=today))
                else:
                    hist.append(ts.get_k_data(c, start=one_year_before, end=today))  # (past to now)
                    # get_k_data's index start from the first trading day in the start year
            except:
                print('cannot get data of ' + c)

    # drop the unmatched data point
    def match(hist_a, hist_b):
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
    for i in range(len(hist)):
        if i == 0:
            continue
        hist[0], hist[i], drop_b = match(hist[0], hist[i])
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
    corr_dict = dict(zip(range(len(hist)), corr))

    # sort by values: sorted(corr_dict.items(), key=lambda item: item[1])
    # corr_index is the N asset's index with the biggest correlation
    corr_index_top = np.array(sorted(corr_dict.items(), key=lambda item: item[1]))[-n:, 0]
    corr_index_bottom = np.array(sorted(corr_dict.items(), key=lambda item: item[1]))[:n, 0]

    corr_index = [corr_index_top, corr_index_bottom]
    corr_index = (np.array(corr_index)).flatten().tolist()

    # remove itself
    if 0 in corr_index:
        corr_index.remove(0)

    # select the correlated N assets
    print('\n[select the correlated N assets]')
    temp_hist = [hist[0]]
    for i in corr_index:
        temp_hist.append(hist[int(i)])
        print('- ' + str(hist[int(i)]['code'].values[0]) + ' corr: ' + str(corr_dict[int(i)]))
    hist = temp_hist

    data = []
    returns = []
    # for each sample
    for i in range(len(hist[0]) - (l + 1)):

        data.append([])
        # for each data
        for j in range(l):
            # weekday at t+1
            data[i].append(
                datetime.datetime.strptime(hist[0]['date'].values[i + j + 1].encode('utf-8'), "%Y-%m-%d").weekday())

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

        returns.append(change)

    return data, returns


def dl(target='300403', pool=['300403', '000001'], n=10, days=200, length=15, label_size=4, test_ratio=0.9,
       data_file=''):
    # parameters:
    # n = how many correlated assets will be used, n asset with biggest corr, and n asset with smallest corr
    # length = how many history days used in prediction
    # info_size = how many factors have been included in each history day

    # get data
    if len(data_file):
        data, returns = get_data(target, pool, n, days, length, data_file)
    else:
        data, returns = get_data(target, pool, n, days, length)
    data = np.array(data)
    i_data = int(len(data) * test_ratio)
    i_label = np.shape(data)[1] - label_size
    output_dimension = np.shape(data)[1] - i_label
    train_data = data[:i_data, :i_label]
    train_label = data[:i_data, i_label:]
    test_data = data[i_data:, :i_label]
    test_label = data[i_data:, i_label:]
    test_returns = returns[i_data:]

    # normalize using Z-score=(x-mu)/std
    def zscore(m):
        for c in range(np.shape(m)[1]):
            mean = np.mean(m[:, c])
            std = np.std(m[:, c])
            for r in range(np.shape(m)[0]):
                m[r, c] = (m[r, c] - mean) / std

    zscore(train_data)
    zscore(test_data)

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
    model.fit(train_data, train_label, nb_epoch=64, batch_size=i_data, verbose=0)
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
        err.append(x1+abs(x2))

    print type_1
    print type_2
    print err

    fig = plt.figure(figsize=(15, 4))
    for i in range(np.shape(predict)[1]):
        ax = fig.add_subplot(1, np.shape(predict)[1], (i + 1))
        ax.grid(True)
        x = np.linspace(1, len(predict), len(predict))
        plt.bar(x, true[:, i], alpha=0.5, color='r')
        plt.bar(x, predict[:, i], alpha=0.5, color='g')
        plt.plot(x, [x*10 for x in test_returns], alpha=0.5, color='k')
    plt.show()

    return err, type_1, type_2
