from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn import preprocessing
import datetime
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
import pickle


def get_data(code=['300403', '000001'], days=200, l=1, data_file=''):
    # parameters:
    # l = how many history days used in prediction

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
            hist = content.values()
    else:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        hist = []
        for c in code:
            if is_index(c):
                hist.append(ts.get_h_data(c, index=True, start=one_year_before, end=today))  # reverse order
            else:
                hist.append(ts.get_h_data(c, start=one_year_before, end=today))  # reverse order (from now to past)

    # reverse data order to (from past to now)
    for i in range(len(hist)):
        hist[i] = hist[i].sort_index(0)

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
        if nb_drop_a > 10:
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
    print('\ndrop: '+str(drop_list))
    for d in drop_list:
        hist.pop(d)

    data = []
    for i in range(len(hist[0]) - (l + 1)):
        data.append([])
        # add data
        for j in range(l):
            # weekday at t+1
            data[i].append(hist[0]['close'].index[i + j + 1].weekday())

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
                data[i].append(h['close'].values[i + j])
                # volume at t+1
                data[i].append(h['volume'].values[i + j + 1])

        # add label
        change = ((hist[0]['close'].values[i + l + 1] - hist[0]['close'].values[i + j]) / hist[0]['close'].values[i + l])
        if change > 0.09:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0.005 and not change > 0.09:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < 0 and not change < -0.09:
            data[i].append(1)
        else:
            data[i].append(0)

        if change < -0.09:
            data[i].append(1)
        else:
            data[i].append(0)

    return data


def dl(code=['300403', '000001'], days=200, length=15, label_size=4, test_ratio=0.9, data_file=''):
    # parameters:
    # length = how many history days used in prediction
    # info_size = how many factors have been included in each history day

    # get data
    if len(data_file):
        data = get_data(code, days, length, data_file)
    else:
        data = get_data(code, days, length)
    data = np.array(data)
    i_data = int(len(data) * test_ratio)
    i_label = np.shape(data)[1] - label_size
    output_dimension = np.shape(data)[1] - i_label
    train_data = data[:i_data, :i_label]
    train_label = data[:i_data, i_label:]
    test_data = data[i_data:, :i_label]
    test_label = data[i_data:, i_label:]

    # normalize using Z-score=(x-mu)/std
    def zscore(m):
        for c in range(np.shape(m)[1]):
            mean = np.mean(m[:, c])
            std = np.std(m[:, c])
            for r in range(np.shape(m)[0]):
                m[r, c] = (m[r, c]-mean)/std
    zscore(train_data)
    zscore(test_data)
    print data[:10, :]

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

    fig = plt.figure(figsize=(15, 4))
    for i in range(np.shape(predict)[1]):
        ax = fig.add_subplot(1, np.shape(predict)[1], (i + 1))
        ax.grid(True)
        x = np.linspace(1, len(predict), len(predict))
        plt.bar(x, true[:, i], alpha=0.5, color='r')
        plt.bar(x, predict[:, i], alpha=0.5, color='g')
    plt.show()

    return
