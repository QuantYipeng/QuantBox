from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn import preprocessing
import datetime
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt


def get_data(code='300403', days=200, l=1):
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)
    hist_index = ts.get_h_data('000001', start=one_year_before, end=today)  # reverse order (from now to past)

    # reverse data order
    hist = hist.sort_index(0)
    hist_index = hist_index.sort_index(0)

    # drop the hist_index where there hist is no available
    to_be_drop = []
    if len(hist_index.index)-len(hist.index) != 0:
        for i in range(len(hist_index.index)):
            exist = 0
            for j in range(len(hist.index)):
                if hist_index.index[i] != hist.index[j]:
                    exist = 0
                else:
                    exist = 1
                    break
            if exist == 0:
                to_be_drop.append(hist_index.index[i])
    hist_index = hist_index.drop(to_be_drop, axis=0)

    data = []
    for i in range(len(hist) - (l + 1)):
        data.append([])
        # add data
        for j in range(l):
            # weekday at t+1
            data[i].append(hist['close'].index[i + j + 1].weekday())

            # target asset
            # changes at t+1 with close at t
            data[i].append(
                ((hist['high'].values[i + j + 1] - hist['close'].values[i + j]) / hist['close'].values[i + j]))
            data[i].append(
                ((hist['close'].values[i + j + 1] - hist['close'].values[i + j]) / hist['close'].values[i + j]))
            data[i].append(
                ((hist['open'].values[i + j + 1] - hist['close'].values[i + j]) / hist['close'].values[i + j]))
            data[i].append(
                ((hist['low'].values[i + j + 1] - hist['close'].values[i + j]) / hist['close'].values[i + j]))
            data[i].append(hist['close'].values[i + j])
            # volume at t+1
            data[i].append(hist['volume'].values[i + j + 1])

            # shanghai index
            # shanghai index changes at t+1 with close at t
            data[i].append(
                ((hist_index['high'].values[i + j + 1] - hist_index['close'].values[i + j]) / hist_index['close'].values[i + j]))
            data[i].append(
                ((hist_index['close'].values[i + j + 1] - hist_index['close'].values[i + j]) / hist_index['close'].values[i + j]))
            data[i].append(
                ((hist_index['open'].values[i + j + 1] - hist_index['close'].values[i + j]) / hist_index['close'].values[i + j]))
            data[i].append(
                ((hist_index['low'].values[i + j + 1] - hist_index['close'].values[i + j]) / hist_index['close'].values[i + j]))
            data[i].append(hist_index['close'].values[i + j])
            # volume at t+1
            data[i].append(hist_index['volume'].values[i + j + 1])

        # add label
        change = ((hist['close'].values[i + l + 1] - hist['close'].values[i + j]) / hist['close'].values[i + l])
        if change > 0.09:
            data[i].append(1)
        else:
            data[i].append(0)

        if change > 0 and not change > 0.09:
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


def dl(code='300403', days=200, length=3, info_size=6, test_ratio=0.9):
    # parameters:
    # length = how many history days used in prediction
    # info_size = how many factors have been included in each history day

    # get data
    data = get_data(code, days, length)
    data = np.array(data)
    i_data = int(len(data) * test_ratio)
    i_label = int(length * info_size)
    input_dimension = length * info_size
    output_dimension = np.shape(data)[1] - i_label
    train_data = data[:i_data, :i_label]
    train_label = data[:i_data, i_label:]
    test_data = data[i_data:, :i_label]
    test_label = data[i_data:, i_label:]
    # normalize using Z-score=(x-mu)/std
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)

    # initiate the model
    model = Sequential()
    model.add(Dense(output_dim=128, input_dim=input_dimension, init='uniform'))
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
    print predict

    fig = plt.figure(figsize=(15, 4))
    for i in range(np.shape(predict)[1]):
        ax = fig.add_subplot(1, np.shape(predict)[1], (i + 1))
        ax.grid(True)
        x = np.linspace(1, len(predict), len(predict))
        plt.bar(x, true[:, i], alpha=0.5, color='r')
        plt.bar(x, predict[:, i], alpha=0.5, color='g')
    plt.show()

    return
