from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import datetime
import tushare as ts
import numpy as np


def get_data(code='300403', days=200, l=1):
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # reverse data order
    hist = hist.sort_index(0)

    data = []
    for i in range(len(hist) - (l + 1)):
        data.append([])
        # add data
        for j in range(l):
            data[i].append(
                ((hist['high'].values[i + j + 1] - hist['close'].values[i + j]) / hist['close'].values[i + j]))
            data[i].append(
                ((hist['close'].values[i + j + 1] - hist['close'].values[i + j]) / hist['close'].values[i + j]))
            data[i].append(
                ((hist['open'].values[i + j + 1] - hist['close'].values[i + j]) / hist['close'].values[i + j]))
            data[i].append(
                ((hist['low'].values[i + j + 1] - hist['close'].values[i + j]) / hist['close'].values[i + j]))
            # data[i].append(hist['close'].values[i + j])
        # add label
        change = ((hist['close'].values[i + l + 1] - hist['close'].values[i + j]) / hist['close'].values[i + l])
        if change > 0.05:
            data[i].append(0)
        else:
            data[i].append(1)
        if change > 0 and not change > 0.05:
            data[i].append(0)
        else:
            data[i].append(1)
        if change < 0:
            data[i].append(0)
        else:
            data[i].append(1)
    return data


def dl(code='300403', days=200, l=3):
    model = Sequential()
    model.add(Dense(input_dim=l * 4, output_dim=32, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=32, output_dim=16, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=16, output_dim=3, init='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    data = get_data(code, days, l)
    data = np.array(data)
    i_data = int(len(data) * 0.7)
    i_label = int(l * 4)
    train_data = data[:i_data, :i_label]
    train_label = data[:i_data, i_label:]
    test_data = data[i_data:, :i_label]
    test_label = data[i_data:, i_label:]

    model.fit(train_data, train_label, nb_epoch=80, batch_size=80, verbose=0)
    # print model.evaluate(test_data, test_label)
    print 'Predicted Label:\n'
    print model.predict_classes(test_data)
    print 'True Label:\n'
    print test_label

    return
