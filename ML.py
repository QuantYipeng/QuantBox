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
            data[i].append(hist['close'].values[i + j])

        # add label
        change = ((hist['close'].values[i + l + 1] - hist['close'].values[i + j]) / hist['close'].values[i + l])
        if change > 0:
            data[i].append(1)
        else:
            data[i].append(0)
        if change > 0.05:
            data[i].append(1)
        else:
            data[i].append(0)
        if change < -0.05:
            data[i].append(1)
        else:
            data[i].append(0)

    return data


def dl(code='300403', days=200, length=3, info_size=5, test_ratio=0.9):
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
    model.add(Dense(output_dim=32, input_dim=input_dimension, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=32, init='uniform'))
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
        ax = fig.add_subplot(1, np.shape(predict)[1], (i+1))
        ax.grid(True)
        x = np.linspace(1, len(predict), len(predict))
        plt.bar(x, true[:, i], alpha=0.5, color='r')
        plt.bar(x, predict[:, i], alpha=0.5, color='g')
    plt.show()

    return
