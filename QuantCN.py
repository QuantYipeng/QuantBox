import datetime
import matplotlib.dates as mpd
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import CQF
import pickle


def is_booming_stock(code='300403', days=365):
    # is this stock continuing raised 5 times?
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # get returns
    ds_by_s = (hist['close'].shift(1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
    ds_by_s = ds_by_s.sort_index(0)
    ds_by_s = np.nan_to_num(ds_by_s)

    is_booming = 0
    for i in range(len(ds_by_s.values)):
        if ds_by_s.values[i] > 0.09:
            is_booming += 1
        else:
            is_booming = 0

        if is_booming == 5:
            print('stock:' + str(code) + ' is a booming stock')
            return 1
    return 0


def get_er_of_mc_gbm(hist, days_for_predict=5, simulation=5000):
    # get returns
    ds_by_s = (hist['close'].shift(1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)

    # get parameters
    s0 = hist.close.values[0]
    dt = 1.0 / len(ds_by_s.values[1:])
    mu = np.mean(ds_by_s.values[1:]) / dt
    sigma = np.sqrt(np.var(ds_by_s.values[1:])) / np.sqrt(dt)

    # calculate the expected return
    expected_price = CQF.get_ep_of_mc_gbm(mu=mu, sigma=sigma, dt=dt, s0=s0, days=days_for_predict,
                                          simulation=simulation)
    expected_return = (expected_price - s0) / s0

    # calculate the p value whether the history returns are normally distributed (p<0.05 means significant)
    p_value = CQF.get_p_value_of_normal_test(np.nan_to_num(ds_by_s))

    return expected_return, p_value


def get_p_value_of_normal_test_history_returns(code='300403', days=365):
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # get returns
    ds_by_s = (hist['close'].shift(1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
    ds_by_s = ds_by_s[1:]  # drop NAN

    # calculate the p value whether the history returns are normally distributed (p<0.05 means significant)
    result = CQF.get_p_value_of_normal_test(ds_by_s)

    return result


def write_all_history_data(file_name='data0220.pkl', days=365):
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
            hist = ts.get_h_data(i, start=one_year_before, end=today)  # reverse order (from now to past)
            code.append(i)
            data.append(hist)
            print('Process:  %0.2f %%' % (100.0 * count / len(stock_info)))
        except:
            continue

    # write into files
    content = [code, data]
    fn = file_name
    with open(fn, 'w') as f:  # open file with write-mode
        pickle.dump(content, f)  # serialize and save object
    return


def load_statistic(file_name='data0220.pkl', days_for_statistic=90, days_for_predict=5, simulation=5000, bottom=0.055, top=0.06):
    fn = file_name
    with open(fn, 'r') as f:
        content = pickle.load(f)  # read file and build object

    def get_all_er_of_mc_gbm(_content, _days_for_statistic, _days_for_predict=5, _simulation=5000):
        # get a list of all stocks' [code, expected return, p-value]
        _code = _content[0]
        _data = _content[1]

        # calculate the expected returns
        _c = []
        _r = []
        _p = []
        _count = 0
        for _i in range(len(_data)):
            _count += 1
            try:
                # get data
                _er, _p_value = get_er_of_mc_gbm(_data[_i][:_days_for_statistic], _days_for_predict, _simulation)
                _c.append(_code[_i])
                _r.append(_er)
                _p.append(_p_value)
                print('[ Process:  %0.2f %% ] Stock Code: %s, Expected Return: %0.4f %%, P-Value: %0.4f %%' % (
                    (100.0 * _count / len(_data)), _code[_i], (_er * 100), (_p_value * 100)))
            except:
                continue

        _result = [_c, _r, _p]

        return _result

    result = get_all_er_of_mc_gbm(content, days_for_statistic, days_for_predict, simulation)

    # load the stock codes
    c = result[0]

    # load the expected returns
    r_str = result[1]
    r = [float(x) for x in r_str]
    r = np.nan_to_num(r)  # use 0 to substitute nan

    # load the p values
    p_str = result[2]
    p = [float(x) for x in p_str]

    # get the stock codes of specific returns, the first dimension of index (index[0]) is the real indices
    index = np.where((r > bottom) & (r < top))

    strings = []
    # print stock codes
    for i in index[0]:
        if is_booming_stock(c[i]) == 1:
            continue
        print('Stock:' + str(c[i]) + ' P-value:' + str(p[i]))
        strings.append('Stock:' + str(c[i]) + ' P-value:' + str(p[i]) + '\n')

    string = "".join(strings)

    '''
    f = open("data/temp.txt", 'w')
    print >> f, string
    '''
    print('\n\n' + string)

    return string


def load_all_statistic(file_name='data0220.pkl', days_for_statistic=90, days_for_predict=5, simulation=5000, bottom=0.05, gap=0.005, top=0.07):
    result = []
    for i in range(int((top-bottom)/gap)):
        result.append(('[ Expected Returns from ' + str((bottom+gap*i)) + ' to ' + str((bottom+gap*(i+1))) + ' ]'))
        result.append(load_statistic(file_name, days_for_statistic, days_for_predict, simulation, (bottom+gap*i), (bottom+gap*(i+1))))

    for i in range(len(result)):
        print result[i]

    return result


def plot_gbm_simulation(code='300403', days_for_predict=100, days_for_statistic=365):
    """
    plot a simulated geometric brownian motion of future close price of the given stock
    :param code: the code of stock
    :param days_for_predict: days to simulate
    :param days_for_statistic: days to statistic mu & sigma
    :return: the predicted price movement diagram
    """

    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days_for_statistic)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # get returns
    ds_by_s = (hist['close'].shift(1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)

    # get parameters
    s0 = hist.close.values[0]  # last close price
    max_return = np.max(ds_by_s.values[1:])
    min_return = np.min(ds_by_s.values[1:])
    dt = 1.0 / len(ds_by_s.values[1:])
    mu = np.mean(ds_by_s.values[1:]) / dt
    sigma = np.sqrt(np.var(ds_by_s.values[1:])) / np.sqrt(dt)

    # print parameters
    print('price (current) = %0.2f' % s0)
    print('return (max) = %0.2f %%' % (max_return * 100))
    print('return (min) = %0.2f %%' % (min_return * 100))
    print('return (mu) = %0.2f %%' % (mu * 100))
    print('return (sigma) = %0.2f %%' % (sigma * 100))
    print('history (len) = %d' % len(ds_by_s))

    # plot simulated geometric brownian motion
    CQF.plot_gbm(mu=mu, sigma=sigma, dt=dt, s0=s0, days=days_for_predict)

    return


def plot_predicts_and_facts(code='300403', days_for_test=365, days_for_predict=5, days_for_statistic=90,
                            simulation=5000):
    # days_for_predict is trading days
    # days_for_statistic is trading days
    # days_for_test is calendar days

    def get_predict(s, days):
        # get returns
        ds_by_s = (s['close'].shift(1) - s['close']) / s[
            'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
        ds_by_s = np.nan_to_num(ds_by_s)  # change nan to zero

        # get parameters
        s_0 = s.close.values[0]
        dt = 1.0 / len(ds_by_s.values)
        mu = np.mean(ds_by_s.values) / dt
        sigma = np.sqrt(np.var(ds_by_s.values)) / np.sqrt(dt)

        # calculate the expected return
        expected_price = CQF.get_ep_of_mc_gbm(mu=mu, sigma=sigma, dt=dt, s0=s_0, days=days,
                                              simulation=simulation)
        return expected_price

    # get history data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    day_before = (datetime.datetime.now() - datetime.timedelta(days=days_for_test)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=day_before, end=today)  # reverse order (from now to past)

    # length test
    if (days_for_test - days_for_statistic - days_for_predict) < 0:
        print('--- (days_for_test - days_for_statistic - days_for_predict) must bigger or equal to 0')
        return
    if len(hist) < (days_for_statistic + days_for_predict):
        print('--- not enough history data. hist:' + str(len(hist))
              + ' at least:' + str(days_for_statistic + days_for_predict))
        return
    else:
        print('--- number of test: ' + str(len(hist) - days_for_statistic - days_for_predict + 1) + '\n')

    # get predicts and facts
    predict_price = []
    fact_price = []
    predict_r = []
    fact_r = []
    p_values_statistic = []
    p_values_predict = []
    for i in range(len(hist) - days_for_statistic - days_for_predict + 1):  # i is test count [1,..)
        print('percent:%0.4f %%' % (100.0 * (1.0 + i) / (len(hist) - days_for_statistic - days_for_predict + 1)))

        sub_hist_statistic = hist[(days_for_predict + i):(days_for_statistic + days_for_predict + i)]  # hist[)
        sub_hist_predict = hist[i:(days_for_predict + i)]  # hist[)
        s0 = hist.close.values[days_for_predict + i]
        p = get_predict(sub_hist_statistic, days_for_predict)
        f = hist['close'].values[i]
        predict_price.insert(0, p)  # insert at beginning
        fact_price.insert(0, f)
        predict_r.insert(0, (1.0 * (p - s0) / s0))
        fact_r.insert(0, (1.0 * (f - s0) / s0))
        p_values_statistic.insert(0, CQF.get_p_value_of_normal_test(sub_hist_statistic.close.values))
        p_values_predict.insert(0, CQF.get_p_value_of_normal_test(sub_hist_predict.close.values))

    # initiate figure
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(hspace=1, wspace=0.1, top=0.95, left=0.05, right=0.95, bottom=0.1)

    # get x axis
    t = []
    count = 0
    for date_string, row in hist.iterrows():
        date_time = date_string.to_pydatetime()
        t.insert(0, mpd.date2num(date_time))
        count += 1
        if count >= len(predict_price):
            break

    # plot ax 1 prices
    ax1 = fig.add_subplot(427)
    plt.plot(t, predict_price, 'r')
    plt.plot(t, fact_price, 'b')

    # adjust axis
    mondays = mpd.WeekdayLocator(mpd.MONDAY)
    all_days = mpd.DayLocator()
    ax1.xaxis.set_major_locator(mondays)
    ax1.xaxis.set_minor_locator(all_days)
    formatter = mpd.DateFormatter('%m-%d-%Y')  # 2-29-2015
    ax1.xaxis.set_major_formatter(formatter)
    ax1.autoscale_view()
    ax1.xaxis_date()
    ax1.grid(True)

    # adjust plot
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.title(
        code + ' predicted prices(red) & fact prices(blue) [' + str(days_for_statistic) + ' to ' + str(
            days_for_predict) + ']')

    # plot ax 2 returns
    ax2 = fig.add_subplot(411)
    plt.fill_between(t, 0, fact_r, facecolor='b', alpha=0.3)
    plt.plot(t, predict_r, 'r')
    plt.plot(t, fact_r, 'b')

    # adjust axis
    mondays = mpd.WeekdayLocator(mpd.MONDAY)
    all_days = mpd.DayLocator()
    ax2.xaxis.set_major_locator(mondays)
    ax2.xaxis.set_minor_locator(all_days)
    formatter = mpd.DateFormatter('%m-%d-%Y')  # 2-29-2015
    ax2.xaxis.set_major_formatter(formatter)
    ax2.autoscale_view()
    ax2.xaxis_date()
    ax2.grid(True)

    # adjust plot
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.title(code + ' predicted returns(red) & fact returns(blue)[' + str(days_for_statistic) + ' to ' +
              str(days_for_predict) + '] -- ' + str(days_for_predict) + ' days return')

    # plot ax 3 p_value statistic history days
    ax3 = fig.add_subplot(412)
    plt.bar(t, p_values_statistic, alpha=0.5, color='g')
    line_05 = plt.Line2D(
        xdata=(t[0], t[-1]), ydata=(0.05, 0.05),
        color='r',
        linewidth=0.5,
        antialiased=True,
    )
    ax3.add_line(line_05)
    line_01 = plt.Line2D(
        xdata=(t[0], t[-1]), ydata=(0.01, 0.01),
        color='k',
        linewidth=0.5,
        antialiased=True,
    )
    ax3.add_line(line_01)

    # adjust axis
    mondays = mpd.WeekdayLocator(mpd.MONDAY)
    all_days = mpd.DayLocator()
    ax3.xaxis.set_major_locator(mondays)
    ax3.xaxis.set_minor_locator(all_days)
    formatter = mpd.DateFormatter('%m-%d-%Y')  # 2-29-2015
    ax3.xaxis.set_major_formatter(formatter)
    ax3.autoscale_view()
    ax3.xaxis_date()
    ax3.grid(True)

    # adjust plot
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.title(code + ' p-values for statistic history days (green bar) [' + str(days_for_statistic) + ' to ' + str(
        days_for_predict) + '] -- normal distribution test for ' + str(days_for_statistic) + ' days ')

    # plot ax 4 p_value predicted future days
    ax4 = fig.add_subplot(413)
    plt.bar(t, p_values_predict, alpha=0.5, color='g')
    line_05 = plt.Line2D(
        xdata=(t[0], t[-1]), ydata=(0.05, 0.05),
        color='r',
        linewidth=0.5,
        antialiased=True,
    )
    ax4.add_line(line_05)
    line_01 = plt.Line2D(
        xdata=(t[0], t[-1]), ydata=(0.01, 0.01),
        color='k',
        linewidth=0.5,
        antialiased=True,
    )
    ax4.add_line(line_01)

    # adjust axis
    mondays = mpd.WeekdayLocator(mpd.MONDAY)
    all_days = mpd.DayLocator()
    ax4.xaxis.set_major_locator(mondays)
    ax4.xaxis.set_minor_locator(all_days)
    formatter = mpd.DateFormatter('%m-%d-%Y')  # 2-29-2015
    ax4.xaxis.set_major_formatter(formatter)
    ax4.autoscale_view()
    ax4.xaxis_date()
    ax4.grid(True)

    # adjust plot
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.title(code + ' p-values for predicted future days (green bar) [' + str(days_for_statistic) + ' to ' + str(
        days_for_predict) + '] -- normal distribution test for ' + str(days_for_predict) + ' days ')

    # plot ax 5 histogram
    fig.add_subplot(428)
    errors = []
    for i in range(len(predict_r)):
        errors.append(fact_r[i] - predict_r[i])
    plt.hist(errors)
    plt.title(code + ' histogram of (fact return - predicted return) [' + str(days_for_statistic) + ' to ' + str(
        days_for_predict) + ']')
    plt.show()

    return


def plot_history_close_line(code='300403', days=365):
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # reverse close price
    close = hist['close'].sort_index(0)  # sort by index (from past to now)

    # plot close price movement
    x = np.linspace(1, len(close), len(close))
    plt.plot(x, close, 'b')
    plt.show()

    return


def plot_history_returns_movement(code='300403', days=365):
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # get returns
    ds_by_s = (hist['close'].shift(1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
    ds_by_s = ds_by_s[1:].sort_index(0)  # drop NAN & sort by index (from past to now)

    # plot the history returns movement
    x = np.linspace(1, len(ds_by_s), len(ds_by_s))
    plt.plot(x, ds_by_s, 'r')
    plt.show()

    return


def plot_history_returns_histogram(code='300403', days=365):
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # get returns
    ds_by_s = (hist['close'].shift(1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
    ds_by_s = ds_by_s[1:]  # drop NAN

    # plot histogram
    plt.hist(ds_by_s, 100)
    plt.show()

    return


def plot_candlestick(code='300403', days=200, width=0.6, color_up='r', color_down='g',
                     transparency=0.8):
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # reverse data order
    hist = hist.sort_index(0)

    # initiate plots
    fig, ax = plt.subplots(figsize=(20, 6))
    fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95)

    # plot candlestick
    half_width = width / 2.0
    for date_string, row in hist.iterrows():
        date_time = date_string.to_pydatetime()
        t = mpd.date2num(date_time)

        _open, _high, _close, _low = row[:4]

        if _close >= _open:
            color = color_up
            lower = _open
            height = _close - _open
        else:
            color = color_down
            lower = _close
            height = _open - _close

        line = plt.Line2D(
            xdata=(t, t), ydata=(_low, _high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )

        rect = plt.Rectangle(
            xy=(t - half_width, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(transparency)
        ax.add_patch(rect)
        ax.add_line(line)

    # adjust axis
    mondays = mpd.WeekdayLocator(mpd.MONDAY)
    all_days = mpd.DayLocator()
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(all_days)
    formatter = mpd.DateFormatter('%m-%d-%Y')  # 2-29-2015
    ax.xaxis.set_major_formatter(formatter)
    ax.autoscale_view()
    ax.xaxis_date()
    ax.grid(True)

    # adjust plot
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title(code + ' [days: ' + str(len(hist['close'])) + ']')
    plt.show()

    return


def plot_candlestick_mc_gbm(code='300403', days_total=80, days_short_predict=5, days_short_statistic=12,
                            days_long_predict=5, days_long_statistic=26):
    # set parameters
    width = 0.6
    color_up = 'r'
    color_down = 'g'
    transparency = 0.8

    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days_total)).strftime('%Y-%m-%d')
    hist = ts.get_h_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # reverse data order
    # hist = hist.sort_index(0)

    # initiate plots
    fig, ax = plt.subplots(figsize=(20, 6))
    fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95)

    # plot candlestick
    half_width = width / 2.0
    for date_string, row in hist.iterrows():
        date_time = date_string.to_pydatetime()
        t = mpd.date2num(date_time)

        _open, _high, _close, _low = row[:4]

        if _close >= _open:
            color = color_up
            lower = _open
            height = _close - _open
        else:
            color = color_down
            lower = _close
            height = _open - _close

        line = plt.Line2D(
            xdata=(t, t), ydata=(_low, _high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )

        rect = plt.Rectangle(
            xy=(t - half_width, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(transparency)
        ax.add_patch(rect)
        ax.add_line(line)

    # plot moving monte carlo geometric brownian motion (MMCGBM)
    def get_predict(s, days, simulation):
        # get returns
        ds_by_s = (s['close'].shift(1) - s['close']) / s[
            'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
        ds_by_s = np.nan_to_num(ds_by_s)  # change nan to zero

        # get parameters
        s0 = s.close.values[0]
        dt = 1.0 / len(ds_by_s.values)
        mu = np.mean(ds_by_s.values) / dt
        sigma = np.sqrt(np.var(ds_by_s.values)) / np.sqrt(dt)

        # calculate the expected return
        expected_price = CQF.get_ep_of_mc_gbm(mu=mu, sigma=sigma, dt=dt, s0=s0, days=days,
                                              simulation=simulation)
        return expected_price

    def get_predicts(hist_data, days_for_predict, days_for_statistic):
        # get predicts and facts
        predict_prices = []
        p_values = []
        for i in range(len(hist_data) - days_for_statistic + 1):  # i is test count [1,..)
            print(
                'percent:%0.4f %%' % (100.0 * (1.0 + i) / (len(hist_data) - days_for_statistic + 1)))

            sub_hist = hist_data[i:(days_for_statistic + i)]  # hist[)
            p = get_predict(sub_hist, days_for_predict, 5000)

            predict_prices.insert(0, p)  # insert at beginning
            p_values.insert(0, CQF.get_p_value_of_normal_test(sub_hist.close.values))

        _t = []
        count = 0
        for _date_string, _row in hist_data.iterrows():
            _date_time = _date_string.to_pydatetime()
            _t.insert(0, mpd.date2num(_date_time))
            count += 1
            if count >= (len(predict_prices) - days_for_predict):
                break

        last_time_datetime = mpd.num2date(_t[-1])
        for i in range(days_for_predict):
            next_datetime = last_time_datetime + datetime.timedelta(days=i)
            _t.append(mpd.date2num(next_datetime))

        return _t, predict_prices, p_values

    s_t, s_prices, s_pvalues = get_predicts(hist, days_short_predict, days_short_statistic)
    l_t, l_prices, l_pvalues = get_predicts(hist, days_long_predict, days_long_statistic)
    plt.plot(s_t, s_prices, 'r')
    plt.plot(l_t, l_prices, 'b')

    # adjust axis
    mondays = mpd.WeekdayLocator(mpd.MONDAY)
    all_days = mpd.DayLocator()
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(all_days)
    formatter = mpd.DateFormatter('%m-%d-%Y')  # 2-29-2015
    ax.xaxis.set_major_formatter(formatter)
    ax.autoscale_view()
    ax.xaxis_date()
    ax.grid(True)

    # adjust plot
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title(code + '  with predicted future prices: [' + str(days_short_statistic) + ' to ' + str(
        days_short_predict) + '](red), [' + str(days_long_statistic) + ' to ' + str(
        days_long_predict) + '](blue)\n<The Predicted Days Are Shown In Calendar Dates Not In Trading Dates>')
    plt.show()

    return
