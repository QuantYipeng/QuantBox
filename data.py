import datetime
import matplotlib.dates as mpd
import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import pickle
from tqdm import tqdm


def download_hist(file_name='data0309.pkl', calendar_days=365):
    # using get_k_hist to download

    # get stock names
    stock_info = ts.get_stock_basics()

    # set date
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    days_before = (datetime.datetime.now() - datetime.timedelta(days=calendar_days)).strftime(
        '%Y-%m-%d')

    # download stocks data
    code = []
    data = []

    for i in tqdm(stock_info.index, desc='[Downloading Stocks]'):
        try:
            # get data
            hist = ts.get_k_data(i, start=days_before, end=today)  # (from past to now)
            code.append(i)
            data.append(hist)
        except:
            continue

    # download indices data
    indices = ['399001', '399006']
    for i in tqdm(indices, desc='[Downloading Indices]'):
        if i in code:
            continue
        else:
            try:
                # get data
                hist = ts.get_k_data(i, index=True, start=days_before, end=today)  # (from past to now)
                code.append(i)
                data.append(hist)
            except:
                continue

    # write into files
    content = dict(zip(code, data))
    print(content['399001'])

    fn = file_name
    with open(fn, 'wb') as f:  # open file with write-mode
        pickle.dump(content, f)  # serialize and save object
    return


def download_deals(file_name='deals0317.pkl', calendar_days=365):
    content = []

    # get stock names
    stock_info = ts.get_stock_basics()

    for code in tqdm(stock_info.index, desc='[Downloading Deals]'):
        try:
            for d in range(calendar_days + 1):
                date = (datetime.datetime.now() - datetime.timedelta(days=d)).strftime(
                    '%Y-%m-%d')
                content.append({code: {date: ts.get_tick_data(code, date)}})
        except:
            continue

    fn = file_name
    with open(fn, 'wb') as f:  # open file with write-mode
        pickle.dump(content, f)  # serialize and save object
    return


def plot_history_closes_line(code='300403',
                             days=365):
    """
    public
    (get_k_hist)
    plot history close price line
    """
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_k_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # plot close price movement
    x = np.linspace(1, len(hist['close']), len(hist['close']))
    plt.plot(x, hist['close'], 'b')
    plt.show()

    return


def plot_history_returns_line(code='300403',
                              days=365):
    """
    public
    (get_k_hist)
    plot history return line
    """
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_k_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # get returns
    ds_by_s = (hist['close'].shift(-1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
    ds_by_s = ds_by_s[:-1]  # drop NAN

    # plot the history returns movement
    x = np.linspace(1, len(ds_by_s), len(ds_by_s))
    plt.plot(x, ds_by_s, 'b')
    plt.show()

    return


def plot_history_returns_histogram(code='300403',
                                   days=365):
    """
    public
    (get_k_hist)
    plot history returns histogram
    """
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_k_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    # get returns
    ds_by_s = (hist['close'].shift(-1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
    ds_by_s = ds_by_s[:-1]  # drop NAN

    # plot histogram
    plt.hist(ds_by_s, 100)
    plt.show()

    return


def plot_history_candlestick(code='300403',
                             days=200,
                             width=0.6,
                             color_up='r',
                             color_down='g',
                             transparency=0.8):
    """
    public
    (get_k_hist)
    plot candle stick
    """
    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    hist = ts.get_k_data(code, start=one_year_before, end=today)

    # initiate plots
    fig, ax = plt.subplots(figsize=(20, 6))
    fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95)

    # plot candlestick
    half_width = width / 2.0
    for date_string, row in hist.iterrows():
        date_time = datetime.datetime.strptime(row[0], "%Y-%m-%d")
        t = mpd.date2num(date_time)

        _open, _close, _high, _low = row[1:5]

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
