import random
import math
import matplotlib.pyplot as plt
import scipy.stats as ss
import datetime
import numpy as np
import pickle
import tushare as ts


def _phi(n=12):
    # use n random numbers to generate a normal_distributed_random_variable
    mean = 1.0 / 2
    sigma = math.sqrt(1.0 / 12)
    s = 0.0
    for i in range(n):
        x = random.random()
        s = s + x
    y = (s - n * mean) / (sigma * math.sqrt(n))
    return y


def _plot_gbm(mu=-0.011,
              sigma=0.3,
              dt=1.0 / 250,
              s0=58.89,
              days=99):
    # plot a simulated geometric brownian motion after [s0] for [days]
    s = [s0]
    for i in range(days):
        s.append(s[-1] * (1 + mu * dt + sigma * _phi() * np.sqrt(dt)))

    x = np.linspace(1, days + 1, days + 1)
    plt.plot(x, s, 'r')
    plt.show()
    return


def _get_ep_of_mc_gbm(mu=-0.011,
                      sigma=0.3,
                      dt=1.0 / 250,
                      s0=58.89,
                      days=99,
                      simulation=5000):
    m = []
    for i in range(simulation):
        # equation from CQF M1S4 page 12
        m.append(s0 * np.exp((mu-1/2*np.square(sigma)) * (days*dt) + sigma * _phi() * np.sqrt(days * dt)))
    return np.mean(m)


def _get_p_value_of_normal_test(l):
    result = ss.normaltest(l)
    return result[1]


def _is_booming_stock(content,
                      code='300403',
                      days_for_statistic=365):
    """
    public
    (for get_k_hist)
    is this stock continuing raised 5 times?
    """

    # content is load from dataxxxx.pkl which looks like {'300403':hist,'002727':hist, ...}
    hist = content[code][-days_for_statistic:]

    ds_by_s = (hist['close'].shift(-1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)
    ds_by_s = np.nan_to_num(ds_by_s)

    is_booming = 0
    for i in range(len(ds_by_s)):
        if ds_by_s[i] > 0.09:
            is_booming += 1
        else:
            is_booming = 0

        if is_booming == 5:
            # booming stock
            return 1
    return 0


def _calculate_mc_gbm(hist,
                      days_for_predict=5,
                      simulation=5000):
    """
    private
    (get_k_hist)
    get the simulated geometric brownian motion result
    """

    ds_by_s = (hist['close'].shift(-1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)

    # get parameters
    s0 = hist['close'].values[-1]
    dt = 1.0 / len(ds_by_s.values[:-1])
    mu = float(np.mean(ds_by_s.values[:-1]) / dt)
    sigma = np.sqrt(np.var(ds_by_s.values[:-1])) / np.sqrt(dt)

    # calculate the expected return
    expected_price = _get_ep_of_mc_gbm(mu=mu,
                                       sigma=sigma,
                                       dt=dt,
                                       s0=s0,
                                       days=days_for_predict,
                                       simulation=simulation)
    expected_return = (expected_price - s0) / s0

    # calculate the p value whether the history returns are normally distributed (p<0.05 means significant)
    p_value = _get_p_value_of_normal_test(np.nan_to_num(ds_by_s))

    return expected_return, p_value


def plot_gbm_simulation(code='300403',
                        days_for_predict=100,
                        calendar_days=365):
    """
    public
    (get_k_hist)
    plot a simulated geometric brownian motion of future close price of the given stock
    """

    # get data
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    one_year_before = (datetime.datetime.now() - datetime.timedelta(days=calendar_days)).strftime('%Y-%m-%d')
    hist = ts.get_k_data(code, start=one_year_before, end=today)  # reverse order (from now to past)

    ds_by_s = (hist['close'].shift(-1) - hist['close']) / hist[
        'close']  # the return from today to tomorrow store in today in reverse order (from now to past)

    # get parameters
    s0 = hist['close'].values[-1]
    dt = 1.0 / len(ds_by_s.values[:-1])
    mu = float(np.mean(ds_by_s.values[:-1]) / dt)
    sigma = np.sqrt(np.var(ds_by_s.values[:-1])) / np.sqrt(dt)
    max_return = np.max(ds_by_s.values[:-1])
    min_return = np.min(ds_by_s.values[:-1])

    # print parameters
    print('price (current) = %0.2f' % s0)
    print('return (max) = %0.2f %%' % (max_return * 100))
    print('return (min) = %0.2f %%' % (min_return * 100))
    print('return (mu) = %0.2f %%' % (mu * 100))
    print('return (sigma) = %0.2f %%' % (sigma * 100))
    print('history (len) = %d' % len(ds_by_s))

    # plot simulated geometric brownian motion
    _plot_gbm(mu=mu, sigma=sigma, dt=dt, s0=s0, days=days_for_predict)

    return


def get_stocks_mc_gbm(hist_file='data0316.pkl',
                      result_file='gbm0316.pkl',
                      days_for_statistic=90,
                      days_for_predict=5,
                      simulation=5000,
                      bottom=0.055,
                      top=0.06,
                      p_value=0.1):
    """
    public
    (get_k_hist)
    function: get the stocks using geometric brownian motion in the specific interval
    return: a sorted list of selected dict. which dict looks like
    [{'code':'300403',
      'return':0.01,
      'p-value':0.05},]
    """
    # for get_k_hist

    fn = hist_file
    with open(fn, 'rb') as f:
        content = pickle.load(f)  # read file and build object

    def get_all_er_of_mc_gbm(_content, _days_for_statistic, _days_for_predict=5, _simulation=5000):
        # calculate the expected returns
        _count = 0
        _raw_results = []
        for _key, _value in _content.items():
            # key: code, value: hist
            _count += 1
            _raw_result = {}
            try:
                _raw_result['code'] = _key
                # calculate the geometric brownian motion monte carlo result
                _raw_result['expected_return'], _raw_result['p_value'] = _calculate_mc_gbm(
                    _value[-_days_for_statistic:],
                    _days_for_predict,
                    _simulation)

                # handle the NaN (use 0.0 instead)
                _raw_result['expected_return'] = np.nan_to_num(_raw_result['expected_return'])
                _raw_result['p_value'] = np.nan_to_num(_raw_result['p_value'])

                # print log
                print('[GMB Monte Carlo:  %0.2f %%] Code: %s, Expected Return: %0.4f %%, P-Value: %0.4f %%' %
                      ((100.0 * _count / len(_content)),
                       (_raw_result['code']),
                       (_raw_result['expected_return'] * 100),
                       (_raw_result['p_value'] * 100)))

                # add into results
                _raw_results.append(_raw_result)

            except:
                continue
        return _raw_results

    raw_results = get_all_er_of_mc_gbm(content, days_for_statistic, days_for_predict, simulation)

    # get the stocks in [bottom, top)
    print('[Refining Result] taking off the booming stocks')
    refined_results = []
    for raw_result in raw_results:
        if bottom <= raw_result['expected_return'] < top:
            if _is_booming_stock(content, raw_result['code'], days_for_statistic) == 1:
                print('[Refining Result] Code: %s is Booming Stock' % raw_result['code'])
            else:
                if raw_result['p_value'] < p_value:
                    print('[Refining Result] Code: %s, Expected Return: %0.4f %%, P-Value: %0.4f %%' %
                          (raw_result['code'],
                           raw_result['expected_return'],
                           raw_result['p_value']))
                    refined_results.append(raw_result)
                else:
                    print('[Refining Result] Code: %s is unstable' % raw_result['code'])

    # sort refined_results by 'expected_return'
    refined_results.sort(key=lambda k: (k.get('expected_return', 0)))

    print('[Final Result]')
    for x in refined_results:
        print(x)

    # store the results
    fn = result_file
    with open(fn, 'wb') as f:  # open file with write-mode
        pickle.dump(refined_results, f)  # serialize and save object

    return refined_results
