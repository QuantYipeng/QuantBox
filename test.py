import data
import gbm
import assembler
import deeplearning
import tushare as ts
import time


# data.download('hist0317.pkl', (365*3))
# gbm.get_stocks_mc_gbm('hist0317.pkl', 'gbm0317.pkl', 60, 5, 5000, 0.04, 0.15, 0.05)
# deeplearning.dl_predict('300176',10,500,15,10,'hist0317.pkl',True)
# assembler.get_stocks_mc_gbm_dl('hist0317.pkl', 'gbm0317.pkl')


def is_in_time_interval(start, target, end):
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


def get_big_deal_statistic(code='300403', date='2017-03-17', vol=0):
    results = []
    df = ts.get_sina_dd(code, date, vol)
    print(len(df))
    print(len([x for x in df['type'] if x == '卖盘']))
    print(len([x for x in df['type'] if x == '中性盘']))
    print(len([x for x in df['type'] if x == '买盘']))

    for index, row in df.iterrows():
        if is_in_time_interval('09:25:00', row['time'], '10:00:00'):
            print('1')
            print(row)
        elif is_in_time_interval('10:00:00', row['time'], '10:30:00'):
            print('2')
            print(row)
        elif is_in_time_interval('10:30:00', row['time'], '11:00:00'):
            print('3')
            print(row)


get_big_deal_statistic()
