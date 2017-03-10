import QuantCN as qc
import numpy as np
import matplotlib.pyplot as plt
import ML
import tushare as ts

# qc.plot_history_close_line('300403', 365)
# qc.plot_history_returns_movement('300403', 365)
# qc.plot_history_returns_histogram('300403', 365)
# qc.plot_candlestick('000514')
# qc.plot_candlestick_mc_gbm('000514', 150, 5, 60, 10, 90)
# qc.plot_predicts_and_facts('000514', 150, 10, 90)
# qc.plot_gbm_simulation('300403', 100, 365):
# qc.write_all_history_data('data0308.pkl', 365)
# qc.load_statistic('data0222.pkl', 90, 5, 5000, 0.055, 0.06)
# qc.load_all_statistic('data0222.pkl', 90, 5, 5000, 0.03, 0.005, 0.05)
# ML.download_data('data0310.pkl', 365)

p = ts.get_stock_basics().index
pl = []
for i in range(len(p)):
    pl.append(p[i])
pl.append('399001')

ML.dl(target='300403', pool=pl, n=10, days=90, length=3, label_size=8, test_ratio=0.7, data_file='data0310.pkl')
