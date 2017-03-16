import tushare as ts
import pickle
import numpy as np
import matplotlib.pyplot as plt
import QuantCN as qc
import ML
import mix

# qc.plot_history_close_line('300403', 365)
# qc.plot_history_returns_movement('300403', 365)
# qc.plot_history_returns_histogram('300403', 365)
# qc.plot_candlestick('000514')
# qc.plot_candlestick_mc_gbm('000514', 150, 5, 60, 10, 90)
# qc.plot_predicts_and_facts('000514', 150, 10, 90)
# qc.plot_gbm_simulation('300403', 100, 365):
qc.get_stocks_of_gbm(file_name='data0316.pkl',
                     days_for_statistic=90,
                     days_for_predict=5,
                     simulation=5000,
                     bottom=0.035,
                     top=0.04)
# ML.download_data('data0316.pkl', 365)
# ML.test_parameters('002110')
# ML.dl('002100', 20, 90, 5, 10, 0.8, 'data0310.pkl', True)
# ML.dl_current('002100', 20, 90, 5, 10, 'data0310.pkl', True)
# mix.get_parameters_of_strong_shares('data0310.pkl', 'parameters0312.pkl',90, 1, 5000, 0.01, 0.03)
