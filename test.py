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
# qc.write_all_history_data('data0308.pkl', 365)
# print(qc.is_booming_stock(code='601212', days=365))
# qc.load_statistic('data0310.pkl', 90, 1, 5000, 0.02, 0.03)
# qc.load_all_statistic('data0222.pkl', 90, 5, 5000, 0.03, 0.005, 0.05)
# ML.download_data('data0310.pkl', 365)
# ML.test_parameters('002110')
# ML.dl('002100', 20, 90, 5, 10, 0.8, 'data0310.pkl', True)
ML.dl_current('002100', 20, 90, 5, 10, 'data0310.pkl', True)
# mix.get_parameters_of_strong_shares('data0310.pkl', 'parameters0312.pkl',90, 1, 5000, 0.01, 0.03)



