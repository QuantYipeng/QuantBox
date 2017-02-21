import QuantCN as qc
import numpy as np
import matplotlib.pyplot as plt


# qc.plot_history_close_line('300403', 365)
# qc.plot_history_returns_movement('300403', 365)
# qc.plot_history_returns_histogram('300403', 365)
# qc.plot_candlestick('000514')
# qc.plot_candlestick_mc_gbm('002034', 150, 5, 20, 10, 60)
# qc.plot_predicts_and_facts('000514', 150, 10, 60)
# qc.plot_gbm_simulation('300403', 100, 365):
# qc.write_all_history_data('data0220.pkl', 365)
# qc.load_statistic('data0220.pkl', 5, 5000, 0.055, 0.06)
qc.load_all_statistic('data0220.pkl', 5, 5000, 0.03, 0.005, 0.05)
