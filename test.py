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
qc.load_statistic(qc.get_all_history_data(365), 5, 10000, 0.05, 0.055)
