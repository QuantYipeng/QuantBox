import QuantCN as qc
import numpy as np
import matplotlib.pyplot as plt

# qc.load_statistic(qc.load_all_er_of_mc_gbm(data_file='data_0220.npy', 5), 0.05, 0.055)
# qc.plot_candlestick('000514')
# qc.predict('000514', 180, 365)
# qc.plot_predicts_and_facts('000514', 10, 60, 150)
# qc.plot_candlestick_mc_gbm('002034', 150, 5, 20, 10, 60)
qc.write_all_history_data('data_0220.npy', days=365)
