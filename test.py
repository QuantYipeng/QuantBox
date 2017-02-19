import QuantCN as qc
import numpy as np
import matplotlib.pyplot as plt

# qc.write_er_mc_gbm('/Users/yipengli/Documents/QuantLibData/20170216_5_60.npy', 5, 10000, 60)
# qc.load_('/Users/yipengli/Documents/QuantLibData/20170216_5_60.npy', 0.05, 0.055)
# qc.plot_candlestick('000514')
# qc.predict('000514', 180, 365)
qc.plot_predicts_and_facts('000514', 10, 60, 150)
# qc.plot_candlestick_mc_gbm('000514', 150, 5, 20, 10, 60)
