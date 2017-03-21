import data
import gbm
import assembler
import deeplearning
import tushare as ts
import numpy as np
import time
import warnings

# data.download_deals('deals0320.pkl', 365)
# data.download_hist('hist0321.pkl', (365*3))
# gbm.get_stocks_mc_gbm('hist0321.pkl', 'gbm0321.pkl', 60, 5, 5000, 0.04, 0.15, 0.05)
assembler.get_stocks_mc_gbm_dl('hist0321.pkl', 'gbm0321.pkl')
# print(deeplearning.dl_predict('600760', 10, 500, 15, 50, 'hist0321.pkl', True))
