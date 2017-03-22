import data
import gbm
import assembler
import deeplearning
import tushare as ts
import numpy as np
import time
import warnings


# data.download_hist('hist0322.pkl', (365*3))
# gbm.get_stocks_mc_gbm('hist0322.pkl', 'gbm0322.pkl', 60, 5, 5000, 0.04, 0.15, 0.05)
# assembler.get_stocks_mc_gbm_dl('hist0322.pkl', 'gbm0322.pkl')
# data.download_deals('deals0320.pkl', 365)
print(deeplearning.dl_predict(target='i_399001', hist_file='hist0322.pkl'))
