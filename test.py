import data
import gbm
import assembler
import deeplearning
import tushare as ts
import numpy as np
import time
import warnings

# data.download('hist0317.pkl', (365*3))
# gbm.get_stocks_mc_gbm('hist0317.pkl', 'gbm0317.pkl', 60, 5, 5000, 0.04, 0.15, 0.05)
# deeplearning.dl_predict('300176',10,500,15,10,'hist0317.pkl',True)
# assembler.get_stocks_mc_gbm_dl('hist0317.pkl', 'gbm0317.pkl')

deeplearning.dl_back_test('300176', 2, 60, 5, 10, 0.9, 'hist0317.pkl', True)
