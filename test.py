import data
import gbm
import assembler
import deeplearning
import tushare as ts
import numpy as np
import time
import warnings


date = '0324'
# data.download_hist('hist'+date+'.pkl')
# gbm.get_stocks_mc_gbm('hist'+date+'.pkl', 'gbm'+date+'.pkl', 20, 5, 5000, 0.05, 0.15, 0.05)
# assembler.get_stocks_mc_gbm_dl('hist'+date+'.pkl', 'gbm'+date+'.pkl')
# data.download_deals('deals'+date+'.pkl', 365)
print(deeplearning.dl_predict(target='600592', hist_file='hist'+date+'.pkl'))
