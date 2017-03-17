import deeplearning
import data
import gbm

# data.download('hist0316.pkl', (365*3))
gbm.get_stocks_mc_gbm(hist_file='hist0316.pkl',
                      gbm_file='gbm0316.pkl',
                      days_for_statistic=60,
                      days_for_predict=5,
                      simulation=5000,
                      bottom=0.04,
                      top=0.15,
                      p_value=0.05)
deeplearning.get_stocks_mc_gbm_dl(hist_file='hist0316.pkl', gbm_file='gbm0316.pkl')

