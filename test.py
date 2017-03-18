import data
import gbm
import assembler


data.download('hist0317.pkl', (365*3))
gbm.get_stocks_mc_gbm(hist_file='hist0317.pkl',
                      gbm_file='gbm0317.pkl',
                      days_for_statistic=60,
                      days_for_predict=5,
                      simulation=5000,
                      bottom=0.04,
                      top=0.15,
                      p_value=0.05)
assembler.get_stocks_mc_gbm_dl(hist_file='hist0317.pkl', gbm_file='gbm0317.pkl')

