import deeplearning
import data
import gbm

gbm.get_stocks_mc_gbm(hist_file='data0316.pkl',
                      result_file='gbm0316.pkl',
                      days_for_statistic=30,
                      days_for_predict=5,
                      simulation=5000,
                      bottom=0.05,
                      top=0.1,
                      p_value=0.1)
# data.download('data0316.pkl', 365)
# deeplearning.get_best_parameters('002110')
# deeplearning.dl_back_test('002100', 20, 90, 5, 10, 0.8, 'data0310.pkl', True)
# deeplearning.dl_predict('002100', 20, 90, 5, 10, 'data0310.pkl', True)
