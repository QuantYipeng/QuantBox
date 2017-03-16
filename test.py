import deeplearning
import data
import gbm

'''
gbm.get_stocks_mc_gbm(hist_file='data0316.pkl',
                      result_file='gbm0316.pkl',
                      days_for_statistic=30,
                      days_for_predict=5,
                      simulation=5000,
                      bottom=0.05,
                      top=0.1,
                      p_value=0.1)
'''
# data.download('data0318.pkl', 365)
# deeplearning.get_best_parameters('300308')
'''
deeplearning.dl_back_test(target='300308',
                          correlations=10,
                          days=120,
                          length=15,
                          label_size=10,
                          test_ratio=0.9,
                          data_file='data0316.pkl',
                          show_figure=True)
'''
deeplearning.dl_predict(target='300308',
                        correlations=10,
                        days=90,
                        length=15,
                        label_size=10,
                        data_file='',
                        show_figure=True)

