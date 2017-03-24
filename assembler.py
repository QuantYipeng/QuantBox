import pickle
import deeplearning


def get_stocks_mc_gbm_dl(hist_file='data0321.pkl', gbm_file='gbm0321.pkl'):
    # 1. low error means history is learn-able,
    #    which also indicates that the history has been recurring
    # 2. low probability means the this situation is not in history
    # 3. the less the red, the better the prediction

    with open(gbm_file, 'rb') as f:
        content = pickle.load(f)  # read file and build object
    results = []
    for i in range(len(content)):
        print('[Processing] ' + str(i + 1) + ' of ' + str(len(content)))
        expected_return, type1, type2, err = deeplearning.dl_predict(target=content[i]['code'],
                                                                     nb_of_correlations=10,
                                                                     days_for_statistic=500,
                                                                     data_length=50,
                                                                     recent_days=20,
                                                                     hist_file=hist_file,
                                                                     show_figure=False)
        results.append({'code': content[i]['code'],
                        'E_return_1d': expected_return,
                        'type1': type1,
                        'type2': type2,
                        'err': err,
                        'E_return_1w': content[i]['expected_return'],
                        'p_value': content[i]['p_value']})
    results.sort(key=lambda k: (k.get('E_return_1d', 0)))

    for result in results:
        print(result)
