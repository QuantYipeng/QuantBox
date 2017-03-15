import pickle
import QuantCN as qc
import ML


def get_parameters_of_strong_shares(data_file='data0310.pkl', parameter_file='parameters0312.pkl',
                                    days_for_statistic=90, days_for_predict=1, simulation=5000, bottom=0.01, top=0.03):
    # [{'code':'300403',
    #   'return':0.01,
    #   'p-value':0.05},]
    gbm = qc.load_statistic(data_file, days_for_statistic, days_for_predict, simulation, bottom, top)
    parameters = []
    for i in range(len(gbm)):
        if gbm[i]['p-value'] < 0.2:
            print('[Getting Parameters] code: ' + gbm[i]['code'] +
                  ' return: ' + str(gbm[i]['return']) +
                  ' p-value: ' + str(gbm[i]['p-value']))
            parameter = ML.test_parameters(gbm[i]['code'])
            print('[Parameters Gotten] code:' + gbm[i]['code'])
            print(parameter)
            parameters.append(parameter)

    with open(parameter_file, 'wb') as f:  # open file with write-mode
        pickle.dump(parameters, f)  # serialize and save object
