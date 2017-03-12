import pickle
import QuantCN as qc
import ML


def get_parameters_of_strong_shares(data_file='data0310.pkl', parameter_file='parameters0312.pkl',
                                    days_for_statistic=90, days_for_predict=1, simulation=5000, bottom=0.01, top=0.03):
    # dic{code,p-value}
    dic = qc.load_statistic(data_file, days_for_statistic, days_for_predict, simulation, bottom, top)
    parameters = []
    for key, value in dic.items():
        if value < 0.2:
            parameters.append(ML.test_parameters(key))
    print(parameters)
    print(dic)
    with open(parameter_file, 'wb') as f:  # open file with write-mode
        pickle.dump(parameters, f)  # serialize and save object
