# functions learnt in CQF
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


def phi(n=12):
    # use n random numbers to generate a normal_distributed_random_variable
    mean = 1.0 / 2
    sigma = math.sqrt(1.0 / 12)
    s = 0.0
    for i in range(n):
        x = random.random()
        s = s + x
    y = (s - n * mean) / (sigma * math.sqrt(n))
    return y


def plot_gbm(mu=-0.01120817537498, sigma=0.3, dt=1.0 / 250, s0=58.89, days=99):
    # plot a simulated geometric brownian motion after [s0] for [days]
    s = [s0]
    for i in range(days):
        s.append(s[-1] * (1 + mu * dt + sigma * phi() * np.sqrt(dt)))

    x = np.linspace(1, days + 1, days + 1)
    plt.plot(x, s, 'r')
    plt.show()
    return


def get_ep_of_mc_gbm(mu=-0.01120817537498, sigma=0.3, dt=1.0 / 250, s0=58.89, days=99, simulation=5000):
    m = []
    for i in range(simulation):
        # equation from CQF M1S4 page 12
        m.append(s0*np.exp((mu-1/2*np.square(sigma))*(days*dt)+sigma*phi()*np.sqrt(days*dt)))
    return np.mean(m)


def get_p_value_of_normal_test(list):
    result = ss.normaltest(list)
    return result[1]
