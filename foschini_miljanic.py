# -*- coding: utf-8 -*- #

""" Simulation for Distributed Power Control
"""

import random
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

stations = 5
iterations = 30


# 利得
def generate_random_gain(stations):
    g = np.random.rand(stations, stations)
    for i in range(stations):
        g[i][i] = random.random() / 2 + 0.5
    return g


g = generate_random_gain(stations)


# 所要SINR
def generate_random_sinr(stations):
    return np.random.rand(stations)


gamma = generate_random_sinr(stations)


# 雑音
def generate_random_noise(stations):
    return np.random.randint(low=1, high=12, size=stations)


delta = generate_random_noise(stations)


def solve_optimal(g, delta, gamma):
    """
    solve optimal power distribution from equation
    :param g: gain matrix
    :param delta: noise
    :param gamma: required SINR
    :return: optimal power for each station
    """
    f = np.zeros([stations, stations])
    for i in range(stations):
        for j in range(stations):
            if not i == j:
                f[i][j] = gamma[i] * g[i][j] / g[i][i]
            else:
                pass
    eta = np.zeros((stations, 1))
    for i in range(stations):
        eta[i][0] = gamma[i] * (delta[i] ** 2) / g[i][i]
    i_minus_f = (np.identity(stations) - f)
    i_minus_f_inverse = np.linalg.inv(i_minus_f)
    p_opt = np.dot(i_minus_f_inverse, eta)
    ret = np.transpose(p_opt)[0]
    return ret


def recieving_sinr(p):
    """
    :param p: list of power for each station
    :return: current reception SINR for each station
    """
    y = np.zeros(stations)
    for i in range(stations):
        tmp = [g[i][j] * p[j] for j in range(stations) if not i == j]
        divisor = sum(tmp) + (delta[i] ** 2)
        dividend = g[i][i] * p[i]
        y[i] = dividend / divisor
    return y


def simulate_step(prev_p, prev_sinr):
    """
    calculate step in foschini-miljanic algorithm
    :param prev_p: previous power for each station
    :return: next power for each station
    """
    return [(gamma[s] / sinr) * prev_p[s] for s, sinr in enumerate(prev_sinr)]


def simulate_steps(prev_p, i):
    """
    perform steps for i times
    :param prev_p: initial power
    :param i: times to run steps
    :return: final power for each station
    """
    if i == 0:
        print("======")
        return prev_p, [prev_p]
    else:
        prev_rec_sinr = recieving_sinr(prev_p)
        print("======\ncurrent power: %s\ncurrent SINR: %s" % (prev_p, prev_rec_sinr))
        next_p = simulate_step(prev_p, prev_rec_sinr)
        res, past_p = simulate_steps(next_p, i - 1)
        return res, ([prev_p] + past_p)


def plot_graph(step_p, opt_p):
    """
    plot change of power through time
    :param step_p: power for each step in the algorithm
    :param opt_p: optimal power
    :return: show graph
    """
    for s in range(stations):
        plt.hlines(opt_p[s], xmin=0, xmax=iterations)
        height = [step_p[i][s] for i in range(len(step_p))]
        plt.plot(range(len(step_p)), height)
    plt.xlabel("iterations")
    plt.ylabel("power")
    plt.show()


def write_3d_graph(step_p, p_opt):
    """
    plot 3d graph of power for first 3 stations
    :param step_p: power for each step in the algorithm
    :param p_opt: optimal power
    :return: show graph
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    max_p = max([max(w) for w in step_p])
    ax.set_xlim3d(0, max_p)
    ax.set_ylim3d(0, max_p)
    ax.set_zlim3d(0, max_p)
    ax.set_xlabel("station 1")
    ax.set_ylabel("station 2")
    ax.set_zlabel("station 3")
    x = [step_p[t][0] for t in range(iterations)]
    y = [step_p[t][1] for t in range(iterations)]
    z = [step_p[t][2] for t in range(iterations)]
    ax.plot(x, y, z, label="power of first 3 stations")
    ax.scatter(p_opt[0], p_opt[1], p_opt[2])
    ax.legend()
    plt.show()


def main():
    initial_power = [1e-6 for _ in range(stations)]
    alg_result, step_p = simulate_steps(initial_power, iterations)
    p_opt = solve_optimal(g, delta, gamma)
    print("algorithm result: ", alg_result)
    print("optimal power: ", p_opt)
    plot_graph(step_p, p_opt)
    write_3d_graph(step_p, p_opt)


if __name__ == '__main__':
    main()
