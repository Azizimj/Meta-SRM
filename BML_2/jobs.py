# scripts run with problem in windows, so here we are

from itertools import product
import os

import numpy as np

handler = 'python'


exp_type = 'linear'
exp_type = 'MAB'


if exp_type == 'linear':
    app_ = 'linear_bandits.py'
    train_lengths = [0, 20, 40, 60, 80, 100]
    train_lengths = [0, 200, 400, 600, 800, 1000]
    Ts = [2000]
    Hs = [20, 100]
    Hs = [20]
    Ks = [6]
    ds = [6]
    iters = [10, 20]
    iters = [100]
    iters = [5]
    algs = ['oracle', 'misspecified', 'mts']  # linear

    for T, iter_, H, d, K, alg, train in product(Ts, iters, Hs, ds, Ks, algs, train_lengths):
        if alg != 'mts' and train>0:
            continue
        if alg == 'mts' and train == 0:
            continue
        os.system(f"{handler} {app_} --T {T} --iters {iter_} --H {H} --d {d} --K {K} --alg {alg} --train {train}")

elif exp_type == 'MAB':

    train_lengths = np.arange(0, 5001, 500)  # MAB
    num_random = [0, 10]  # MAB
    Ts = [10000]  # MAB
    # Ts = [100]  # MAB
    Hs = [10]  # MAB
    Hs = [20]  # MAB
    Ks = [6]
    iters = [5]
    algs = ['mts-no-cov', 'mts', 'oracle', 'misspecified']  # MAB
    # algs = ['oracle', 'misspecified']  # MAB
    app_ = 'exp.py'
    instance_ = 'block'
    types = ['standard']

    for T, type_, iter_, H, K, alg, train, num_rand in product(Ts, types, iters, Hs, Ks, algs, train_lengths, num_random):
        if alg not in ['mts', 'mts-no-cov'] and (train>0 or num_rand>0):
            continue
        if alg in ['mts', 'mts-no-cov'] and train == 0:
            continue
        if alg in ['mts'] and num_rand==0:
            continue
        if alg in ['mts-no-cov'] and num_rand>0:
            continue
        os.system(f"{handler} {app_} --type {type_} --instance {instance_} --T {T} --iters {iter_} --H {H} --K {K} "
                  f"--alg {alg} --train {train} --num_random {num_rand}")

