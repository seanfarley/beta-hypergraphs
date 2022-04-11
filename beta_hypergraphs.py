#!/usr/bin/env python3

import itertools
import time

import numba as nb
import numpy as np


def deg_seq(hyperedges, vertices=None):
    """Get the sequence of degrees."""
    if vertices is None:
        vertices = list(set(itertools.chain.from_iterable(hyperedges)))

    n = len(vertices)
    d = np.zeros(n)
    for i in range(n):
        for e in hyperedges:
            if vertices[i] in e:
                d[i] += 1

    return d


def beta_fixed_point_R(degrees, k, sets, max_iter=500, tol=0.0001, beta=None):
    """Use a fixed point algorithm to calculate the MLE."""
    n = len(degrees)
    if beta is None:
        beta = np.zeros(n)

    convergence = False
    steps = 0

    # There are more efficient methods for calculating e^{beta_S} for each S in
    # n\choose k-1, e.g using a dynamic algorithm
    # sets = list(itertools.combinations(range(n), k - 1))
    prod_beta = np.ones(len(sets))

    while(not convergence and steps < max_iter):
        exp_beta = np.exp(beta)
        old_beta = beta.copy()
        if any(np.isinf(old_beta)):
            return None

        for i in range(len(sets)):
            prod_beta[i] = np.prod([exp_beta[i] for i in sets[i]])

            if np.isinf(prod_beta[i]):
                print("Infinite beta")
                return

        for i in range(n):
            sum_q = 0
            for j in range(len(sets)):
                tt = sets[j]
                if i not in tt:
                    sum_q += prod_beta[j] / (1
                                             + prod_beta[j]
                                             * exp_beta[i])
                    if np.isinf(sum_q):
                        print("Infinite beta")
                        return

            beta[i] = np.log(degrees[i]) - np.log(sum_q)

        diff = max(abs(old_beta - beta))
        # print(f"diff= {diff} -------- steps= {steps}")
        # print(beta)
        if diff < tol:
            convergence = True

        steps += 1

    # print(steps)
    # print(diff)

    if steps == max_iter:
        return

    return beta


def beta_fixed_point(degrees, k, sets, max_iter=500, tol=0.0001, beta=None):
    """Use a fixed point algorithm to calculate the MLE."""
    n = len(degrees)
    if beta is None:
        beta = np.zeros(n)

    convergence = False
    steps = 0

    # There are more efficient methods for calculating e^{beta_S} for each S in
    # n\choose k-1, e.g using a dynamic algorithm
    # sets = np.asarray(list(itertools.combinations(range(n), k - 1)))
    prod_beta = np.ones(len(sets))

    while(not convergence and steps < max_iter):
        exp_beta = np.exp(beta)
        old_beta = beta.copy()
        if np.any(np.isinf(old_beta)):
            return None

        for i, s in enumerate(sets):
            prod_beta[i] = np.prod(exp_beta[np.asarray(s)])

        if np.any(np.isinf(prod_beta)):
            print("Infinite beta")
            return

        for i in range(n):
            ind = np.array([j for j in range(len(sets)) if i not in sets[j]])
            sum_q = np.sum(prod_beta[ind] / (1 + prod_beta[ind] * exp_beta[i]))
            if np.isinf(sum_q):
                print("Infinite beta")
                return

            beta[i] = np.log(degrees[i]) - np.log(sum_q)

        diff = np.max(np.abs(old_beta - beta))
        # print(f"diff= {diff} -------- steps= {steps}")
        # print(beta)
        if diff < tol:
            convergence = True

        steps += 1

    # print(steps)
    # print(diff)

    if steps == max_iter:
        return

    return beta


def fixed_point_general_R(degrees, k_list, max_iter=500, tol=0.0001,
                          beta=None):
    """Use a fixed point algorithm to get the MLE."""
    n = len(degrees)
    if beta is None:
        beta = np.zeros(n)

    convergence = False
    steps = 0

    all_index_sets = list()
    index_k = 0
    prod_exp_beta_list = list()
    for k in k_list:
        index_k += 1
        sets = list(itertools.combinations(range(n), k - 1))
        all_index_sets.append(sets)
        prod_exp_beta_list.append(np.ones(len(sets)))

    while (not convergence and steps < max_iter):
        exp_beta = np.exp(beta)
        old_beta = beta.copy()
        if any(np.isinf(old_beta)):
            print("Infinite beta")
            return

        for index_k in range(len(k_list)):
            sets = all_index_sets[index_k]
            prod_exp_beta = prod_exp_beta_list[index_k]
            for t, s in enumerate(sets):
                prod_exp_beta[t] = np.prod([exp_beta[tt] for tt in s])
                if np.isinf(prod_exp_beta[t]):
                    print("Infinite beta")
                    return

            prod_exp_beta_list[index_k] = prod_exp_beta

        for i in range(n):
            sum_q_beta = list(range(len(k_list)))
            for index_k in range(len(k_list)):
                sets = all_index_sets[index_k]
                prod_exp_beta = prod_exp_beta_list[index_k]
                for j in range(len(sets)):
                    if i not in sets[j]:
                        sum_q_beta[index_k] += (
                            prod_exp_beta[j]
                            / (1 + prod_exp_beta[j] * exp_beta[i])
                        )
                        if np.isinf(sum_q_beta[index_k]):
                            print("Infinite beta")
                            return
            beta[i] = np.log(degrees[i]) - np.log(np.sum(sum_q_beta))

        diff = max(abs(old_beta - beta))
        # print(f"diff= {diff} -------- steps= {steps}")
        # print(beta)
        if diff < tol:
            convergence = True

        steps += 1
    # print(steps)
    # print(diff)
    if steps == max_iter:
        print("Max iterations reached")
        return

    return beta


def fixed_point_general(degrees, k_list, max_iter=500, tol=0.0001, beta=None):
    """Use a fixed point algorithm to get the MLE."""
    n = len(degrees)
    if beta is None:
        beta = np.zeros(n)

    convergence = False
    steps = 0

    all_index_sets = list()
    index_k = 0
    prod_exp_beta_list = list()
    for k in k_list:
        index_k += 1
        sets = list(itertools.combinations(range(n), k - 1))
        all_index_sets.append(sets)
        prod_exp_beta_list.append(np.ones(len(sets)))

    while (not convergence and steps < max_iter):
        exp_beta = np.exp(beta)
        old_beta = beta.copy()
        if any(np.isinf(old_beta)):
            print("Infinite beta")
            return

        for index_k in range(len(k_list)):
            sets = all_index_sets[index_k]
            prod_exp_beta = prod_exp_beta_list[index_k]
            for t, s in enumerate(sets):
                prod_exp_beta[t] = np.prod([exp_beta[tt] for tt in s])
                if np.isinf(prod_exp_beta[t]):
                    print("Infinite beta")
                    return

            prod_exp_beta_list[index_k] = prod_exp_beta

        for i in range(n):
            sum_q_beta = list(range(len(k_list)))
            for index_k in range(len(k_list)):
                sets = all_index_sets[index_k]
                prod_exp_beta = prod_exp_beta_list[index_k]
                for j in range(len(sets)):
                    if i not in sets[j]:
                        sum_q_beta[index_k] += (
                            prod_exp_beta[j]
                            / (1 + prod_exp_beta[j] * exp_beta[i])
                        )
                        if np.isinf(sum_q_beta[index_k]):
                            print("Infinite beta")
                            return
            beta[i] = np.log(degrees[i]) - np.log(np.sum(sum_q_beta))

        diff = max(abs(old_beta - beta))
        # print(f"diff= {diff} -------- steps= {steps}")
        # print(beta)
        if diff < tol:
            convergence = True

        steps += 1
    # print(steps)
    # print(diff)
    if steps == max_iter:
        print("Max iterations reached")
        return

    return beta


def main():
    # for correctness
    k = 3
    K53 = list(itertools.combinations(range(5), k))
    degs = deg_seq(K53)
    sets = list(itertools.combinations(range(len(degs)), k - 1))

    beta_K53 = beta_fixed_point(degs, k=k, sets=sets, max_iter=10000)
    print(np.isclose(beta_K53, 3.07028833 * np.ones(5)))

    # for performance
    n = 25
    Kn53 = list(itertools.combinations(range(n), 3))
    degs = deg_seq(Kn53)
    sets = list(itertools.combinations(range(len(degs)), k - 1))

    print(f"Running R-converted code (with n={n})")
    tic = time.perf_counter()
    beta_fixed_point_R(degs, k=k, sets=sets, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point_R took {toc - tic:0.4f} seconds")

    # print(beta_Kn53)

    print(f"Running python vectorized code (with n={n})")
    tic = time.perf_counter()
    beta_fixed_point(degs, k=3, sets=sets, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point took {toc - tic:0.4f} seconds")

    njit_fp = nb.njit(beta_fixed_point)
    njit_sets = np.asarray(sets)

    print(f"Running python jit'd code (with n={n})")
    tic = time.perf_counter()
    njit_fp(degs, k=3, sets=njit_sets, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point jit'd took {toc - tic:0.4f} seconds")

    # d10_3 = (36, 36, 36, 36, 36, 36, 36, 36, 36, 36)

    # beta = fixed_point_general(d10_3, [3, ], max_iter=10000)
    # print(beta)

    # [3.07028833 3.07028833 3.07028833 3.07028833 3.07028833 3.07028833
    #  3.07028833 3.07028833 3.07028833 3.07028833]
    # 3331
    # 9.997253637461512e-05


if __name__ == "__main__":
    main()
