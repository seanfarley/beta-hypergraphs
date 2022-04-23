#!/usr/bin/env python3

import itertools
import time

import numba as nb
import numpy as np

from numba.typed import List


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
        if np.any(np.isinf(old_beta)):
            return None

        for i in range(len(sets)):
            prod_beta[i] = np.prod(np.array([exp_beta[i] for i in sets[i]]))

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


def beta_fixed_point(degrees, k, sets, max_iter=500, tol=0.0001, beta=None):
    """Use a fixed point algorithm to calculate the MLE."""
    n = len(degrees)
    sn = len(sets)
    an = np.arange(n)
    asn = np.arange(sn)
    if beta is None:
        beta = np.zeros(n)

    convergence = False
    steps = 0

    # There are more efficient methods for calculating e^{beta_S} for each S in
    # n\choose k-1, e.g using a dynamic algorithm
    # sets = np.asarray(list(itertools.combinations(range(n), k - 1)))

    ind = np.array([[j for j in asn if i not in sets[j]] for i in an])
    sets = np.asarray(sets)

    while(not convergence and steps < max_iter):
        exp_beta = np.exp(beta)
        old_beta = beta.copy()
        if np.any(np.isinf(old_beta)):
            return None

        # for i, s in enumerate(sets):
        #     prod_beta[i] = np.prod(exp_beta[np.asarray(s)])
        prod_beta = np.prod(exp_beta[sets], axis=1)

        if np.any(np.isinf(prod_beta)):
            print("Infinite beta")
            return

        # for i in range(n):
        #     ind = np.array([j for j in range(len(sets)) if i not in sets[j]])
        #     sum_q = np.sum(prod_beta[ind]
        #                    / (1 + prod_beta[ind] * exp_beta[i]))
        #     if np.isinf(sum_q):
        #         print("Infinite beta")

        sum_q = np.sum(prod_beta[ind] / (1 + (prod_beta[ind].T * exp_beta).T),
                       axis=1)

        if np.any(np.isinf(sum_q)):
            print("Infinite beta")
            return

        beta = np.log(degrees) - np.log(sum_q)

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


def fixed_point_general_R(degrees, k_list, all_index_sets, max_iter=500,
                          tol=0.0001, beta=None):
    """Use a fixed point algorithm to get the MLE."""
    n = len(degrees)
    if beta is None:
        beta = np.zeros(n)

    convergence = False
    steps = 0

    prod_exp_beta_list = List()
    for s in all_index_sets:
        prod_exp_beta_list.append(np.ones(len(s)))

    while (not convergence and steps < max_iter):
        exp_beta = np.exp(beta)
        old_beta = beta.copy()
        if np.any(np.isinf(old_beta)):
            print("Infinite beta")
            return

        for index_k in range(len(k_list)):
            sets = all_index_sets[index_k]
            prod_exp_beta = prod_exp_beta_list[index_k]
            for t, s in enumerate(sets):
                eba = np.array([exp_beta[tt] for tt in s])
                prod_exp_beta[t] = np.prod(eba)
                if np.isinf(prod_exp_beta[t]):
                    print("Infinite beta")
                    return

            prod_exp_beta_list[index_k] = prod_exp_beta

        for i in range(n):
            sum_q_beta = np.zeros(len(k_list))
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

        diff = np.max(np.abs(old_beta - beta))
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


def fixed_point_general(degrees, k_list, all_index_sets, max_iter=500,
                        tol=0.0001, beta=None):
    """Use a fixed point algorithm to get the MLE."""
    n = len(degrees)
    if beta is None:
        beta = np.zeros(n)

    convergence = False
    steps = 0

    prod_exp_beta_list = list()
    for s in all_index_sets:
        prod_exp_beta_list.append(np.ones(len(s)))

    ns = np.asarray(all_index_sets)

    k_ind = range(len(k_list))
    ind = np.array([[[j for j in range(len(all_index_sets[k]))
                      if i not in all_index_sets[k][j]]]
                    for k in range(len(k_list))
                    for i in range(n)])

    while (not convergence and steps < max_iter):
        exp_beta = np.exp(beta)
        old_beta = beta.copy()
        if np.any(np.isinf(old_beta)):
            print("Infinite beta")
            return

        prod_exp_beta_list = np.prod(exp_beta[ns], axis=2)
        # for i in range(n):
        #     k_ind = range(len(k_list))
        #     ind = list(k_ind)

        #     # creates an "index" array
        #     for index_k in k_ind:
        #         iset = all_index_sets[index_k]
        #         ind[index_k] = np.array([j for j in range(len(iset))
        #                                  if i not in iset[j]])

        #     sum_q_beta = np.sum(
        #         prod_exp_beta_list[k_ind, ind]
        #         / (1 + prod_exp_beta_list[k_ind, ind] * exp_beta[i])
        #     )
        #     beta[i] = np.log(degrees[i]) - np.log(sum_q_beta)

        sum_q_beta = np.sum(prod_exp_beta_list[k_ind, ind]
                            / (1 + prod_exp_beta_list[k_ind, ind].T
                               * exp_beta).T, axis=2)
        # TODO don't know why it's a list of lists
        sum_q_beta = sum_q_beta[:, 0]
        beta = np.log(degrees) - np.log(sum_q_beta)

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
    n = 5
    k = 3
    K53 = list(itertools.combinations(range(n), k))
    degs = deg_seq(K53)
    sets = List(itertools.combinations(range(len(degs)), k - 1))
    fp_njit = nb.njit(beta_fixed_point_R)
    fpg_njit = nb.njit(fixed_point_general_R)

    beta_K53 = beta_fixed_point(degs, k=k, sets=sets, max_iter=10000)
    all_passed = np.allclose(beta_K53, 3.07028833 * np.ones(5))
    print(f"Testing correctness for n={n}, k={k}; passing={all_passed}")
    print()

    print(f"Precompiling python jit'd code (with n={n}, k={k})")
    tic = time.perf_counter()
    fp_njit(degs, k=k, sets=sets, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point jit'd took {toc - tic:0.4f} seconds")
    print()

    # for performance
    n = 25
    k = 3
    Knk = list(itertools.combinations(range(n), k))
    degs = deg_seq(Knk)
    sets = List(itertools.combinations(range(len(degs)), k - 1))

    print(f"Running R-converted code (with n={n}, k={k})")
    tic = time.perf_counter()
    beta_fixed_point_R(degs, k=k, sets=sets, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point_R took {toc - tic:0.4f} seconds")
    print()

    # print(beta_Kn3)

    print(f"Running python vectorized code (with n={n}, k={k})")
    tic = time.perf_counter()
    beta_fixed_point(degs, k=k, sets=sets, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point took {toc - tic:0.4f} seconds")
    print()

    print(f"Running python jit'd code (with n={n}, k={k})")
    tic = time.perf_counter()
    fp_njit(degs, k=k, sets=sets, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point_R jit'd took {toc - tic:0.4f} seconds")
    # print()

    # d10_3 = (36, 36, 36, 36, 36)
    # n = len(d10_3)
    # k_list = List([k, ])
    # all_index_sets = List()
    # for k in k_list:
    #     sets = List(itertools.combinations(range(n), k - 1))
    #     all_index_sets.append(sets)

    # print(f"Precompiling python jit'd code (with n={n}, k={k_list})")
    # tic = time.perf_counter()
    # fpg_njit(d10_3, k_list, all_index_sets, max_iter=10000)
    # toc = time.perf_counter()
    # print(f"fixed_point_general jit'd took {toc - tic:0.4f} seconds")
    # print()

    # d10_3 = (36, 36, 36, 36, 36, 36, 36, 36, 36, 36)
    # n = len(d10_3)
    # k_list = List([k, ])
    # all_index_sets = List()
    # for k in k_list:
    #     sets = List(itertools.combinations(range(n), k - 1))
    #     all_index_sets.append(sets)

    # print(f"Running python R-converted code (with n={n}, k={k_list})")
    # tic = time.perf_counter()
    # fixed_point_general_R(d10_3, List(k_list), all_index_sets, max_iter=10000)
    # toc = time.perf_counter()
    # print(f"fixed_point_general_R took {toc - tic:0.4f} seconds")
    # print()

    # print(f"Running python vectorized code (with n={n}, k={k_list})")
    # tic = time.perf_counter()
    # fixed_point_general(d10_3, k_list, all_index_sets, max_iter=10000)
    # toc = time.perf_counter()
    # print(f"fixed_point_general took {toc - tic:0.4f} seconds")
    # print()

    # # [3.07028833 3.07028833 3.07028833 3.07028833 3.07028833 3.07028833
    # #  3.07028833 3.07028833 3.07028833 3.07028833]
    # # 3331
    # # 9.997253637461512e-05

    # print(f"Running python jit'd code (with n={n}, k={k_list})")
    # tic = time.perf_counter()
    # fpg_njit(d10_3, k_list, all_index_sets, max_iter=10000)
    # toc = time.perf_counter()
    # print(f"fixed_point_general_R jit'd took {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()
