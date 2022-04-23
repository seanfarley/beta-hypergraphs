#!/usr/bin/env python3

import itertools
import time

import cupy as cp
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


@cp.fuse
def bfp_prodadd(exp_beta, ind3, iind3):
    return exp_beta * ind3 + iind3


bfp_prodadd_k = cp.ReductionKernel(
    'T exp_beta, T ind3, T iind3',
    'T z',
    'exp_beta * ind3 + iind3',
    'a * b',
    'z = a',
    '1',
    'bfp_prodadd_k'
)


@cp.fuse
def bfp_prodexp(prod_beta, mask, exp_beta):
    return (prod_beta * mask) / (1 + (prod_beta * mask * exp_beta))


bfp_prodexp_k = cp.ReductionKernel(
    'T prod_beta, T mask, T exp_beta',
    'T z',
    '(prod_beta * mask) / (1 + (prod_beta * mask * exp_beta))',
    'a + b',
    'z = log(a)',
    '0',
    'bfp_prodexp_k'
)


@cp.fuse
def logdiff(ldegs, sum_q3):
    return ldegs - cp.log(sum_q3)


@cp.fuse
def maxabs(old_beta, beta):
    return cp.max(cp.abs(old_beta - beta))


def beta_fixed_point(degrees, k, sets, ind, max_iter=500, tol=0.0001, beta=None):
    """Use a fixed point algorithm to calculate the MLE."""
    n = len(degrees)
    if beta is None:
        beta = cp.zeros(n)

    convergence = False
    steps = 0

    # There are more efficient methods for calculating e^{beta_S} for each S in
    # n\choose k-1, e.g using a dynamic algorithm

    ldegs = cp.log(degrees)

    while(not convergence and steps < max_iter):
        exp_beta = cp.exp(beta)
        old_beta = beta.copy()
        # if cp.any(np.isinf(old_beta)):
        #     return None

        prod_beta = cp.prod(exp_beta[sets], axis=1)

        sum_q = cp.sum(prod_beta[ind]
                       / (1 + (prod_beta[ind]
                               * exp_beta.reshape(-1, 1))),
                       axis=1)

        beta = logdiff(ldegs, sum_q)
        diff = maxabs(old_beta, beta)
        # diff = cp.max(cp.abs(old_beta - beta))
        if diff < tol:
            convergence = True

        steps += 1

    # print(steps)
    # print(diff)

    if steps == max_iter:
        return

    return beta


def beta_fixed_point2(degrees, k, ind2, ind3, iind3, max_iter=500, tol=0.0001, beta=None):
    """Alternative using less if-statements at the cost of more memory."""
    n = len(degrees)
    if beta is None:
        beta = cp.zeros(n)

    convergence = False
    steps = 0

    # There are more efficient methods for calculating e^{beta_S} for each S in
    # n\choose k-1, e.g using a dynamic algorithm

    ldegs = cp.log(degrees)

    while(not convergence and steps < max_iter):
        exp_beta = cp.exp(beta)
        old_beta = beta.copy()
        # if cp.any(np.isinf(old_beta)):
        #     return None

        pb = bfp_prodadd(exp_beta, ind3, iind3)
        prod_beta = cp.prod(pb, axis=1)

        pe = bfp_prodexp(prod_beta, ind2, exp_beta.reshape(-1, 1))
        sum_q = cp.sum(pe, axis=1)

        beta = logdiff(ldegs, sum_q)
        diff = maxabs(old_beta, beta)
        # diff = cp.max(cp.abs(old_beta - beta))
        if diff < tol:
            convergence = True

        steps += 1

    # print(steps)
    # print(diff)

    if steps == max_iter:
        return

    return beta


def beta_fixed_point3(degrees, k, ind2, ind3, iind3, max_iter=500, tol=0.0001, beta=None):
    """Alternative using less if-statements at the cost of more memory."""
    n = len(degrees)
    if beta is None:
        beta = cp.zeros(n)

    convergence = False
    steps = 0

    # There are more efficient methods for calculating e^{beta_S} for each S in
    # n\choose k-1, e.g using a dynamic algorithm

    ldegs = cp.log(degrees)

    exp_beta = cp.exp(beta)
    old_beta = beta.copy()

    while(not convergence and steps < max_iter):
        # exp_beta = cp.exp(beta)
        # old_beta = beta.copy()
        # if cp.any(np.isinf(old_beta)):
        #     return None

        prod_beta = bfp_prodadd_k(exp_beta, ind3, iind3, axis=1)

        beta = ldegs - bfp_prodexp_k(prod_beta, ind2, exp_beta.reshape(-1, 1), axis=1)

        diff = maxabs(old_beta, beta)
        if diff < tol:
            convergence = True

        old_beta = beta
        exp_beta = cp.exp(beta)
        steps += 1

    # print(steps)
    # print(diff)

    if steps == max_iter:
        return

    return beta


def main():
    # for correctness
    n = 5
    k = 3
    K53 = list(itertools.combinations(range(n), k))
    degs = deg_seq(K53)
    sets = list(itertools.combinations(range(len(degs)), k - 1))

    sn = len(sets)
    an = np.arange(n)
    asn = np.arange(sn)
    ind = np.array([[j for j in asn if i not in sets[j]] for i in an])

    degs_cp = cp.array(degs)
    sets_cp = cp.array(sets)
    ind_cp = cp.array(ind)

    print(f"Running cupy vectorized code (with n={n}, k={k})")
    tic = time.perf_counter()
    beta_K53 = beta_fixed_point(degs_cp, k, sets_cp, ind_cp, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point took {toc - tic:0.4f} seconds")
    print()

    all_passed = cp.allclose(beta_K53, 3.07028833 * cp.ones(5))
    print(f"Testing correctness for n={n}, k={k}; passing={all_passed}")
    print()

    ind2 = np.array([[i not in sets[j] for j in asn] for i in an])
    iind3 = np.transpose(ind2)
    ind3 = ~iind3

    ind2 = cp.array(ind2, dtype=float)
    ind3 = cp.array(ind3, dtype=float)
    iind3 = cp.array(iind3, dtype=float)

    print(f"Running cupy second vectorized code (with n={n}, k={k})")
    tic = time.perf_counter()
    beta_K53 = beta_fixed_point2(degs_cp, k, ind2, ind3, iind3, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point2 took {toc - tic:0.4f} seconds")
    print()

    all_passed = cp.allclose(beta_K53, 3.07028833 * cp.ones(5))
    print(f"Testing correctness for n={n}, k={k}; passing={all_passed}")
    print()

    print(f"Running python third vectorized code (with n={n}, k={k})")
    tic = time.perf_counter()
    beta_K53 = beta_fixed_point3(degs_cp, k, ind2, ind3, iind3, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point3 took {toc - tic:0.4f} seconds")
    print()

    all_passed = cp.allclose(beta_K53, 3.07028833 * cp.ones(5))
    print(f"Testing correctness for n={n}, k={k}; passing={all_passed}")
    print()

    # for performance
    n = 25
    k = 4
    Kn53 = list(itertools.combinations(range(n), k))
    degs = deg_seq(Kn53)
    sets = list(itertools.combinations(range(len(degs)), k - 1))

    sn = len(sets)
    an = np.arange(n)
    asn = np.arange(sn)
    ind = np.array([[j for j in asn if i not in sets[j]] for i in an])

    degs_cp = cp.array(degs)
    sets_cp = cp.array(sets)
    ind_cp = cp.array(ind)

    print(f"Running python vectorized code (with n={n}, k={k})")
    tic = time.perf_counter()
    beta_fixed_point(degs_cp, k, sets_cp, ind_cp, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point took {toc - tic:0.4f} seconds")
    print()

    ind2 = np.array([[i not in sets[j] for j in asn] for i in an])
    iind3 = np.transpose(ind2)
    ind3 = ~iind3

    ind2 = cp.array(ind2, dtype=float)
    ind3 = cp.array(ind3, dtype=float)
    iind3 = cp.array(iind3, dtype=float)

    print(f"Running python second vectorized code (with n={n}, k={k})")
    tic = time.perf_counter()
    beta_fixed_point2(degs_cp, k, ind2, ind3, iind3, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point2 took {toc - tic:0.4f} seconds")
    print()

    print(f"Running python third vectorized code (with n={n}, k={k})")
    tic = time.perf_counter()
    beta_K53 = beta_fixed_point3(degs_cp, k, ind2, ind3, iind3, max_iter=10000)
    toc = time.perf_counter()
    print(f"beta_fixed_point3 took {toc - tic:0.4f} seconds")

    # print(f"Proof of slow memory testing on the gpu (with n={n}, k={k})")
    # tic = time.perf_counter()
    # ind2 = cp.array([[i not in sets_cp[j] for j in asn] for i in an])
    # iind3 = cp.transpose(ind2)
    # ind3 = ~iind3
    # toc = time.perf_counter()
    # print(f"Memory testing (e.g. `if i in set`) took {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()
