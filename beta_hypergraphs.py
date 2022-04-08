#!/usr/bin/env python3

import itertools
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


def beta_fixed_point(degrees, k, max_iter=500, tol=0.0001, beta=None):
    """Use a fixed point algorithm to calculate the MLE."""
    n = len(degrees)
    if beta is None:
        beta = np.zeros(n)

    print(np.exp(beta))
    convergence = False
    steps = 0

    # There are more efficient methods for calculating e^{beta_S} for each S in
    # n\choose k-1, e.g using a dynamic algorithm
    sets = list(itertools.combinations(range(n), k - 1))
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
        print(f"diff= {diff} -------- steps= {steps}")
        print(beta)
        if diff < tol:
            convergence = True

        steps += 1

    print(steps)
    print(diff)

    if steps == max_iter:
        return

    return beta


def main():
    K53 = list(itertools.combinations([1, 2, 3, 4, 5], 3))
    print(deg_seq(K53))
    fp = beta_fixed_point(deg_seq(K53), k=3, max_iter=10000)
    print(np.isclose(fp, 3.07028833 * np.ones(5)))


if __name__ == "__main__":
    main()
