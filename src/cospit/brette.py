#!/usr/bin/env python3
"""Implementation of the mixture method for correlated spike trains described in
"Generation of correlated spike train" Brette 2009."""

import numpy as np


def mixture_method(
        target_rates, correlation_coefficients,
        learning_rate_scale=0.0001, training_steps=20000):
    """Implementation of the mixture method for correlated spike trains described in
    "Generation of correlated spike train" Brette 2009.

    target_rates -- target rates of correlated spike trains
    correlation_coefficients -- total correlation coefficients (Brette definition)
    learning_rate_scale -- scales the learning rate of both gradients
                           (frequencies and probabilities)

    Returns mixing probabilities and rates of source spike trains.
    """

    # we want to be close to the variable names of the publication, so we
    # ignore the snake_case naming style error. We also ignore too-many-local.
    # pylint: disable=C0103,R0914

    R = target_rates
    C = correlation_coefficients

    n_targets = len(R)

    # learning rates
    b = learning_rate_scale / n_targets
    a = (1. / n_targets) * b

    # source rates
    nu = np.copy(R)

    # mixing probabilities
    P = np.identity(n_targets)

    for _ in range(training_steps):

        X = np.einsum("ik,jk,k->ij", P, P, nu)
        A = X - C
        np.fill_diagonal(A, 0)

        # same as: 4 * np.einsum("ik,kj,j->ij", A, P, nu)
        dPE = 4 * A @ P @ np.diag(nu)

        # same as: 2 * np.einsum("ki,li,kl->i", P, P, A)
        dnuE = 2 * np.diag(P.T @ A @ P)

        U = np.heaviside(P @ nu - R, 0)
        dPF = np.outer(U, nu)

        # same as: np.einsum("j,ji->i", U, P)
        dnuF = np.dot(U, P)

        # perform update
        dnu = a * dnuE + b * dnuF
        dP = a * dPE + b * dPF
        nu -= dnu
        P -= dP

        # clip to [0, 1] (it's a probability)
        P.clip(0, 1, out=P)

        # clip to positive rates
        nu.clip(0, out=nu)

    # complete rates and probabilities
    nu = np.concatenate([nu, R - P @ nu])

    P = np.block([[P, np.identity(n_targets)]])

    if np.any(nu < 0):
        raise RuntimeError(
            "optimization failed, not all rates positive: {}".format(nu))

    return P, nu


def total_correlation_from_pearson_correlation_coefficent(rate_1, rate_2, pcc):
    """Returns total correlation coefficient (Brette) from Pearson correlation coefficient.

    rate_1 -- rate of first spike train
    rate_2 -- rate of second spike train
    pcc -- Pearson correlation coefficient
    """

    return pcc * np.sqrt(rate_1 * rate_2)
