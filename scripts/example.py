#!/usr/bin/env python3
"""Demonstrates how to use cospit to generate correlated spike trains.
"""


import numpy as np
import cospit

if __name__ == "__main__":

    # draw random target rates and correlations
    rng = np.random.default_rng(42)
    MIN_RANDOM_RATE = 50
    MAX_RANDOM_RATE = 200
    N_TARGETS = 5
    target_rates = rng.integers(
        MIN_RANDOM_RATE,
        MAX_RANDOM_RATE,
        size=N_TARGETS).astype(float)

    MIN_CORRELATION = 0
    MAX_CORRELATION = 0.3
    pearson_correlation_coefficients = rng.uniform(
        MIN_CORRELATION, MAX_CORRELATION, size=np.triu_indices(N_TARGETS, 1)[0].size)

    # generate spike trains with targeted rates and correlations
    DURATION = 1000
    target_spike_trains = cospit.generate_correlated_spike_trains(
        target_rates, pearson_correlation_coefficients, DURATION, rng)

    # calculate resulting correlations
    BIN_WIDTH = 0.001
    generated_correlations = np.array(
        cospit.calculate_pearson_correlations(
            target_spike_trains,
            DURATION,
            BIN_WIDTH))

    # calculate resulting rates
    generated_rates = np.array([np.round(len(t) / DURATION)
                                for t in target_spike_trains])

    np.set_printoptions(precision=3)

    print(
        f"Target Pearson correlation coefficients: \t{pearson_correlation_coefficients}")
    print(f"Generated correlations: \t\t\t{generated_correlations}")

    print(f"Target spike train rates: \t{target_rates}")
    print(f"Generated rates (rounded): \t{generated_rates}")
