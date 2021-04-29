#!/usr/bin/env python3
"""Implementation of Brette 2009's mixture method for correlated spike trains.
"""

import numpy as np

from .spiketrains import generate_poisson_spike_train, mix_spike_trains
from .brette import total_correlation_from_pearson_correlation_coefficent, mixture_method


def generate_correlated_spike_trains(
        target_rates, pccs, duration, rng):
    """Generates correlated spike trains based on Brette 2009 (https://doi.org/10.1162/neco.2009.12-07-657).

    target_rates -- sequence (`list`, for example) of the target rates of the spike traines.
    pccs -- sequence (`list`, for example) of the Pearson correlation coefficients
           in the order of numpy's `triu_indices`.
    duration -- duration of the generated spike trains.
    rng -- numpy random number generator.

    Returns generated correlated spike trains.
    """

    if np.triu_indices(len(target_rates), 1)[0].size != len(pccs):
        raise RuntimeError("wrong number of Pearson correlation coefficients")

    brette_correlations = np.zeros((len(target_rates), len(target_rates)))
    upper_triangle_indices = np.triu_indices_from(brette_correlations, 1)
    index_pairs = zip(*upper_triangle_indices)

    # convert Pearson correlation coefficient to Brette's total correlations
    brette_correlations[upper_triangle_indices] = \
        [total_correlation_from_pearson_correlation_coefficent(
            target_rates[i], target_rates[j], pccs[n]) for n,
         (i, j) in enumerate(index_pairs)]
    # construct symmetrical matrix
    brette_correlations = np.maximum(
        brette_correlations, brette_correlations.T)

    # get mixtures probabilities and source rates
    mixture_propabilities, source_rates = mixture_method(
        target_rates, brette_correlations)

    # generate source spike trains
    source_spike_trains = [
        generate_poisson_spike_train(
            rate, duration, rng) for rate in source_rates]

    # mix target spike trains from source spike trains
    target_spike_trains = [
        mix_spike_trains(
            p_row,
            source_spike_trains) for p_row in mixture_propabilities]

    return target_spike_trains
