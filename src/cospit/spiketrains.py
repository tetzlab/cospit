#!/usr/bin/env python3
"""Functions for generating Poisson spike trains and some statistical analysis."""

import numpy as np


def generate_poisson_spike_train(rate, duration, rng):
    """Returns a Poisson spike train.

    rate -- the rate of the spike train
    duration -- the duration of the spike train
    rng -- random number generator
    """

    n_events = np.round(rate * duration)

    if n_events:
        size = rng.poisson(n_events)
        return np.sort(np.random.uniform(0.0, duration, size))

    return np.array([])


def sample_spike_train(probability, source_spike_train):
    """Samples spike train.

    Draws samples from source_spike_train with the same given probability for each spike.

    probability -- the probability to draw each spike
    source_spike_train - the source spike train
    """

    source_spike_train = np.asarray(source_spike_train)

    to_keep = np.random.uniform(size=len(source_spike_train)) < probability
    return source_spike_train[to_keep]


def mix_spike_trains(probabilities, source_spike_trains):
    """Mixes spike trains to one target spike train.

    Mixes spike trains with given probabilities.

    probabilities -- sequence (`list`, for example) of probabilities
    source_spike_trains -- sequence (`list`, for example) of spike trains
    """

    samples = [
        sample_spike_train(
            p, s) for p, s in zip(probabilities, source_spike_trains)]
    return np.sort(np.concatenate(samples))


def binary_vector_from_spike_train(spike_train, duration, bin_width):
    """Transforms a spike train into a binary vector.

    spike_train -- spike train
    duration -- assumed duration of spike train (can be smaller or larger than last spike time)
    """

    n_bins = int(np.round(duration / bin_width + 1))
    bins = np.linspace(0, duration, n_bins)

    return np.histogram(spike_train, bins)[0] > 0


def pearson_correlation_coefficient(binary_vector_1, binary_vector_2):
    """Calculates Pearson correlation coefficient between binary vectors.

    binary_vector_1 -- sequence (`list`, for example) of bools
    binary_vector_2 -- sequence (`list`, for example) of bools
    """

    binary_vector_1 = np.asarray(binary_vector_1)
    binary_vector_2 = np.asarray(binary_vector_2)

    bv_minus_avg_1 = binary_vector_1 - np.mean(binary_vector_1)
    bv_minus_avg_2 = binary_vector_2 - np.mean(binary_vector_2)

    pcc = np.dot(bv_minus_avg_1, bv_minus_avg_2)
    pcc /= np.sqrt(np.dot(bv_minus_avg_1, bv_minus_avg_1)
                   * np.dot(bv_minus_avg_2, bv_minus_avg_2))

    return pcc


def calculate_pearson_correlations(spike_trains, duration, bin_width):
    """Calculates pearson correlation coefficients between spike trains.

    Assumes same duration for all spike trains.

    spike_trains -- sequence (`list`, for example) of spike trains
    duration -- assumed duration of spike trains
    bin_width -- bin width (time) for conversion to binary vector

    Returns Pearson correlation coefficients in the order of numpy's `triu_indices`.
    """

    binary_vectors = [binary_vector_from_spike_train(
        train, duration, bin_width) for train in spike_trains]

    # upper triangle indices without diagonal
    upper_triangle_indices = np.triu_indices(len(spike_trains), 1)
    return [pearson_correlation_coefficient(
        binary_vectors[i], binary_vectors[j]) for i, j in zip(*upper_triangle_indices)]
