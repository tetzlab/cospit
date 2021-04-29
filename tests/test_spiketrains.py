#!/usr/bin/env python3
"""Tests spike train functions."""

import unittest
import numpy as np
import cospit


class TestSpikeTrains(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng()

    def test_zero_rate(self):
        spike_train = cospit.spiketrains.generate_poisson_spike_train(
            0, 1000, self.rng)
        self.assertEqual(spike_train.size, 0)

    def test_zero_duration(self):
        spike_train = cospit.spiketrains.generate_poisson_spike_train(
            1000, 0, self.rng)
        self.assertEqual(spike_train.size, 0)

    def test_sample_all(self):
        spike_train = [1, 2, 3]
        self.assertTrue(
            np.array_equal(
                cospit.sample_spike_train(
                    1,
                    spike_train),
                spike_train))

    def test_sample_none(self):
        spike_train = [1, 2, 3]
        self.assertEqual(cospit.sample_spike_train(0, spike_train).size, 0)

    def test_binary_vector_from_spike_train(self):
        spike_train = [0.2, 0.3, 1, 2]
        self.assertTrue(
            np.array_equal(cospit.binary_vector_from_spike_train(spike_train, 2, 0.5),
                           [True, False, True, True]))

    def test_pearson_correlation_coefficient_full_correlation(self):

        binary_vector_1 = [True, False, True, True]

        self.assertEqual(
            cospit.pearson_correlation_coefficient(
                binary_vector_1, binary_vector_1),
            1)

    def test_pearson_correlation_coefficient_full_anti_correlation(self):

        binary_vector_1 = [True, False, True, True]
        binary_vector_2 = np.invert(binary_vector_1)

        self.assertEqual(
            cospit.pearson_correlation_coefficient(
                binary_vector_1, binary_vector_2),
            -1)

    def test_calculate_pearson_correlations(self):

        spike_train_1 = [0.2, 0.3, 1, 2]
        spike_train_2 = [0.7]
        spike_train_3 = np.copy(spike_train_1)

        spike_trains = [spike_train_1, spike_train_2, spike_train_3]

        self.assertEqual(
            cospit.calculate_pearson_correlations(spike_trains, 2, 0.5),
            [-1, 1, -1])

    def test_mix_spike_trains_full(self):

        spike_train_1 = [0.2, 0.3, 1, 2]
        spike_train_2 = [0.7, 1.5]

        self.assertTrue(np.array_equal(
            cospit.mix_spike_trains([1, 1], [spike_train_1, spike_train_2]),
            [0.2, 0.3, 0.7, 1, 1.5, 2]))

    def test_mix_spike_trains_only_one(self):

        spike_train_1 = [0.2, 0.3, 1, 2]
        spike_train_2 = [0.7, 1.5]

        self.assertTrue(np.array_equal(
            cospit.mix_spike_trains([1, 0], [spike_train_1, spike_train_2]),
            spike_train_1))

        self.assertTrue(np.array_equal(
            cospit.mix_spike_trains([0, 1], [spike_train_1, spike_train_2]),
            spike_train_2))


if __name__ == '__main__':
    unittest.main()
