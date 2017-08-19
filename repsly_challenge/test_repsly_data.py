from unittest import TestCase
from repsly_data import RepslyData

import numpy as np

class TestRepslyData(TestCase):
    def setUp(self):
        self.file_name = 'data/trial_users_analysis.csv'
        self.two_users_file = 'data/trial_two_users.csv'
        self.ten_users_file = 'data/trial_ten_users.csv'

    def _test_prepare_data(self, file_name, expected_data_shapes):
        repsly_data = RepslyData()

        # make the call
        X_all, y_all = repsly_data._prepare_data(file_name=file_name)

        np.testing.assert_array_equal((X_all.shape, y_all.shape), expected_data_shapes)

    def test_prepare_data_fast(self):
        test_vectors = {
            self.two_users_file: ((2, 1+15*16), (2, )),
            self.ten_users_file: ((10, 1 + 15 * 16), (10,))
        }
        for file_name in test_vectors.keys():
            self._test_prepare_data(file_name, expected_data_shapes=test_vectors[file_name])

    def test_prepare_data_slow(self):
        test_vectors = {
            self.file_name: ((6466, 1+15*16), (6466, ))
        }
        for file_name in test_vectors.keys():
            self._test_prepare_data(file_name, expected_data_shapes=test_vectors[file_name])
