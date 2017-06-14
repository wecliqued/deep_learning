from unittest import TestCase
from repsly_data import RepslyData

import csv
import numpy as np

class TestRepslyData(TestCase):
    def setUp(self):
        self.array = np.ndarray
        self.file_name = 'data/trial_users_analysis.csv'
        self.two_users_file = 'data/trial_two_users.csv'
        self.ten_users_file = 'data/trial_ten_users.csv'

        self.batch_size_fast = 1
        self.batch_size_slow = 64
        self.train_size = 0.8

        self.full_size_slow = round(103456 / 16)
        self.train_size_slow = round(0.8 * self.full_size_slow)
        self.val_size_slow = round(0.1 * self.full_size_slow)
        self.test_size_slow =  self.full_size_slow - self.train_size_slow - self.val_size_slow

        self.repsly_data = RepslyData()

    def test_convert_row_to_int(self):
        repsly_data = self.repsly_data

        columns_dict = {'ActiveReps': 5,
                         'ActivitiesPerRep': 6,
                         'AuditsCnt': 8,
                         'ClientNotesCnt': 9,
                         'Edition': 2,
                         'FormsCnt': 10,
                         'ImportCnt': 19,
                         'MessagesCnt': 7,
                         'NewPlaceCnt': 11,
                         'OrdersCnt': 12,
                         'PhotosCnt': 13,
                         'Purchased': 1,
                         'ScheduleCnt': 16,
                         'ScheduledPlacesCnt': 18,
                         'ScheduledRepsCnt': 17,
                         'StatusChangedCnt': 14,
                         'TrialDate': 4,
                         'TrialStarted': 3,
                         'WorkdayStartCnt': 15,
                         '\ufeffUserID': 0}

        data_indexes = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        trial_started = 'TrialStarted'
        row = np.array(['13336', '0', 'N/A', '2016-01-07', '4', '2', '-1', '3', '-4', '0', '0', '0', '0', '0', '0', '0', '5', '1', '0', '2'])
        expected_row_data = np.array([0, 0, 0, 6, 0, 2, -1, 3, -4, 0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 2])
        first_date = first_date='2016-01-01'
        row_data = repsly_data._convert_row_to_int(columns_dict, data_indexes, trial_started, row, first_date)
        np.testing.assert_array_equal(row_data, expected_row_data)

    def test_read_user_data_fast(self):
        repsly_data = self.repsly_data

        file_name = self.two_users_file

        with open(file_name) as f:
            mycsv = csv.reader(f)
            user_data_gen = repsly_data._read_user_data(mycsv)

            X, y = next(user_data_gen)
            self.assertIsInstance(X, np.ndarray)
            self.assertEqual(y, 0)
            np.testing.assert_array_equal(X.shape, [16, 16])
            expected_ninth_row = [3, 1, 40, 0, 0, 2, 0, 9, 0, 18, 0, 1, 0, 0, 0, 0]
            np.testing.assert_array_equal(X[9], expected_ninth_row)

            X, y = next(user_data_gen)
            self.assertIsInstance(X, np.ndarray)
            self.assertEqual(y, 1)
            np.testing.assert_array_equal(X.shape, [16, 16])
            expected_sweetsixteen_row = [2,245,45,0,0,362,7265,0,0,0,0,242,0,0,0,66]
            np.testing.assert_array_equal(X[15], expected_sweetsixteen_row)

    def test_read_user_data_slow(self):
        repsly_data = self.repsly_data

        file_name = self.file_name

        with open(file_name) as f:
            for X, y in repsly_data._read_user_data(csv.reader(f)):
                np.testing.assert_array_equal(X.shape, [16, 16])

    def _test_prepare_data(self, file_name, no_of_lines, mode, expected_data_shape):
        repsly_data = self.repsly_data

        repsly_data._prepare_data(file_name, mode)

        self.assertIsNotNone(repsly_data.X_all)
        self.assertIsNotNone(repsly_data.y_all)
        self.assertIsInstance(repsly_data.X_all, np.ndarray)
        self.assertIsInstance(repsly_data.y_all, np.ndarray)

        X_all_shape = np.shape(repsly_data.X_all)
        y_all_shape = np.shape(repsly_data.y_all)

        self.assertGreaterEqual(X_all_shape[0], no_of_lines // 16)
        self.assertEqual(X_all_shape[0], y_all_shape[0])

        np.testing.assert_array_equal(X_all_shape[1:], expected_data_shape)

    def test_prepare_data_fast(self):
        expected_data_shape = {
            (self.two_users_file, 31, 'FC'): [1+15*16],
            (self.two_users_file, 31, 'CONV'): [16, 16]
        }
        for params in expected_data_shape.keys():
            self._test_prepare_data(*params, expected_data_shape=expected_data_shape[params])

    def test_prepare_data_slow(self):
        expected_data_shape = {
            (self.file_name, 103456, 'FC'):  [1+15*16],
            (self.file_name, 103456, 'CONV'): [16, 16]
        }
        for params in expected_data_shape.keys():
            self._test_prepare_data(*params, expected_data_shape=expected_data_shape[params])

    def _test_read_data(self, file_name, mode, expected_lengths):
        repsly_data = self.repsly_data

        repsly_data._prepare_data(file_name, mode)
        repsly_data.read_data(file_name, mode)

        for data_set in expected_lengths.keys():
            self.assertEqual(len(repsly_data.X[data_set]), expected_lengths[data_set])

    def test_read_data_fast(self):
        expected_lengths = {
            (self.ten_users_file, 'FC'): {'train': 8, 'validation': 1, 'test': 1},
            (self.ten_users_file, 'CONV'): {'train': 8, 'validation': 1, 'test': 1}
        }

        for params in expected_lengths.keys():
            self._test_read_data(*params, expected_lengths=expected_lengths[params])

    def test_read_data_slow(self):
        expected_lengths = {
            (self.file_name, 'FC'):
                {'train': self.train_size_slow,
                 'validation': self.val_size_slow,
                 'test': self.test_size_slow},
            (self.file_name, 'CONV'):
                {'train': self.train_size_slow,
                'validation': self.val_size_slow,
                'test': self.test_size_slow}
        }

        for params in expected_lengths.keys():
            self._test_read_data(*params, expected_lengths=expected_lengths[params])


    def _test_read_batch(self, file_name, mode, expected_length, batch_size, expected_X_shape, expected_y_shape):
        repsly_data = self.repsly_data

        repsly_data._prepare_data(file_name, mode)
        repsly_data.read_data(file_name, mode)

        for data_set in ['train', 'validation', 'test']:
            i = 0

            for X, y in repsly_data.read_batch(batch_size=batch_size, data_set=data_set):
                np.testing.assert_array_equal(X.shape, expected_X_shape)
                np.testing.assert_array_equal(y.shape, expected_y_shape)
                i = i + 1

            self.assertEqual(i, expected_length[data_set])

    def _test_read_batch_dispatch(self, expected_length, batch_size, expected_X_shape, expected_y_shape):
        for params in expected_length.keys():
            _, mode = params
            self._test_read_batch(*params,
                                  expected_length=expected_length[params],
                                  batch_size=batch_size,
                                  expected_X_shape=expected_X_shape[mode],
                                  expected_y_shape=expected_y_shape)

    def test_read_batch_fast(self):
        expected_length = {
            (self.ten_users_file, 'FC'):
                {'train': 8,
                 'validation': 1,
                 'test': 1},
            (self.ten_users_file, 'CONV'):
                {'train': 8,
                 'validation': 1,
                 'test': 1
                 }
        }

        expected_X_shape = {
            'FC': [self.batch_size_fast, 1 + 15 * 16],
            'CONV': [self.batch_size_fast, 16, 16]
        }

        expected_y_shape = self.batch_size_fast
        batch_size = self.batch_size_fast

        self._test_read_batch_dispatch(expected_length, batch_size, expected_X_shape, expected_y_shape)

    def test_read_batch_slow(self):
        expected_length = {
            (self.file_name, 'FC'): {
                'train': self.train_size_slow // self.batch_size_slow,
                'validation': self.val_size_slow // self.batch_size_slow,
                'test': self.test_size_slow // self.batch_size_slow},
            (self.file_name, 'CONV'): {
                'train': self.train_size_slow // self.batch_size_slow,
                'validation': self.val_size_slow // self.batch_size_slow,
                'test': self.test_size_slow // self.batch_size_slow
            }
        }

        expected_X_shape = {
            'FC': [self.batch_size_slow, 1 + 15 * 16],
            'CONV': [self.batch_size_slow, 16, 16]
        }

        expected_y_shape = self.batch_size_slow
        batch_size = self.batch_size_slow

        self._test_read_batch_dispatch(expected_length, batch_size, expected_X_shape, expected_y_shape)
