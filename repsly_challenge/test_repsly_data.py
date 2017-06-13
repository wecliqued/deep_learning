from encodings import johab
from unittest import TestCase
from repsly_data import RepslyData

import os
import csv
from datetime import datetime
import numpy as np


class TestRepslyData(TestCase):
    def setUp(self):
        self.array = np.ndarray
        self.file_name = 'data/trial_users_analysis.csv'
        self.two_users_file = 'data/trial_two_users.csv'
        self.ten_users_file = 'data/trial_ten_users.csv'

        self.expected_data_shape_fast = {
            (self.two_users_file, 31, 'FC'): [1+15*16],
            (self.two_users_file, 31, 'CONV'): [16, 16]
        }
        self.expected_data_shape_slow = {
            (self.file_name, 103456, 'FC'):  [1+15*16],
            (self.file_name, 103456, 'CONV'): [16, 16]
        }
        self.batch_size = 4
        self.train_size = 0.8

        self.expected_X_shape = {
            'FC': [4, 1+15*16],
            'CONV': [4, 16, 16]
        }

        self.expected_y_shape = [4]

        self.expected_epoch_size_fast = {
            (self.ten_users_file, 'FC', 'train'): 2,
            (self.ten_users_file, 'FC', 'validation'): 0,
            (self.ten_users_file, 'FC', 'test'): 0,
            (self.ten_users_file, 'CONV', 'train'): 2,
            (self.ten_users_file, 'CONV', 'validation'): 0,
            (self.ten_users_file, 'CONV', 'test'): 0
        }

        train_size = (0.8 * 103456 // (self.batch_size * 16))
        val_test_size = (0.1 * 103456 // (self.batch_size * 16))
        full_size = 103456//(self.batch_size*16)

        self.expected_epoch_size_slow = {
            (self.file_name, 'FC', 'train'): train_size,
            (self.file_name, 'FC', 'validation'): val_test_size,
            (self.file_name, 'FC', 'test'): (val_test_size + (full_size-(train_size+(val_test_size*2)))),
            (self.file_name, 'CONV', 'train'): train_size,
            (self.file_name, 'CONV', 'validation'): val_test_size,
            (self.file_name, 'CONV', 'test'): (val_test_size + (full_size-(train_size+(val_test_size*2))))
        }

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
        for params in self.expected_data_shape_fast.keys():
            self._test_prepare_data(*params, expected_data_shape=self.expected_data_shape_fast[params])

    def test_prepare_data_slow(self):
        for params in self.expected_data_shape_slow.keys():
            self._test_prepare_data(*params, expected_data_shape=self.expected_data_shape_slow[params])


    def _test_read_batch(self, file_name, mode, data_set, expected_length):
        repsly_data = self.repsly_data

        repsly_data._prepare_data(file_name, mode)
        repsly_data.read_data(file_name, mode)

        self.i = 0

        for X, y in repsly_data.read_batch(batch_size=self.batch_size, data_set=data_set):
            np.testing.assert_array_equal(X.shape, self.expected_X_shape[mode])
            np.testing.assert_array_equal(y.shape, self.expected_y_shape)
            self.i = self.i + 1

        print(file_name, mode, data_set, expected_length)
        self.assertEqual(self.i, expected_length)

    def test_read_batch_fast(self):
        for params in self.expected_epoch_size_fast.keys():
            self._test_read_batch(*params, expected_length=self.expected_epoch_size_fast[params])

    def test_read_batch_slow(self):
        for params in self.expected_epoch_size_slow.keys():
            self._test_read_batch(*params, expected_length=self.expected_epoch_size_slow[params])
