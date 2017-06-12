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

    def _test_read_data(self, file_name, no_of_lines, mode):
        repsly_data = self.repsly_data

        self.assertFalse(hasattr(repsly_data, 'X_all'))
        self.assertFalse(hasattr(repsly_data, 'X'))
        repsly_data._prepare_data(file_name, mode)

        repsly_data.read_data(file_name, 'FC')

        self.assertIsNotNone(repsly_data.X)      # 'RepslyData' object has no attribute 'X' -> not defined as self.X
        self.assertIsNotNone(repsly_data.X_all)
        self.assertIsInstance(repsly_data.X_all, np.ndarray)
        X_all_shape = np.shape(repsly_data.X_all)
        self.assertGreaterEqual(X_all_shape[0], no_of_lines // 16)

        if mode is 'FC':
            np.testing.assert_array_equal(X_all_shape[1], 1+15*16)
            np.testing.assert_array_equal(repsly_data.y_all.shape, [X_all_shape[0]])
        elif mode is 'CONV':
            np.testing.assert_array_equal(X_all_shape[1], [16, 16])
            np.testing.assert_array_equal(repsly_data.y_all.shape, [X_all_shape[0]])
        else:
            raise 'TEST: Unsupported mode in {}, should be FC or CONV'.format(mode)

    def test_prepare_data_fast_for_FC_mode(self):
        file_name = self.two_users_file
        no_of_lines = 31
        self._test_read_data(file_name, no_of_lines, 'FC')

    def test_prepare_data_slow_for_FC_mode(self):
        file_name = self.file_name
        no_of_lines = 103456
        self._test_read_data(file_name, no_of_lines, 'FC')

    def test_prepare_data_fast_for_CONV_model(self):
        file_name = self.two_users_file
        no_of_lines = 31
        self._test_read_data(file_name, no_of_lines, 'CONV')

    def test_prepare_data_slow_for_CONV_model(self):
        file_name = self.file_name
        no_of_lines = 103456
        self._test_read_data(file_name, no_of_lines, 'CONV')

    def test_read_batch(self,file_name, mode, length):
        repsly_data = self.repsly_data

        repsly_data._prepare_data(file_name, mode)
        repsly_data.read_data(file_name, mode)
        self.i = 0
        for X, y in repsly_data.read_batch(batch_size=4, data_set='train'):
            np.testing.assert_array_equal(X.shape, [4, 1+15*16])
            np.testing.assert_array_equal(y.shape, [4])
            self.i = self.i + 1
        self.assertEqual(self.i, length)

    def test_read_batch_fast_for_FC_mode(self):
        file_name = self.ten_users_file

        self.test_read_batch(file_name,'FC', 2)

    def test_read_batch_slow_for_FC_mode(self):
        file_name = self.file_name

        self.test_read_batch(file_name, 'FC', (0.8 * 103456 // (4 * 16)))

    def test_read_batch_fast_for_CONV(self):
        file_name = self.ten_users_file

        self.test_read_batch(file_name, 'CONV', 2)

    def test_read_batch_slow_for_CONV(self):
        file_name = self.file_name
        pass