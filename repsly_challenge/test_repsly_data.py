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
        self.repsly_data = RepslyData(file_name=self.file_name)

        repsly_data = self.repsly_data

        editions = {'N/A': 0, 'visibility': 1, 'jos jedan': 2, 'i ima jos neki': 3}

        self.data_from_file = np.array([
            [18380, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [18380, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [18380, 0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [18381, 0, 0, 1, 0, 14, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [18381, 0, 0, 1, 0, 14, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [18381, 0, 0, 1, 0, 14, 2, 0, 0, 1, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int)

        self.X_shape = [2, 5 + 3 * (25 - 8)]

        self.X = np.array([
            [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 14, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int)
        # first 2 are the type
        self.y = np.array([1, 0])

    def assertCanBeCastToInteger(self, s):
        a = int(s)
        self.assertIsInstance(a, int)

    def string_to_datetime(self, s):
        return datetime.strptime(s, '%Y-%m-%d')

    def string_to_days(self, s, first_date):
        return (self.string_to_datetime(s) - first_date).days

    def assertIsDate(self, s):
        d = self.string_to_datetime(s)
        self.assertIsInstance(d, datetime)
        i = d.day

    def test_trial_users_analysis_file(self):
        '''
        Trying to figure put if we understand input data
        :return:
        '''

        first_date = self.string_to_datetime('2016-1-1')

        self.assertTrue(os.path.exists(self.file_name))
        with open(self.file_name) as csvfile:
            mycsv = csv.reader(csvfile)

            edition_classes = set()
            isFirstLine = True
            for row in mycsv:
                # check first line
                if isFirstLine:
                    self.assertEqual(len(row), 22)
                    for i in row:
                        self.assertIsInstance(i, str)
                    isFirstLine = False
                    isSecondLine = True
                # check every other row
                else:
                    # row[0] is user ID
                    integer_fields = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                    for i in integer_fields:
                        self.assertCanBeCastToInteger(row[i])

                    edition_class_class_field = 2
                    edition_classes = edition_classes | {row[edition_class_class_field]}

                    date_field = 3
                    ds = row[date_field]
                    d = self.string_to_days(ds, first_date)
                    self.assertIsInstance(d, int)
                    self.assertGreaterEqual(d, 0)

    def test_get_classes_from_file(self):
        repsly_data = self.repsly_data
        editions = repsly_data.get_classes_from_file()
        self.assertIsInstance(editions, dict)
        self.assertEqual(len(editions), 4)
        self.assertIn('N/A', editions.keys())
        self.assertIn('organization', editions.keys())

    def test_read_row_from_file(self):
        repsly_data = self.repsly_data
        row = ['18380', '0', 'N/A', '2016-10-15', '0', '0', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0',
               '0', '0', '0', '0']
        editions = repsly_data.get_classes_from_file()
        encoded_row = repsly_data.read_row_from_file(row)
        expected_row = [18380, 0, 0, 0, 1, 288, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        np.testing.assert_array_equal(encoded_row, expected_row)

    def test_read_from_file(self):
        repsly_data = self.repsly_data
        print('Reading from file...')
        self.array = repsly_data.read_from_file()
        print('Reading from file fisnihed!')
        print(self.array)
        self.assertEqual(np.shape(self.array)[1], 25)
#        np.testing.assert_array_equal(np.shape(self.array), [224, 25])

    # def test_normalize_column(self):
    #     repsly_data = self.repsly_data
    #
    #     num_of_columns = 5
    #     num_of_rows = 4
    #
    #     ix = np.array(range(num_of_columns))
    #     x = np.array(range(num_of_columns*num_of_rows)).reshape([num_of_rows, num_of_columns])
    #     print(x)
    #     repsly_data.normalize_column(x, ix)
    #     print(x)
    #     mean_columns = np.mean(x, axis=0)
    #     std_columns = np.std(x, axis=0)
    #     np.testing.assert_array_equal(mean_columns.shape, [num_of_columns])
    #     np.testing.assert_array_equal(std_columns.shape, [num_of_columns])
    #
    #     for i in range(num_of_columns):
    #         if i in ix:
    #             self.assertEquals(mean_columns[i], 0)
    #             self.assertEquals(std_columns[i], 1)
    #         else:
    #             self.assertNotEquals(mean_columns[i], 0)
    #             self.assertNotEquals(std_columns[i], 1)

    def test_prepare_data_for_plain_nn(self):
        repsly_data = self.repsly_data
        data_from_file = self.data_from_file
        X, y = self.X, self.y
        np.testing.assert_array_equal(X.shape, self.X_shape)

        X_t, y_t = repsly_data.prepare_data_for_plain_nn(data_from_file, 3)

        np.testing.assert_array_equal(X_t, X)
        np.testing.assert_array_equal(y_t, y)

    def test_read_data_for_plain_nn(self):
        repsly_data = self.repsly_data

        repsly_data.read_data_for_plain_nn()

        for X, y in repsly_data.read_batch(batch_size=5, data_set='train'):
            np.testing.assert_array_equal(X.shape, [5, 277])
            np.testing.assert_array_equal(y.shape, [5])
