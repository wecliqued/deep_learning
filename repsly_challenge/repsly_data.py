import csv
from datetime import datetime
import numpy as np

class RepslyData:
    def __init__(self):
        pass

    def _string_to_datetime(self, s):
        return datetime.strptime(s, '%Y-%m-%d')

    def _string_to_days(self, s, first_date):
        return (self._string_to_datetime(s) - self._string_to_datetime(first_date)).days

    def _convert_row_to_int(self, columns_dict, data_indexes, trial_started, row, first_date):
        # convert all strings into int
        row_data = np.zeros_like(row, dtype=int)
        for i in data_indexes:
            if i is columns_dict[trial_started]:
                # convert Date to int
                row_data[i] = self._string_to_days(row[i], first_date)
            else:
                row_data[i] = int(row[i])
        return row_data

    def _read_user_data(self, mycsv, num_of_rows=16, user_columns='UserID', day_column='TrialDate', trial_started='TrialStarted', purchased_column='Purchased', ignore_columns=['Edition'], first_date='2016-01-01'):
        columns = next(mycsv)
        columns_dict = {columns[i]: i for i in range(len(columns))}

        num_of_columns = len(columns) - len(ignore_columns) - 3 # UserID, Purchased, TrialDate

        data_columns = [c for c in columns if c not in (np.concatenate([ignore_columns, [user_columns, day_column, purchased_column]]))]
        # check that trial_started is the first column
        assert(data_columns[0] == trial_started)
        data_indexes = [columns_dict[c] for c in data_columns]

        X, y = None, 0
        for row in mycsv:
            day = int(row[columns_dict[day_column]])
            if day is 0:
                # yield if you have something
                if X is not None:
                    yield X, y
                # allocate and initialize new output data
                X = np.zeros([num_of_rows, num_of_columns], dtype=np.int)
                y = int(row[columns_dict['Purchased']])

            row_data = self._convert_row_to_int(columns_dict, data_indexes, trial_started, np.array(row), first_date)
            X[day] = row_data[data_indexes]

        if X is not None:
            yield X, y

    def _prepare_data(self, file_name, mode):
        assert(mode in ['FC', 'CONV'])
        self.X_all, self.y_all = None, None
        with open(file_name) as f:
            mycsv = csv.reader(f)
            for X, y in self._read_user_data(mycsv):
                # the first column is a trial start day and there is no need to repeat it
                # the rest are columns that are possibly different each day and we will flatten them
                if mode is 'FC':
                    trial_start_day = X[0, 0]
                    repeating_columns =  np.reshape(X[:, 1:], [-1])
                    X = np.concatenate([[trial_start_day], repeating_columns])
                if self.X_all is None:
                    X_all_shape = np.concatenate([[0], X.shape])
                    self.X_all = np.zeros(X_all_shape)
                    self.y_all = np.array([], dtype=np.int)
                self.X_all = np.concatenate([self.X_all, [X]])
                self.y_all = np.concatenate([self.y_all, [y]])

    def read_data(self, file_name, mode, train_size=0.8):
        self._prepare_data(file_name, mode) # mode = 'FC' or 'CONV'
        no_of_data = self.X_all.shape[0]
        no_of_train_data = round(no_of_data * train_size)
        no_of_validation_data = round(no_of_data * ((1.0-train_size) / 2))
        no_of_test_data = no_of_data - (no_of_train_data-no_of_validation_data)

        np.random.seed(0)
        ix = np.random.permutation(no_of_data)

        self.X, self.y = {}, {}

        self.X['train'] = self.X_all[ix[:no_of_train_data], :]
        self.y['train'] = self.y_all[ix[:no_of_train_data]]

        self.X['validation'] = self.X_all[ix[no_of_train_data:no_of_train_data+no_of_validation_data], :]
        self.y['validation'] = self.y_all[ix[no_of_train_data:no_of_train_data+no_of_validation_data]]

        self.X['test'] = self.X_all[ix[no_of_train_data+no_of_validation_data:], :]
        self.y['test'] = self.y_all[ix[no_of_train_data+no_of_validation_data:]]

    def read_batch(self, batch_size, data_set='train', endless=False):
        no_of_data = self.X[data_set].shape[0]
        X, y = self.X[data_set], self.y[data_set]

        once = False
        while (endless or not once):
            shuffle = np.random.permutation(no_of_data)
            for i in range(no_of_data // batch_size):
                yield X[shuffle[i * batch_size:(i + 1) * batch_size], :], \
                      y[shuffle[i * batch_size:(i + 1) * batch_size]]
            if no_of_data % batch_size > 0:
                yield X[shuffle[-no_of_data % batch_size], :], \
                      y[shuffle[-no_of_data % batch_size]]
            once = True

