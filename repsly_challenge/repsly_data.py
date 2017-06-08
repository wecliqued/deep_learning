import csv
from datetime import datetime
import numpy as np
import math

# ideja: treba nam funkcija koja cita batch

class RepslyData:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_array = None
        self.final_array = np.ndarray
        self.edition_ix = 2
        self.first_date = '2016-1-1'
        pass

    def get_full_array(self):
        return self.data_array

    def get_classes_from_file(self):
        x = dict()
        number = 0
        # todo 1: otvori fajl, procitaj sve u ix stupcu i vrati skup razlicitih stringova
        # open and read csv, store all found values for classes and assign them with appropriate number
        with open(self.file_name) as csvfile:
            mycsv = csv.reader(csvfile)
            isFirstLine = True
            isFound = False
            for row in mycsv:
                # check first line
                isFound=False
                if isFirstLine:
                    isFirstLine = False
                else:
                    key = row[2]
                    #date = row[3]
                    #if date < self.first_date:
                    #    self.first_date = date
                    if key in x.keys():
                        isFound = True
                    if not isFound:
                        x[key] = number
                        number += 1
        return x

    def _string_to_datetime(self, s):
        return datetime.strptime(s, '%Y-%m-%d')

    def _string_to_days(self, s, first_date):
        return (self._string_to_datetime(s) - self._string_to_datetime(first_date)).days

    def read_row_from_file(self, row):
        encoded_row = []
        # todo 2: sve castaj u integer, neke razvuci u one hot vektor
        position = 0
        for element in row:
            if position == 2:
                classes_ids = self.get_classes_from_file()
                arr = [0, 0, 0, 0]
                arr[len(arr) - 1 - classes_ids.get(element)] = 1
                encoded_row = encoded_row + arr
            elif position == 3:
                day = self._string_to_days(element, self.first_date)
                encoded_row.append(day)
            else:
                if position == 1:
                    bought = int(element)
                else:
                    encoded_row.append(int(element))
            position += 1
        if len(encoded_row) > 0:
            encoded_row.append(bought)
        return encoded_row

    # todo 3> popravi sto smo strgali
    def read_from_file(self):
        editions = self.get_classes_from_file()

        i = 0
        with open(self.file_name) as csvfile:
            mycsv = csv.reader(csvfile)
            isFirstLine = True
            for row in mycsv:
                # check first line
                if isFirstLine:
                    isFirstLine = False
                # check every other row
                else:
                    row_arr = self.read_row_from_file(row)
                    print('self.data_array:', self.data_array)
                    print('row_arr:', row_arr)
                    if self.data_array is None:
                        self.data_array = np.array([row_arr])
                    else:
                        self.data_array = np.append(self.data_array, [row_arr], axis=0)

                if i % 100 == 0:
                    print('row: ', i)
                i = i + 1
        self.final_array = np.asarray(self.data_array)
        return np.asarray(self.data_array)

    # def normalize_column(self, x, ix):
    #     '''
    #
    #     :param x: numpy 2D arrray
    #     :param ix: index of column
    #     '''
    #     mean = np.mean(x, axis=0)[ix]
    #     std = np.std(x, axis=0)[ix]
    #     x[:, ix] = (x[:, ix] - mean) / std

    def prepare_data_for_plain_nn(self, data, num_of_entries_per_id):
        num_of_trials = data.shape[0] // num_of_entries_per_id
        num_of_columns = data.shape[1]
        num_of_repeating_features = num_of_columns-8
        X = np.zeros([num_of_trials, 5+num_of_entries_per_id*num_of_repeating_features], dtype=np.int)
        y = np.zeros([num_of_trials])

        data_reshaped = np.reshape(data, [num_of_trials, -1])
        y = data_reshaped[:, -1]
        X[:, 0:4] = data_reshaped[:, 1:5]
        X[:, 4] = data_reshaped[:, 5]
        for i in range(num_of_entries_per_id):
            features = data_reshaped[:, i * num_of_columns + 7:i * (num_of_columns) + 7 + num_of_repeating_features]
            if i is 0:
                X[:, i*num_of_repeating_features+5:(i+1)*(num_of_repeating_features)+5] = features
            else:
                X[:, i*num_of_repeating_features+5:(i+1)*(num_of_repeating_features)+5] = features - last_features
            last_features = features

        return X, y

    def read_data_for_plain_nn(self):
        data = self.read_from_file()
        X, y = self.prepare_data_for_plain_nn(data, 16)
        no_of_data = X.shape[0]
        no_of_train_data = int(no_of_data * 0.9)
        no_of_validation_data = int(no_of_data * 0.1)

        np.random.seed(0)
        ix = np.random.permutation(no_of_data)

        self.X, self.y = {}, {}

        self.X['train'] = X[ix[:no_of_train_data], :]
        self.y['train'] = y[ix[:no_of_train_data]]

        self.X['validation'] = X[ix[no_of_train_data:no_of_train_data+no_of_validation_data], :]
        self.y['validation'] = y[ix[no_of_train_data:no_of_train_data+no_of_validation_data]]

        self.X['test'] = X[ix[no_of_train_data:no_of_train_data+no_of_validation_data:], :]
        self.y['test'] = y[ix[no_of_train_data:no_of_train_data+no_of_validation_data:]]

    def read_batch(self, batch_size, data_set='train'):
        no_of_data = self.X[data_set].shape[0]
        X, y = self.X[data_set], self.y[data_set]

        for i in range(no_of_data // batch_size):
            yield X[i * batch_size:(i + 1) * batch_size, :], \
                  y[i * batch_size:(i + 1) * batch_size]
