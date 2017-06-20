import pandas as pd
from datetime import datetime
import numpy as np

class RepslyData:
    def __init__(self):
        pass

    def _prepare_data(self, file_name, mode='FC'):
        assert(mode in ['FC', 'CONV'])
        self.X_all, self.y_all = None, None

        # load the dataset, create data frame from csv with all columns
        df = pd.read_csv(file_name)

        # prepare X
        # create index for columns that need to be casted to 15 trial days
        col_index = [0] + list(range(4, len(df.columns)))

        # cast the data columns into new X_all data frame
        X = df.iloc[:, col_index].pivot(index='UserID', columns='TrialDate')

        # insert new TrialStarted column and bind it with the rest of the X_all columns
        X.insert(0, 'TrialStarted', df.iloc[:, [0, 3]].groupby(
            'UserID').TrialStarted.unique().str[0])

        # convert TrialStarted column from string type to timestamp
        X['TrialStarted'] = pd.to_datetime(X["TrialStarted"])

        # subtract TrialStarted from first date
        first_date = datetime.strptime('2016-01-01', '%Y-%m-%d')
        X['TrialStarted'] = (X['TrialStarted'] - first_date).dt.days.astype(
            'int64')

        # prepare y
        y = df.iloc[:, [0, 1]].groupby('UserID').Purchased.unique().str[0]

        # convert X and y to numpy arrays
        self.X_all, self.y_all = X.values, y.values

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
                yield X[shuffle[-(no_of_data % batch_size):], :], \
                      y[shuffle[-(no_of_data % batch_size):]]
            once = True

