import numpy as np
import tensorflow as tf

class BatchReader:
    def __init__(self):
        pass

    def _prepare_data(self, **params):
        '''
        Must be overridden in inherited classes.

        :param params: input parameter specific to inherited class, usually input file name.
        :return: X, y tuple, where X is input and y is output data
        '''
        pass

    def _partition_into_data_sets(self, all_X, all_y, train_set_percentage):
        '''
        Partitions data into train, validation and test data set. We assume that y is last y_len columns in df.

        :param all_X: X data for train, validation and test data sets
        :param all_y: y data for train, validation and test data sets
        :param train_set_percentage: percentage of data to be used for training. The rest will be split equally between
               validation and test data sets.
        :return: A dictionary with keys ['train', 'validation', 'test'] and (X, y) tuple as values.
        '''
        # calculate absolute sizes for train, validation and (implied test) sets
        total_size = all_X.shape[0]
        train_size = round(total_size * train_set_percentage)
        validation_size = (total_size - train_size) // 2
        test_size = total_size - (train_size + validation_size)

        # create random, but reproducible permutation of input data
        np.random.seed(0)
        p = np.random.permutation(total_size)
        ix = {'train':      p[           : train_size],
              'validation': p[train_size : -test_size],
              'test':       p[-test_size :           ]}

        # create randomized (shuffled) view on data
        return {data_set: (all_X[ix[data_set]], all_y[ix[data_set]]) for data_set in ['train', 'validation', 'test']}

    def read_data(self, **params):
        train_set_percentage = params['train_set_percentage'] if 'train_set_percentage' in params else 0.8

        self.all_X, self.all_y = self._prepare_data(**params)
        self.data = self._partition_into_data_sets(self.all_X, self.all_y, train_set_percentage)

    def input_fn(self, batch_size, data_set='train'):
        '''
        Input function to be used by tf.estimator.Estimator subclasses

        :param batch_size: size of the batch
        :param data_set: one of 'train', 'validation' or 'test'

        :return: function returning
        '''
        X, y = self.data[data_set]
        shuffle = (data_set == 'train')
        return tf.estimator.inputs.numpy_input_fn(x={'x': X}, y=y, batch_size=batch_size, shuffle=shuffle)

    def read_batch(self, batch_size, data_set='train', endless=False):
        '''
        Iterator for reading batches.

        read_data() must be called prior to calling read_batch().

        After the whole epoch was read, it must be called again. The newly returned iterator will reshuffle the data.

        :param batch_size: size of the batch
        :param data_set: one of 'train', 'validation' or 'test'
        :param endless: if True (dafault is False), creates an endless iterator.
        :return:
        '''
        # get all X and y data for data_set
        X, y = self.data[data_set]
        no_of_data = X.shape[0]

        once = False
        while (endless or not once):
            # get a new permutation of the data for each epoch
            shuffle = np.random.permutation(no_of_data)
            for i in range(no_of_data // batch_size):
                ix = shuffle[i * batch_size : (i + 1) * batch_size]
                yield X[ix],  y[ix]
            # if some data is left, construct a smaller last batch
            if no_of_data % batch_size > 0:
                ix = shuffle[-(no_of_data % batch_size):]
                yield X[ix],  y[ix]
            once = True
