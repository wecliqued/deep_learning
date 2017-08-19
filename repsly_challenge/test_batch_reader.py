from unittest import TestCase
import numpy as np
from batch_reader import BatchReader
from math import ceil

class DummyBatchReaader(BatchReader):
    def _prepare_data(self, **params):
        assert params['X_shape'][0] == params['y_shape'][0]

        all_X = np.random.rand(*params['X_shape'])
        all_y = np.random.rand(*params['y_shape'])

        return all_X, all_y

class TestBatchReader(TestCase):
    def test__partition_into_data_sets(self):
        all_X = np.random.rand(13, 7)
        all_y = np.random.rand(13)

        # make the call
        data = BatchReader()._partition_into_data_sets(all_X, all_y, 0.8)

        # check that each data set has expected size
        exp_data_set_sizes = {'train': 10, 'validation': 1, 'test': 2}
        for data_set in ['train', 'validation', 'test']:
            X, y = data[data_set]
            np.testing.assert_array_equal(X.shape, (exp_data_set_sizes[data_set], 7))
            np.testing.assert_array_equal(y.shape,  exp_data_set_sizes[data_set])

        # check that no data is lost or added but only reordered and partitioned
        shuffled_all_X = np.concatenate([X for X, y in data.values()])
        shuffled_all_y = np.concatenate([y for X, y in data.values()])
        np.testing.assert_array_equal(np.sort(shuffled_all_X, axis=0), np.sort(all_X, axis=0))
        np.testing.assert_array_equal(np.sort(shuffled_all_y, axis=0), np.sort(all_y, axis=0))

    def test_read_batch(self):
        batch_reader = DummyBatchReaader()

        # read  data
        batch_reader.read_data(X_shape=(53, 7), y_shape=(53, ), train_set_percentage=.8)

        batch_size = 3
        data_sets = ['train', 'validation', 'test']
        exp_epoch_size = {'train': 42, 'validation': 5, 'test': 6}
        exp_no_batches = {data_set: ceil(exp_epoch_size[data_set] / batch_size) for data_set in data_sets}

        # read all batches of an epoch for all data_sets
        epochs = {data_set: list(batch_reader.read_batch(batch_size, data_set)) for data_set in data_sets}

        all_X, all_y = {}, {}
        for data_set in data_sets:
            # read all batches
            batches = epochs[data_set]

            # check if epoch has the right size
            self.assertEqual(len(batches), exp_no_batches[data_set])

            # check if all batches add up to the epoch size
            all_X[data_set] = np.concatenate([X for X, y in epochs[data_set]])
            all_y[data_set] = np.concatenate([y for X, y in epochs[data_set]])
            np.testing.assert_array_equal(all_X[data_set].shape, (exp_epoch_size[data_set], 7))
            np.testing.assert_array_equal(all_y[data_set].shape, (exp_epoch_size[data_set], ))

            # check shape of all batches except the last one (the last one can be smaller)
            for X, y in batches[:-1]:
                np.testing.assert_array_equal(X.shape, (batch_size, 7))
                np.testing.assert_array_equal(y.shape, (batch_size))

            # check shape of the last batch (one or more rows, but not more than batch size)
            X, y = batches[-1]
            self.assertIn(X.shape[0], np.arange(start=1, stop=batch_size+1))
            self.assertEqual(X.shape[1], 7)
            self.assertIn(y.shape[0], np.arange(start=1, stop=batch_size + 1))

        # concatenate all datasets and reconstruct the original data shape
        all_X = np.concatenate(list(all_X.values()))
        all_y = np.concatenate(list(all_y.values()))
        np.testing.assert_array_equal(all_X.shape, (53, 7))
        np.testing.assert_array_equal(all_y.shape, (53, ))

        # shapes are ok, but now we have to check the content
        # sorting columns is not enough to check that in general, but it is close enough in practice)
        np.testing.assert_array_equal(all_X.sort(), batch_reader.all_X.sort())
        np.testing.assert_array_equal(all_y.sort(), batch_reader.all_y.sort())

