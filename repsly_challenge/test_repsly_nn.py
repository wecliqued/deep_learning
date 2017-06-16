from unittest import TestCase
import tensorflow as tf
import numpy as np

from repsly_nn import RepslyFC
from repsly_data import RepslyData


class TestRepslyFC(TestCase):
    def setUp(self):
        self.repsly_fc = RepslyFC()

        self.archs = {
            'without_batch_norm': {
                'no_of_layers': 3,
                'hidden_size': 257,
                'use_batch_norm': False,
                'keep_prob': 0.8,
                'input_keep_prob': 0.9},
            'with_batch_norm': {
                'no_of_layers': 4,
                'hidden_size': 137,
                'use_batch_norm': True,
                'keep_prob': 0.5,
                'input_keep_prob': 0.8,
                'batch_norm_decay': 0.95},
        }

        np.random.seed(0)
        self.X = np.random.randint(0-4, 4, [3, 241])
        self.y = np.random.randint(0, 1, [3])
        self.keep_prob = 0.9

        ten_users_file = 'data/trial_ten_users.csv'
        self.data = RepslyData()
        self.data.read_data(ten_users_file, 'FC')
        self.batch_size = 1

    def print_trainable_variables(self):
        size = tf.Dimension(0)
        print('*' * 80)
        for v in tf.trainable_variables():
            print('{}[{}]'.format(v.name, v.shape))
            size += np.prod(v.shape)
        print('TOTAL SIZE: {}\n{}'.format(size, '*' * 80))

    def expected_num_trainable_variables(self, arch):
        input_size = 241
        num_variables = 0
        no_of_layers = arch['no_of_layers']
        use_batch_normalization = arch['use_batch_norm']
        hidden_size = arch['hidden_size']
        for _ in range(no_of_layers):
            if use_batch_normalization:
                # each neuron has input_size weights,
                # but no bias because it is disabled by biases_initializer=None
                # in tf.contrib.layers.fully_connected()
                num_variables += input_size * hidden_size
                # each neuron has one learnable parameter beta,
                # but no learnable parameter gamma because it is disabled by scale=False
                # in tf.contrib.layers.batch_norm (not needed for ReLU)
                num_variables += hidden_size
            else:
                # each neuron has input_size weights and one bias
                num_variables += (input_size + 1) * hidden_size
            input_size = hidden_size

        # last layer is a linear classifier: input_size weights and one bias
        num_variables += (input_size+1) * 2

        return num_variables

    def test__create_placeholders(self):
        repsly_nn = self.repsly_fc

        X, y, keep_prob = repsly_nn._create_placeholders()

    def test__create_model(self):
        repsly_nn = self.repsly_fc
        for arch_name, arch in self.archs.items():
            # drop everything created so far
            tf.reset_default_graph()

            repsly_nn._create_placeholders()

            # one of the easiest sanity checks is the number of variables created
            self.assertEqual(repsly_nn.get_num_of_trainable_variables(), 0)
            repsly_nn._create_model(arch)
            self.assertEqual(repsly_nn.get_num_of_trainable_variables(), self.expected_num_trainable_variables(arch))

    def _test__calculate_f1_score(self, repsly_nn):
        tp, fp, tn, fn = 2, 3, 5, 7
        feed_dict = {repsly_nn.prediction: [1]*(tp+fp)   + [0]*(tn+fn),
                     repsly_nn.y:          [1]*tp+[0]*fp + [0]*tn+[1]*fn
        }
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        expected_f1_score = 2 * precision * recall / (precision+recall)
        with tf.Session() as sess:
            f1_score = sess.run(repsly_nn.f1_score, feed_dict)
            self.assertEqual(f1_score, expected_f1_score)


    def test_create_net(self):
        repsly_nn = self.repsly_fc
        arch = self.archs['with_batch_norm']

        repsly_nn.create_net(arch)

        self._test__calculate_f1_score(repsly_nn)

        print('_name_extension():', repsly_nn._name_extension())

        # todo: finish test :)
#        self.fail()

    def test_train(self):
        repsly_nn = self.repsly_fc
        arch = self.archs['with_batch_norm']
        data = self.data
        batch_size = self.batch_size

        repsly_nn.create_net(arch)
        repsly_nn.train(data, batch_size, epochs=2)

        # todo: finish test :)
#        self.fail()

    def test_checkpoint_save_and_restore(self):
        repsly_nn = self.repsly_fc
        arch = self.archs['with_batch_norm']
        data = self.data
        batch_size = self.batch_size
        read_batch = data.read_batch(batch_size)

        # create network
        repsly_nn.create_net(arch)

        # check that all variables are created
        self.assertEqual(repsly_nn.get_num_of_trainable_variables(), self.expected_num_trainable_variables(arch))

        # create feed dictionary for loss calculation
        batch = next(read_batch)
        feed_dict = repsly_nn._create_feed_dictionary(batch, is_training=False)

        # Saver must be created *after* variables are created, otherwise it will fail
        repsly_nn._create_checkpoint_saver()

        # first, we will create session and initialize variables
        # and then we will run one batch and calculate loss
        # that loss will be used to check if restore is working
        with tf.Session() as sess1:
            # initialize global variables
            sess1.run(tf.global_variables_initializer())

            loss1 = sess1.run(repsly_nn.loss, feed_dict=feed_dict)
            print('loss1:', loss1)

            saved_path = repsly_nn._save_checkpoint(sess1)
            self.assertIsNotNone(saved_path)

        with tf.Session() as sess2:
            restored = repsly_nn._restore_checkpoint(sess2)
            self.assertTrue(restored)

            loss2 = sess2.run(repsly_nn.loss, feed_dict=feed_dict)
            print('loss2:', loss2)

            self.assertEqual(loss1, loss2)
