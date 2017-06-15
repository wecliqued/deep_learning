from unittest import TestCase
import tensorflow as tf
import numpy as np

from repsly_nn import RepslyFC
from repsly_data import RepslyData


class TestRepslyFC(TestCase):
    def setUp(self):
        self.repsly_fc = RepslyFC()
        self.arch = [(100, False), (200, False)]

        self.arch_dict = {'keep_prob': 0.8, 'input_keep_prob': 0.9, 'batch_norm_decay': 0.9}

        np.random.seed(0)
        self.X = np.random.randint(0-4, 4, [3, 241])
        self.y = np.random.randint(0, 1, [3])
        self.keep_prob = 0.9

        ten_users_file = 'data/trial_ten_users.csv'
        self.data = RepslyData()
        self.data.read_data(ten_users_file, 'FC')
        self.batch_size = 1

    def expected_num_variables(self):
        input_size = 241
        num_variables = 0
        for hidden_size, use_batch_normalization in self.arch:
            num_variables += (input_size+1) * hidden_size
            input_size = hidden_size
            if use_batch_normalization:
                num_variables += 2*hidden_size
        num_variables += (input_size+1) * 2

        return num_variables

    def test__create_placeholders(self):
        repsly_nn = self.repsly_fc

        X, y, keep_prob = repsly_nn._create_placeholders()

    def test__create_model(self):
        repsly_nn = self.repsly_fc
        arch = self.arch

        repsly_nn._create_placeholders()

        # one of the easiest sanity checks is the number of variables created
        self.assertEqual(repsly_nn.get_num_of_variables(), 0)
        repsly_nn._create_model(arch)
        self.assertEqual(repsly_nn.get_num_of_variables(), self.expected_num_variables())

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
        arch = self.arch
        arch_dict = self.arch_dict

        repsly_nn.create_net(arch, arch_dict)

        self._test__calculate_f1_score(repsly_nn)

        print('_name_extension():', repsly_nn._name_extension())

        # todo: finish test :)
#        self.fail()

    def test_train(self):
        repsly_nn = self.repsly_fc
        arch = self.arch
        arch_dict = self.arch_dict
        data = self.data
        batch_size = self.batch_size

        repsly_nn.create_net(arch, arch_dict)
        repsly_nn.train(data, batch_size, epochs=2)

        # todo: finish test :)
#        self.fail()

    def test_checkpoint_save_and_restore(self):
        repsly_nn = self.repsly_fc
        arch = self.arch
        arch_dict = self.arch_dict
        data = self.data
        batch_size = self.batch_size
        read_batch = data.read_batch(batch_size)

        # create network
        repsly_nn.create_net(arch, arch_dict)

        # check that all variables are created
        self.assertEqual(repsly_nn.get_num_of_variables(), self.expected_num_variables())

        # create feed dictionary for loss calculation
        batch = next(read_batch)
        feed_dict = repsly_nn._create_feed_dictionary(batch, training=False)

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
