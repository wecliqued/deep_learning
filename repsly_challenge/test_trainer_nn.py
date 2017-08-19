from unittest import TestCase
import tensorflow as tf
import numpy as np

from trainer_nn import TrainerFF
from test_batch_reader import DummyBatchReaader

class TestTrainer(TestCase):
    def setUp(self):
        self.trainer = TrainerFF(input_size=241)

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

        self.data = DummyBatchReaader()
        self.data.read_data(X_shape=(231, 241), y_shape=(231, ))
        self.batch_size = 31

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
        trainer = self.trainer

        X, y, keep_prob = trainer._create_placeholders()

    def test__create_model(self):
        trainer = self.trainer
        for arch_name, arch in self.archs.items():
            # drop everything created so far
            tf.reset_default_graph()

            trainer._create_placeholders()

            # one of the easiest sanity checks is the number of variables created
            self.assertEqual(trainer.get_num_of_trainable_variables(), 0)
            trainer._create_model(arch)
            self.assertEqual(trainer.get_num_of_trainable_variables(), self.expected_num_trainable_variables(arch))

    def _test__calculate_f1_score(self, trainer):
        tp, fp, tn, fn = 2, 3, 5, 7
        feed_dict = {trainer.prediction: [1]*(tp+fp)   + [0]*(tn+fn),
                     trainer.y:          [1]*tp+[0]*fp + [0]*tn+[1]*fn
        }
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        expected_f1_score = 2 * precision * recall / (precision+recall)
        with tf.Session() as sess:
            f1_score = sess.run(trainer.f1_score, feed_dict)
            self.assertEqual(f1_score, expected_f1_score)


    def test_create_net(self):
        trainer = self.trainer
        arch = self.archs['with_batch_norm']

        trainer.create_net(arch)

        self._test__calculate_f1_score(trainer)

        print('name_extension():', trainer.name_extension())

        # todo: finish test :)
#        self.fail()

    def test_train(self):
        trainer = self.trainer
        arch = self.archs['with_batch_norm']
        data = self.data
        batch_size = self.batch_size

        trainer.create_net(arch)
        trainer.train(data, batch_size, epochs=2)

        # todo: finish test :)
#        self.fail()

    def test_checkpoint_save_and_restore(self):
        trainer = self.trainer
        arch = self.archs['with_batch_norm']
        data = self.data
        batch_size = self.batch_size
        read_batch = data.read_batch(batch_size)

        # create network
        trainer.create_net(arch)

        # check that all variables are created
        self.assertEqual(trainer.get_num_of_trainable_variables(), self.expected_num_trainable_variables(arch))

        # create feed dictionary for loss calculation
        batch = next(read_batch)
        feed_dict = trainer._create_feed_dictionary(batch, is_training=False)

        # Saver must be created *after* variables are created, otherwise it will fail
        trainer._create_checkpoint_saver()

        # first, we will create session and initialize variables
        # and then we will run one batch and calculate loss
        # that loss will be used to check if restore is working
        with tf.Session() as sess1:
            # initialize global variables
            sess1.run(tf.global_variables_initializer())

            loss1 = sess1.run(trainer.loss, feed_dict=feed_dict)
            print('loss1:', loss1)

            saved_path = trainer._save_checkpoint(sess1)
            self.assertIsNotNone(saved_path)

        with tf.Session() as sess2:
            restored = trainer._restore_checkpoint(sess2)
            self.assertTrue(restored)

            loss2 = sess2.run(trainer.loss, feed_dict=feed_dict)
            print('loss2:', loss2)

            self.assertEqual(loss1, loss2)
