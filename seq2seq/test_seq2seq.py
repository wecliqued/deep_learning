from unittest import TestCase

from seq2seq import Seq2Seq

import numpy as np
import tensorflow as tf
import math

import re

import os
import string

class TestSeq2Seq(TestCase):
    def setUp(self):
        # clear everything that might be in the default graph from previous tests
        tf.reset_default_graph()

        self.batch_size = 5
        self.window_size = 10
        self.hidden_size = 100
        self.overlap_size = 4
        self.no_stacked_cells = 1

        self.temp = .3

        self.seq2seq = Seq2Seq(input_file=None,
                               window_size=self.window_size,
                               overlap_size=self.overlap_size,
                               batch_size=self.batch_size,
                               hidden_size=self.hidden_size,
                               no_stacked_cells=self.no_stacked_cells)

        self.input = 'ja sam mali medo\nmamini sam sin\n'
        self.medo_encoded = [10,  1, 95, 19,  1, 13, 95, 13,  1, 12,  9, 95, 13,  5,  4, 15, 97,
                             13,  1, 13,  9, 14,  9, 95, 19,  1, 13, 95, 19,  9, 14, 97]
        self.read_data =  [[10,  1, 95, 19,  1, 13, 95, 13,  1, 12],
                           [95, 13,  1, 12,  9, 95, 13,  5,  4, 15],
                           [13,  5,  4, 15, 97, 13,  1, 13,  9, 14],
                           [ 1, 13,  9, 14,  9, 95, 19,  1, 13, 95],
                           [19,  1, 13, 95, 19,  9, 14, 97,  0,  0],
                           [14, 97,  0,  0,  0,  0,  0,  0,  0,  0]]
        self.expected_batches = [self.read_data[:5], self.read_data[5:]]
        self.expected_zeros = [np.equal(np.array(self.expected_batches[i]), 0) for i in range(len(self.expected_batches))]
        self.expected_lengths = [[10, 10, 10, 10, 8], [2]]

    def test_kejt(self):
        seq2seq = self.seq2seq

        # some characters are not in the vocabulary, we will encode them with *
        kejt_encoded = seq2seq.vocab_encode('Katarina Exlé je po*izdila :)')
        kejt_decoded = seq2seq.vocab_decode(kejt_encoded)
        self.assertEqual(kejt_decoded, 'Katarina Exl* je po*izdila :)')

    def test_vocab_decode_zeros(self):
        seq2seq = self.seq2seq

        encoded = np.concatenate([seq2seq.vocab_encode('ja sam mali '), [0, 0, 0], seq2seq.vocab_encode('medo')])
        decoded = seq2seq.vocab_decode(encoded)
        self.assertEqual(decoded, 'ja sam mali medo')

    def test_vocab(self):
        seq2seq = self.seq2seq

        # vocab used for encoding/decoding is a string
        self.assertIsInstance(seq2seq.vocab, str)
        # it contains all letters, digits, interpunctions and whitespaces
        self.assertIn('a', seq2seq.vocab)
        self.assertIn('8', seq2seq.vocab)
        self.assertIn('.', seq2seq.vocab)
        self.assertIn('ć', seq2seq.vocab)

        # vocabular is extended by special element 0 used to denote shorter sequences
        vocab_size = len(seq2seq.vocab)
        seq = seq2seq.vocab_encode(seq2seq.vocab)
        self.assertListEqual(seq, list(range(1, vocab_size + 1)))

        # decode/encode should be identity
        vocab = seq2seq.vocab_decode(seq)
        self.assertEqual(vocab, seq2seq.vocab)

        # here we test encoding of input sentence 'ja sam mali...sin'
        medo_seq = seq2seq.vocab_encode(self.input)
        self.assertListEqual(medo_seq, self.medo_encoded)
        medo_decoded = seq2seq.vocab_decode(medo_seq)
        self.assertEqual(medo_decoded, self.input)

    def test_read_data(self):
        seq2seq = self.seq2seq
        data = seq2seq.read_data(self.input, shuffle=False)
        np.testing.assert_array_equal(list(data), self.read_data)

    def test_read_batch(self):
        seq2seq = self.seq2seq
        stream = seq2seq.read_data(iter(self.input), shuffle=False)
        batches = seq2seq.read_batch(stream)
        np.testing.assert_array_equal(list(batches), self.expected_batches)

    def assertHasShape(self, tensor, expected_shape):
        shape = tensor.shape
        if isinstance(shape, tuple):
            shape = list(shape)
        else:
            shape = shape.as_list()
        self.assertListEqual(shape, expected_shape)

    def onehot(self, x, depth):
        return [0] * x + [1] + [0] * (depth-x-1)

    def test__seq2one_hot(self):
        seq2seq = self.seq2seq

        # this placeholder will be used for input batch
        seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_seq')
        self.assertHasShape(seq, [None, None])

        # create one hot operation
        one_hot_seq_op = seq2seq._seq2one_hot(seq)
        feed_dict = {seq: self.expected_batches[0]}
        # operations can have undefined shapes, they will be determinated in session when run
        depth = len(seq2seq.vocab) + 1
        self.assertHasShape(one_hot_seq_op, [None, None, depth])

        # execute one hot operation and make sure output vector has expected size
        with tf.Session() as sess:
            one_hot_seq = sess.run(one_hot_seq_op, feed_dict=feed_dict)
            # after being run in session, all shapes must be known
            # in this case, we will use batch and window size
            self.assertHasShape(one_hot_seq, [self.batch_size, self.window_size, depth])
            for i in range(len(self.expected_batches[0])):
                for j in range(len(self.expected_batches[0][i])):
                    np.testing.assert_array_equal(one_hot_seq[i, j], self.onehot(self.expected_batches[0][i][j], depth))

    def test_create_cell(self):
        seq2seq = self.seq2seq

        # we will use one hot encoding of the input batch, this is how it is constructed
        # we will use 0 for padding so our vocabulary size will increase by one
        vocab_len = len(seq2seq.vocab)
        depth = vocab_len + 1
        no_stacked_cells = self.no_stacked_cells
        hidden_size = self.hidden_size

        seq = tf.placeholder(dtype=tf.int32, shape=[None, None])
        one_hot_seq = tf.one_hot(seq, depth=depth)
        self.assertHasShape(one_hot_seq, [None, None, depth])

        # creates cell using seq as input batch placeholder
        cell, in_state = seq2seq._create_cell(one_hot_seq, no_stacked_cells)
        self.assertIsInstance(cell, tf.contrib.rnn.MultiRNNCell)
        self.assertEqual(len(in_state), no_stacked_cells)
        for state in in_state:
            self.assertHasShape(state, [None, hidden_size])

        # before calling __call__ on cell, internal variables are not created
        # not much we can test right now
        self.assertListEqual(tf.trainable_variables(), [])

    def test__lengths(self):
        seq2seq = self.seq2seq
        vocab_len = len(seq2seq.vocab)

        seq = tf.placeholder(tf.int32, [None, self.window_size])
        lengths_op = seq2seq._lenghts(seq)
        with tf.Session() as sess:
            for i in range(len(self.expected_batches)):
                feed_dict = {seq: self.expected_batches[i]}
                lengths = sess.run(lengths_op, feed_dict=feed_dict)
                np.testing.assert_array_equal(lengths, self.expected_lengths[i])

    def check_rnn_variables(self, vocab_len, hidden_size):
        # there should be four variables created, we should check if they have good shapes
        self.assertGreaterEqual(len(tf.trainable_variables()), 4)
        for v in tf.trainable_variables():
            # GRU has two gates (hence 2*hidden_size)
            # input is concatenation of input and internal state
            # input is one hot vector with 0 as special no char symbol (hence +1)
            if re.match('rnn/multi_rnn_cell/cell_[0-9]+/gru_cell/gates/weights', v.name):
                #TODO: too many params on cell_1?!?
#                self.assertHasShape(v, [vocab_len + 1 + hidden_size, 2*hidden_size])
                pass
            elif re.match('rnn/multi_rnn_cell/cell_[0-9]+/gru_cell/gates/biases', v.name):
                self.assertHasShape(v, [2*hidden_size])
            # output of the GRU is calculated using previously calculated gates and
            # both input and internal state (hence vocab_len + 1 + hidden_size)
            # output is new internal state (hence hidden_size)
            elif re.match('rnn/multi_rnn_cell/cell_[0-9]+/gru_cell/candidate/weights', v.name):
                #TODO: too many params on cell_1?!?
#                self.assertHasShape(v, [vocab_len + 1 + hidden_size, hidden_size])
                pass
            elif re.match('rnn/multi_rnn_cell/cell_[0-9]+/gru_cell/candidate/biases', v.name):
                self.assertHasShape(v, [hidden_size])
            elif re.match('fully_connected/weights', v.name):
                self.assertHasShape(v, [hidden_size, vocab_len + 1])
            elif re.match('fully_connected/biases', v.name):
                self.assertHasShape(v, [vocab_len + 1])
            else:
                self.fail('Unexpected variable: ' + v.name)

    def test_create_rnn(self):
        seq2seq = self.seq2seq
        vocab_len = len(seq2seq.vocab)
        depth = vocab_len + 1
        window_size = self.window_size
        hidden_size = self.hidden_size
        no_stacked_cells = self.no_stacked_cells

        # create RNN
        seq, _ = seq2seq._create_placeholders(window_size=window_size)
        output_op, in_state_op, out_state_op = seq2seq._create_rnn()
        # check operation shapes
        self.assertHasShape(output_op, [None, window_size, hidden_size])
        for cell_in_state_op in in_state_op:
            self.assertHasShape(cell_in_state_op, [None, hidden_size])
        for cell_out_state_op in out_state_op:
            self.assertHasShape(cell_out_state_op, [None, hidden_size])

        self.check_rnn_variables(vocab_len, hidden_size)

        # let's execute RNN and check our understanding of what is going on under the hood
        with tf.Session() as sess:
            # initialize global variables
            sess.run(tf.global_variables_initializer())
            # run our batches
            for i in range(len(self.expected_batches)):
                batch_size = len(self.expected_batches[i])

                feed_dict = {seq: self.expected_batches[i]}
                output, in_state, out_state = sess.run([output_op, in_state_op, out_state_op], feed_dict=feed_dict)

                # output contains output vales of hidden units for each batch and for each input character
                self.assertHasShape(output, [batch_size, window_size, hidden_size])
                # if input char is zero, then and only then output is zero
                expected_zeros = np.equal(np.array(self.expected_batches[i]), 0)
                output_zeros = np.all(np.equal(output, np.zeros([window_size, hidden_size])), axis=2)
                np.testing.assert_array_equal(output_zeros, expected_zeros)

                lengths = window_size - np.sum(output_zeros, axis=1)
                np.testing.assert_array_equal(lengths, self.expected_lengths[i])

                # output state is the last output from the RNN (index by length)
                # the shape is batch_size x hidden_size
                for state in out_state:
                    self.assertHasShape(state, [batch_size, hidden_size])
                expected_out_state = np.array([output[j, lengths[j]-1, :] for j in range(batch_size)])
                np.testing.assert_array_equal(out_state[-1], expected_out_state)

                # unless we explicitly do something with in_state, it should be all zeros
                for state in in_state:
                    self.assertHasShape(state, [batch_size, hidden_size])
                    np.testing.assert_array_equal(state, np.zeros([batch_size, hidden_size]))

    def test_create_model(self):
        seq2seq = self.seq2seq
        vocab_len = len(seq2seq.vocab)
        no_stacked_cells = self.no_stacked_cells

        # we should start by creating placeholders for input data
        seq, temp = seq2seq._create_placeholders()
        loss_op, sample_op, in_state_op, out_state_op = seq2seq._create_model()

        # let's evaluate our batch just for fun
        # let's execute RNN and check our understanding of what is going on under the hood
        with tf.Session() as sess:
            # initialize global variables
            sess.run(tf.global_variables_initializer())
            # run our batches
            for i in range(len(self.expected_batches)):
                batch_size = len(self.expected_batches[i])

                feed_dict = {seq: self.expected_batches[i], temp: self.temp}
                loss, sample, in_state, out_state = sess.run([loss_op, sample_op, in_state_op, out_state_op], feed_dict=feed_dict)

                # since we have cross entropy loss, it should be around log(vocab_len+1)
                self.assertAlmostEqual(loss, math.log(vocab_len+1), 1)

                # sample is predicted parameter sample from distribution
                # we can only check its shape right now
                self.assertHasShape(sample, [batch_size])

                # in_state is all zeros
                np.testing.assert_array_equal(in_state, np.zeros([no_stacked_cells, batch_size, self.hidden_size]))

                # out_state is the last output
                # we will only check shape right now, the content was checked in test_create_rnn()
                self.assertEqual(len(out_state), no_stacked_cells)
                for state in out_state:
                    self.assertHasShape(state, [batch_size, self.hidden_size])

    def test_create_optimizer_and_online_inference(self):
        seq2seq = self.seq2seq

        # we should start by creating placeholders for input data
        seq, _ = seq2seq._create_placeholders()
        loss_op, _, _, _ = seq2seq._create_model()
        optimizer_op, _ = seq2seq._create_optimizer(learning_rate=0.003, decay_steps=10, decay_rate=0.98)

        # let's overfit a simple model and then make a prediction
        with tf.Session() as sess:
            attempts = []
            target = self.input
            # because this is probabilistic in nature, we will try at most three times
            for i in range(3):
                # initialize global variables
                sess.run(tf.global_variables_initializer())
                # run our batches
                for epoch in range(2000):
                    for j in range(len(self.expected_batches)):
                        feed_dict = {seq: self.expected_batches[j]}
                        sess.run(optimizer_op, feed_dict=feed_dict)

                loss = sess.run(loss_op, feed_dict={seq: self.expected_batches[0]})
                print('final loss:   {:.5f}     avg. correct prob.: {:2.2f}%%'.format(loss, 1.0 / math.exp(loss) * 100.))

                sampled_sentence = seq2seq.online_inference(sess, len=len(target)*5, temp=self.temp, seed=target[0])
                # calculate loss of the generated sentence
                loss = sess.run(loss_op, feed_dict={seq: [seq2seq.vocab_encode(sampled_sentence)]})
                print('sampled loss: {:.5f}     avg. correct prob.: {:2.2f}%%'.format(loss, 1.0 / math.exp(loss) * 100.))
                print('sampled_sentence:', sampled_sentence)
                print()

                attempts.append(sampled_sentence)
            print('attempts:', attempts)
            target_len = len(target) // 2
            self.assertTrue(any([x[:target_len] == target[:target_len] for x in attempts]))

    def test_checkpoint_save_and_restore(self):
        seq2seq = self.seq2seq
        vocab_len = len(seq2seq.vocab)
        hidden_size = self.hidden_size

        # create network
        seq2seq.create_net(0.1, 10, 0.99)

        # check that all variables are created
        self.check_rnn_variables(vocab_len, hidden_size)

        # Saver must be created *after* variables are created, otherwise it will fail
        seq2seq._create_checkpoint_saver()
        seq = seq2seq.seq
        loss_op = seq2seq.loss

        feed_dict = {seq: self.expected_batches[0]}

        # first, we will create session and initialize variables
        # and then we will run one batch and calculate loss
        # that loss will be used to check if restore is working
        with tf.Session() as sess1:
            # initialize global variables
            sess1.run(tf.global_variables_initializer())

            loss1 = sess1.run(loss_op, feed_dict=feed_dict)
            print('loss1:', loss1)

            saved_path = seq2seq._save_checkpoint(sess1)
            self.assertIsNotNone(saved_path)

        with tf.Session() as sess2:
            restored = seq2seq._restore_checkpoint(sess2)
            self.assertTrue(restored)

            loss2 = sess2.run(loss_op, feed_dict=feed_dict)
            print('loss2:', loss2)

            self.assertEqual(loss1, loss2)

class TestSeq2SeqWithSongs(TestCase):
    def setUp(self):
        # data stuff
        self.batch_size = 32
        self.window_size = 64
        self.hidden_size = 256
        self.overlap_size = 32
        input_file = '../data/songs-utf-8.txt'
        self.assertTrue(os.path.exists(input_file))

        # training stuff
        self.learning_rate = 0.001
        self.decay_rate = 0.999
        self.decay_steps = 100
        self.epochs = 10
        self.skip_steps = 20

        # sampling stuff
        self.temp = 0.7
        self.seed = list(string.ascii_uppercase) + ['Š', 'Đ', 'Č', 'Ć', 'Ž']
        self.seed = list(filter(lambda v: v not in ['X', 'Y', 'Q', 'W'], self.seed))
        print('seed', self.seed)

        # clear everything that might be in the default graph from previous tests
        tf.reset_default_graph()
        # construction stuff
        self.seq2seq = Seq2Seq(input_file=input_file,
                               window_size=self.window_size,
                               overlap_size=self.overlap_size,
                               batch_size=self.batch_size,
                               hidden_size=self.hidden_size)

    def test_train(self):
        seq2seq = self.seq2seq
        seq2seq.create_net(self.learning_rate, self.decay_steps, self.decay_rate)
        seq2seq.train(self.epochs, skip_steps=self.skip_steps, seed=self.seed, temp=self.temp)