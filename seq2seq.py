import string
import tensorflow as tf
import numpy as np

import os
import time

from tempfile import TemporaryFile

class Seq2Seq:

    def __init__(self, input_file, window_size, overlap_size, batch_size, hidden_size, no_stacked_cells=3, vocab=None):
        if vocab == None:
            vocab = string.ascii_letters + string.digits + string.punctuation + string.whitespace + 'ČĆŽĐŠ' + 'ćčžđš'
        self.vocab = vocab

        self.input_file_name = input_file
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.no_stacked_cells = no_stacked_cells

        # checkpoint stuff
        self.checkpoint_base_path = 'checkpoints/'

        # summary stuff
        self.summary_train_path = 'graphs/train'

    def vocab_encode(self, text):
        return [self.vocab.index(x)+1 if x in self.vocab else self.vocab.index('*')+1 for x in text]

    def vocab_decode(self, seq):
        text_array = [self.vocab[i-1] for i in seq if i>0]
        return ''.join(text_array)

    def read_data(self, input=None, shuffle=True):
        # if input is not provided, we will open a file and read all data into it
        # this should work for "small" datasets - 100MB is small :)
        if not input:
            with open(self.input_file_name) as f:
                input = f.read()

        seq = self.vocab_encode(input)
        for i in range(0, len(seq), self.window_size - self.overlap_size):
            if shuffle:
                j = np.random.random_integers(0, len(seq))
            else:
                j = i
            chunk = seq[j:j+self.window_size]
            chunk += [0] * (self.window_size-len(chunk))
            yield chunk

    def read_batch(self, stream):
        batch = []
        for x in stream:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def _seq2one_hot(self, seq):
        """
        Encodes input tensor into one hot tensor of depth len(self.vocab) + 1. The size of vocabulary is increased by
        one because we will pad shorter sequences with zeros.  
        
        :param seq: tensor (should be placeholder for input batch in our case) 
        :return: one hot tensor of depth len(vocab)+1
        """
        # we will use 0 for padding so our vocabulary size will increase by one
        vocab_len = len(self.vocab)
        one_hot_seq = tf.one_hot(seq, depth=vocab_len+1)
        return one_hot_seq

    def _lenghts(self, seq):
        """
        
        :param seq: one hot encoding  
        :return: sequence of batch_size where elements are lengths of the sequence in the batch
        """
        return tf.reduce_sum(tf.sign(seq), 1)

    def _create_cell(self, seq, no_stacked_cells):
        """
        Creates GRU cell
        :param seq: placeholder of the input batch
        :return: cell and placeholder for its internal state
        """
        batch_size = tf.shape(seq)[0]

        ##########################################################################################################
        #
        # TODO: Create a stacked MultiRNNCell from GRU cells
        #       First, you have to use tf.contrib.rnn.GRUCell() to construct cells
        #       Since around May 2017, there is new way of constructing MultiRNNCell and you need to create
        #       one cell for each layer. Old code snippets that used [cell * no_stacked_cells] that you can
        #       find online might not work with the latest Tensorflow
        #
        #       After construction GRUCell objects, use it to construct tf.contrib.rnn.MultiRNNCell().
        #
        # YOUR CODE BEGIN
        #
        ##########################################################################################################

        cell = None # you

        ##########################################################################################################
        #
        # YOUR CODE END
        #
        ##########################################################################################################

        multi_cell_zero_state = cell.zero_state(batch_size, tf.float32)
        in_state_shape = tuple([None, self.hidden_size] for _ in range(no_stacked_cells))
        in_state = tuple(tf.placeholder_with_default(cell_zero_state, [None, self.hidden_size], name='in_state') for cell_zero_state in multi_cell_zero_state)

        return cell, in_state

    def _create_rnn(self):
        with tf.name_scope('RNN_cell'):
            seq = self.seq
            self.cell, self.in_state = self._create_cell(seq, self.no_stacked_cells)

            self.lenghts = self._lenghts(seq)
            self.one_hot_seq = self._seq2one_hot(seq)

            ##########################################################################################################
            #
            # TODO: Create a dynamically unrolled RNN from previously created stacked GRU cell
            #
            #       First, we created a stacked GRU cell.
            #       Next step was to get a actual length of input sequence and then encode it into one hot vector.
            #
            #       Now, you have to use tf.nn.dynamic_rnn to create dynamically unrolled RNN.
            #
            # YOUR CODE BEGIN
            #
            ##########################################################################################################

            self.output, self.out_state = None

            ##########################################################################################################
            #
            # YOUR CODE END
            #
            ##########################################################################################################

            return self.output, self.in_state, self.out_state

    def _create_placeholders(self, batch_size=None, window_size=None):

        ##########################################################################################################
        #
        # TODO: Create a placeholder for input data and for sampling temperature
        #
        #       Input data shape is [batch_size, window_size].
        #       Temperature is scalar of tf.float32 type
        #
        #       Use tf.placeholder() to create them
        #
        # YOUR CODE BEGIN
        #
        ##########################################################################################################

        with tf.name_scope('input_data'):
            self.seq = None
        with tf.name_scope('model_params'):
            self.temp = None

        ##########################################################################################################
        #
        # YOUR CODE END
        #
        ##########################################################################################################

        return self.seq, self.temp

    def _create_model(self):
        depth = len(self.vocab)+1
        self._create_rnn()

        ##########################################################################################################
        #
        # TODO: calculate loss
        #
        #       First, you have to caluculate logits. Use tf.contrib.layers.fully_connected() to create a fully
        #       connected layer together with its variables.
        #
        #       Then you can calculate labels by one hot encoding input sequence.
        #
        #       Finally, use tf.nn.softmax_cross_entropy_with_logits and tf.reduce_mean to calculate loss.
        #       Remember to slide labels by one position to match predicted logits.
        #
        # YOUR CODE BEGIN
        #
        ##########################################################################################################

        with tf.name_scope('loss'):
            self.logits = None
            self.labels = None
            self.loss = None

        ##########################################################################################################
        #
        # YOUR CODE END
        #
        ##########################################################################################################

        ##########################################################################################################
        #
        # TODO: Given logits, sample the next character
        #
        #       Hint: Use tf.multinomial() function.
        #
        # YOUR CODE BEGIN
        #
        ##########################################################################################################

        with tf.name_scope('sample'):
            self.sample = tf.multinomial(self.logits[:, -1] / self.temp, 1)[:, 0]

        ##########################################################################################################
        #
        # YOUR CODE END
        #
        ##########################################################################################################

        return self.loss, self.sample, self.in_state, self.out_state

    def _create_optimizer(self, learning_rate, decay_steps, decay_rate):
        # we'll use this for logging and tensorboard naming
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        ##########################################################################################################
        #
        # TODO: Create Adam optimizer with decaying learning rate
        #
        #       First create a global step and then use it to create decaying learning rate with
        #       tf.train.exponential_decay(). Finally, create AdamOptimizer.
        #
        # YOUR CODE BEGIN
        #
        ##########################################################################################################

        with tf.name_scope('optimizer'):
            # global step is needed for logging and tensorboard
            self.global_step = None

            # first we create a decaying learning rate tensor
            self.lr = None

            # and then we create optimizer (Adam is the default choice)
            self.optimizer = None

        ##########################################################################################################
        #
        # YOUR CODE END
        #
        ##########################################################################################################

        return self.optimizer, self.global_step

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('lr', self.lr)
            self.summary = tf.summary.merge_all()
        return self.summary

    def online_inference(self, sess, len, temp, seed=None):
        next_input = self.vocab_encode([seed])
        next_state = None
        sampled_sentence = seed

        for _ in range(len-1):
            feed_dict = {self.seq: [next_input], self.temp: temp}
            if next_state is not None:
                feed_dict[self.in_state] = next_state

            next_input, next_state = sess.run([self.sample, self.out_state], feed_dict=feed_dict)
            sampled_sentence += self.vocab_decode(next_input)[:1]
        return sampled_sentence

    def create_net(self, learning_rate, decay_steps, decay_rate):
        """
        Creates complete graph and everything that can without tf.Session.
        :param learning_rate: 
        :param decay_steps: 
        :param decay_rate: 
        :return: 
        """
        self._create_placeholders()
        self._create_model()
        self._create_optimizer(learning_rate, decay_steps, decay_rate)
        self._create_summaries()
        self._create_summary_writer()

    def train(self, epochs, skip_steps, seed, temp):
        with tf.Session() as sess:
            # restore checkpoint if possible
            # if not, initialize variables and start from beginning
            self._create_checkpoint_saver()
            if not self._restore_checkpoint(sess):
                sess.run(tf.global_variables_initializer())

            start = time.time()
            for i in range(epochs):
                for batch in self.read_batch(self.read_data()):
                    feed_dict = {
                        self.seq: batch
                    }
                    batch_loss, _, iteration = sess.run([self.loss, self.optimizer, self.global_step], feed_dict=feed_dict)
                    if iteration % skip_steps == 1:
                        # write summaries and save checkpoint
                        print('#' * 64)
                        print('[step={0:04d}] loss = {1:.1f}   time = {2:.1f} sec'.format(iteration-1, batch_loss, time.time() - start))
                        self._save_checkpoint(sess)
                        self._add_summary(sess, feed_dict)
                        # make inference just for fun
                        sample = self.online_inference(sess,
                                                       len=self.window_size * 4,
                                                       temp=temp,
                                                       seed = np.random.choice(seed))
                        print('sample: {}'.format(sample))

    # Boring stuff bellow :)

    # used to generate names for checkpoints (needs refactoring)
    def _model_extension(self):
        dataset_name = os.path.basename(self.input_file_name)
        return '-{}'.format(dataset_name)

    def _arch_extension(self):
        return '-stacked_layers{}-hidden_size{}-window_size{}-overlap_size{}'.format(
            self.no_stacked_cells, self.hidden_size, self.window_size, self.overlap_size
        )

    def _train_extension(self):
        return '-lr{}-dr{}-ds{}'.format(self.learning_rate, self.decay_rate, self.decay_steps)

    def _name_extension(self):
        return self._model_extension() + self._arch_extension() + self._train_extension()

    def _create_checkpoint_saver(self):
        self.checkpoint_namebase = os.path.join(self.checkpoint_base_path, 'seq2seq{}/checkpoint'.format(self._name_extension()))
        self.checkpoint_path = os.path.dirname(self.checkpoint_namebase)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        print('Checkpoint directory is:', os.path.abspath(self.checkpoint_path))

        self.saver = tf.train.Saver()
        return self.saver

    def _save_checkpoint(self, sess):
        saver = self.saver

        saved_path = saver.save(sess, self.checkpoint_namebase, global_step=self.global_step)
        return saved_path

    def _restore_checkpoint(self, sess):
        saver = self.saver

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        return False

    def _create_summary_writer(self):
        graph = tf.get_default_graph()
        summary_dir = os.path.join(self.summary_train_path, 'graph{}'.format(self._name_extension()))
        self.writer = tf.summary.FileWriter(summary_dir, graph)

    def _add_summary(self, sess, feed_dict):
        summary, global_step = sess.run([self.summary, self.global_step], feed_dict=feed_dict)
        self.writer.add_summary(summary, global_step=global_step)


