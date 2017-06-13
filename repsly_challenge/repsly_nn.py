import tensorflow as tf
import numpy as np
import os
import time

class RepslyNN:
    def __init__(self):
        pass

    def get_num_of_variables(self):
        '''
        This is very useful for sanity checking. If you have a wrong idea of how many variables you are using,
        something is very wrong (with you or with the code).
        :return: number of *trainable* variables in the graph
        '''
        return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])

    ################################################################################################################
    #
    # THE FOLLOWING THREE METHODS SHOULD BE OVERRIDDEN IN SUBCLASSES
    #
    ################################################################################################################

    def _create_placeholders(self):
        pass

    def _create_model(self, arch):
        pass

    def _create_feed_dictionary(self, batch):
        pass

    ################################################################################################################
    #
    # THE USUAL STUFF
    #
    ################################################################################################################

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.labels = tf.one_hot(self.y, 2)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

    def _create_prediction(self):
        self.prediction = tf.argmax(self.logits, axis=1)

    def _calculate_f1_score(self):
        '''
        F1 score is used instead of accuracy in case of strongly biased classes. Google it up :)
        :return: F1 score, what else?!?
        '''
        logits = self.logits
        labels = self.labels
        tp = tf.reduce_sum(tf.to_int32(tf.logical_and(tf.equal(logits, labels), tf.equal(logits, 1))))
        fp = tf.reduce_sum(tf.to_int32(tf.logical_and(tf.not_equal(logits, labels), tf.equal(logits, 1))))
        fn = tf.reduce_sum(tf.to_int32(tf.logical_and(tf.not_equal(logits, labels), tf.equal(logits, 0))))

        precission = tp / (tp + fp)
        recall = tp / (tp + fn)

        self.f1_score = 2 * precission * recall / (precission + recall)

    def _create_optimizer(self):
        '''
        We use Adam optimizer, no need to experiment further.
        :return:
        '''
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

            lr = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                            global_step=self.global_step,
                                            decay_steps=self.decay_steps,
                                            decay_rate=self.decay_rate,
                                            name='learning_rate')

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, global_step=self.global_step)

    def create_net(self, arch, arch_dict, learning_rate=0.001, decay_steps=20, decay_rate=0.999):
        '''
        Creates neural network by calling all the functions in the right order.
        :param arch: data structure used by the _create_model(), typically the number and size of hidden layers
        :param arch_params: optional parameters used by _create_feed_dictionary(). E.g. arch_dict = {keep_prob: 0.9}.
        :param learning_rate:
        :param decay_steps:
        :param decay_rate:
        :return:
        '''
        # save for latter
        self.arch = arch
        self.arch_dict = arch_dict
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        # create network and do all the wiring
        # do not change the order because something might break (it will)
        self._create_placeholders()
        self._create_model(arch)
        self._create_loss()
        self._create_prediction()
        self._calculate_f1_score()
        self._create_optimizer()

        # create summary writters for train and validation sets
        self._create_summary_writers()

    def train(self, data, batch_size, epochs):
        '''
        Train network.
        :param data: data source
        :param epochs: number of epochs to train :)
        '''
        skip_steps = 20

        with tf.Session() as sess:
            # restore checkpoint if possible
            # if not, initialize variables and start from beginning
            self._create_checkpoint_saver()
            if not self._restore_checkpoint(sess):
                sess.run(tf.global_variables_initializer())

            start = time.time()
            for i in range(epochs):
                train_read_batch = data.read_batch(batch_size, 'train')
                validation_read_batch = data.read_batch(batch_size, 'validation', endless=True)
                for train_batch in train_read_batch:
                    train_feed_dict = self._create_feed_dictionary(train_batch)
                    # calculate current loss without updating variables
                    iteration, train_loss = sess.run([self.global_step, self.loss], feed_dict=train_feed_dict)
                    if iteration % skip_steps == 0:
                        # write train summary
                        self._add_summary(sess, train_feed_dict, 'train')

                        # calculate validation loss and write summary
                        validation_feed_dict = self._create_feed_dictionary(next(validation_read_batch))
                        validation_loss = self._add_summary(sess, validation_feed_dict, 'validation')

                        # save checkpoint
                        self._save_checkpoint(sess)

                        # printout losses
                        print('[{:05d}/{:.1f} sec]   train/validation loss = {:.5f}/{:.5f}'.\
                              format(iteration, time.time() - start, train_loss, validation_loss))

                    # finally, do the backpropagation and update the variables
                    sess.run(self.optimizer, feed_dict=train_feed_dict)

    ################################################################################################################
    #
    # SUMMARY STUFF
    #
    ################################################################################################################

    def _name_extension(self):
        desc = {type(self).__name__: str(self.arch)}
        desc.update(self.arch_dict)
        desc.update({
            'lr': str(self.learning_rate),
            'dr': str(self.decay_rate),
            'ds': str(self.decay_steps)})
        return os.path.join(*['{}-{}'.format(k, desc[k]).replace(" ", "") for k in desc.keys()])

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('f1_score', self.f1_score)
            self.summary = tf.summary.merge_all()

    def _create_summary_writers(self):
        self._create_summaries()
        graph = tf.get_default_graph()
        name_extension = self._name_extension()
        modes = ['train', 'validation']

        self.summary_writer = {mode: tf.summary.FileWriter(os.path.join('graphs', mode, name_extension), graph) for mode in modes}

    def _add_summary(self, sess, feed_dict, mode):
        loss, summary, global_step = sess.run([self.loss, self.summary, self.global_step], feed_dict=feed_dict)
        self.summary_writer[mode].add_summary(summary, global_step=global_step)
        return loss

    ################################################################################################################
    #
    # CHECKPOINT STUFF
    #
    ################################################################################################################

    def _create_checkpoint_saver(self):
        self.checkpoint_namebase = os.path.join('checkpoints', self._name_extension(), 'checkpoint')
        self.checkpoint_path = os.path.dirname(self.checkpoint_namebase)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        print('Checkpoint directory is:', os.path.abspath(self.checkpoint_path))

        print('Creating tf.train.Saver()...', end='')
        self.saver = tf.train.Saver()
        print('done')
        return self.saver

    def _save_checkpoint(self, sess):
        saver = self.saver

        saved_path = saver.save(sess, self.checkpoint_namebase, global_step=self.global_step)
        return saved_path

    def _restore_checkpoint(self, sess):
        saver = self.saver

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        print('self.checkpoint_path:', self.checkpoint_path)
        print('ckpt:', ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            print('ckpt.model_checkpoint_path:', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        return False


class RepslyFC(RepslyNN):
    def _create_placeholders(self):
        '''
        Creates placeholders for input and dropout parameters.
        :return:
        '''
        with tf.name_scope('input_data'):
            self.X = tf.placeholder(tf.float32, shape=[None, 241], name='X')
            self.y = tf.placeholder(tf.int32, shape=[None], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return self.X, self.y, self.keep_prob

    def _create_model(self, arch):
        '''
        Creates fully connected network.
        :param arch: list of hidden layer sizes
        '''
        with tf.name_scope('model'):
            h = self.X
            for hidden_size in arch:
                h = tf.contrib.layers.fully_connected(h, hidden_size)
                h = tf.nn.dropout(h, keep_prob=self.keep_prob)

            # linear classifier at the end
            self.logits = tf.contrib.layers.fully_connected(h, 2, activation_fn=None)

    def _create_feed_dictionary(self, batch):
        X, y = batch
        keep_prob = self.arch_dict['keep_prob']
        return {self.X: X, self.y: y, self.keep_prob: keep_prob}

