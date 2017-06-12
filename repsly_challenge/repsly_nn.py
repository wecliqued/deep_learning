import tensorflow as tf
import numpy as np
import os

class RepslyNN:
    def __init__(self):
        pass

    def get_num_of_variables(self):
        return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])

    ################################################################################################################
    #
    # THE FOLLOWING TWO CLASSES SHOULD BE OVERRIDDEN IN SUBCLASSES
    #
    ################################################################################################################

    def _create_placeholders(self):
        pass

    def _create_model(self, arch):
        pass

    ################################################################################################################
    #
    # THE USUAL STUFF
    #
    ################################################################################################################

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))

    def _create_prediction(self):
        self.prediction = tf.argmax(self.logits, axis=1)

    def _calculate_f1_score(self):
        logits = self.logits
        labels = self.y
        tp = tf.reduce_sum(tf.to_int32(tf.logical_and(tf.equal(logits, labels), tf.equal(logits, 1))))
        fp = tf.reduce_sum(tf.to_int32(tf.logical_and(tf.not_equal(logits, labels), tf.equal(logits, 1))))
        fn = tf.reduce_sum(tf.to_int32(tf.logical_and(tf.not_equal(logits, labels), tf.equal(logits, 0))))

        precission = tp / (tp + fp)
        recall = tp / (tp + fn)

        self.f1_score = 2 * precission * recall / (precission + recall)

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

            lr = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                            global_step=self.global_step,
                                            decay_steps=self.decay_steps,
                                            decay_rate=self.decay_rate,
                                            name='learning_rate')

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, global_step=self.global_step)

    def create_net(self, arch, learning_rate=0.001, decay_steps=20, decay_rate=0.999):
        '''
        Creates neural network.
        :param arch: data structure used by the _create_model(), typically the number and size of hidden layers
        :param learning_rate:
        :param decay_steps:
        :param decay_rate:
        :return:
        '''
        # save for latter
        self.arch = arch
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        # create network and do all the wiring
        # do not change the order because everything will break
        placeholders = self._create_placeholders()
        self._create_model(arch)
        self._create_loss()
        self._create_prediction()
        self._calculate_f1_score()
        self._create_optimizer()

        # create summary writters for train and validation sets
        self._create_summary_writers()

        # placeholders are needed for feeding the data into train()
        return placeholders

    def train(self, data, epochs):
        pass

    ################################################################################################################
    #
    # SUMMARY STUFF
    #
    ################################################################################################################

    def _name_extension(self):
        desc = {'type': type(self).__name__,
                'arch': str(self.arch),
                'lr': str(self.learning_rate),
                'dr': str(self.decay_rate),
                'ds': str(self.decay_steps)
                }
        return os.path.join(*['{}-{}'.format(k, desc[k]) for k in desc.keys()])

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('f1_score', self.f1_score)
            self.summary = tf.summary.merge_all()

    def _create_summary_writers(self):
        self._create_summaries()
        graph = tf.get_default_graph()
        train_summary_dir = os.path.join('graph', 'train', self._name_extension())
        validation_summary_dir = os.path.join('graph', 'validation', self._name_extension())

        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph)
        self.validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, graph)

    ################################################################################################################
    #
    # CHECKPOINT STUFF
    #
    ################################################################################################################


class RepslyFC(RepslyNN):
    def _create_placeholders(self):
        with tf.name_scope('input_data'):
            self.X = tf.placeholder(tf.float32, shape=[None, 241], name='X')
            self.y = tf.placeholder(tf.float32, shape=[None], name='y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return self.X, self.y, self.keep_prob

    def _create_model(self, arch):
        '''
        Default implementation of _create_model() will just create fully connected network. You should override
        this method together with _create_placeholders().
        :param arch:
        :return:
        '''
        with tf.name_scope('model'):
            h = self.X
            for hidden_size in arch:
                h = tf.contrib.layers.fully_connected(h, hidden_size)
            h = tf.nn.dropout(h, keep_prob=self.keep_prob)

            # linear classifier at the end
            self.logits = tf.contrib.layers.fully_connected(h, 2, activation_fn=None)

