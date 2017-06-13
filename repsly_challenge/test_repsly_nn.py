from unittest import TestCase
import tensorflow as tf
import numpy as np

from repsly_nn import RepslyFC
from repsly_data import RepslyData


class TestRepslyFC(TestCase):
    def setUp(self):
        self.repsly_fc = RepslyFC()
        self.arch = [100, 200]
        self.arch_dict = {'keep_prob': 0.9}
        np.random.seed(0)
        self.X = np.random.randint(0-4, 4, [3, 241])
        self.y = np.random.randint(0, 1, [3])
        self.keep_prob = 0.9

        ten_users_file = 'data/trial_ten_users.csv'
        self.data = RepslyData()
        self.data.read_data(ten_users_file, 'FC')
        self.batch_size = 1

    def test__create_placeholders(self):
        repsly_nn = self.repsly_fc

        X, y, keep_prob = repsly_nn._create_placeholders()

    def test__create_model(self):
        repsly_nn = self.repsly_fc
        arch = self.arch

        X, y, keep_prob = repsly_nn._create_placeholders()

        # one of the easiest sanity checks is the number of variables created
        self.assertEqual(repsly_nn.get_num_of_variables(), 0)
        logits = repsly_nn._create_model(arch)
        self.assertEqual(repsly_nn.get_num_of_variables(), (241+1)*100+(100+1)*200+(200+1)*2)

    def test_create_net(self):
        repsly_nn = self.repsly_fc
        arch = self.arch
        arch_dict = self.arch_dict

        X, y, keep_prob = repsly_nn.create_net(arch, arch_dict)

        print('_name_extension():', repsly_nn._name_extension())

        # todo: finish test :)
        self.fail()

    def test_train(self):
        repsly_nn = self.repsly_fc
        arch = self.arch
        arch_dict = self.arch_dict
        data = self.data
        batch_size = self.batch_size

        X, y, keep_prob = repsly_nn.create_net(arch, arch_dict)
        repsly_nn.train(data, batch_size, epochs=1)

        # todo: finish test :)
        self.fail()
