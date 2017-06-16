from unittest import TestCase

import numpy as np

from ensamble import Ensamble

from repsly_nn import RepslyFC
from repsly_data import RepslyData

class TestEnsamble(TestCase):
    def setUp(self):

        self.ens = Ensamble()

    def test_sample_range_lin(self):
        x = {'lin' : (-3, 7)}
        a = np.array([self.ens.sample(x) for i in range(10000)])
        np.testing.assert_approx_equal(np.mean(a), np.mean(x['lin']), significant=2)
        np.testing.assert_approx_equal(np.median(a), np.mean(x['lin']), significant=2)

    def test_sample_range_log(self):
        x = {'log' : (0.01, 10000)}
        a = np.array([self.ens.sample(x) for i in range(10000)])
        np.testing.assert_approx_equal(np.median(a), np.exp(np.mean(np.log(x['log']))), significant=1)

    def test_sample_range_inv_log(self):
        x = {'inv-log' : (0.9, 0.999)}
        a = np.array([self.ens.sample(x) for i in range(10000)])
        np.testing.assert_approx_equal(np.median(a), 1.0 - np.exp(np.mean(np.log([1.0 - x['inv-log'][0], 1.0 - x['inv-log'][1]]))), significant=3)

    def test_sample_range_bool(self):
        x = 'bool'
        a = np.array([self.ens.sample(x) for i in range(10000)])
        np.testing.assert_approx_equal(np.mean(a), 0.5, significant=2)

    def test_sample_arch(self):
        arch = {
                'no_of_layers': {'lin': (3, 6)},
                'hidden_size': {'lin': (64, 512)},
                'use_batch_norm': 'bool',
                'keep_prob': {'lin': (0.4, 0.9)},
                'input_keep_prob': {'lin': (0.8, 1.0)},
                'batch_norm_decay': {'inv-log': (0.9, 0.99)},
        }
        for _ in range(10000):
            sampled_arch = self.ens.sample(arch)
            self.assertIsInstance(sampled_arch, dict)
            self.assertIn(sampled_arch['no_of_layers'], range(3, 6+1))
            self.assertIn(sampled_arch['hidden_size'], range(64, 512+1))
            self.assertIsInstance(sampled_arch['use_batch_norm'], bool)
            self.assertGreaterEqual(sampled_arch['keep_prob'], 0.4)
            self.assertLessEqual(sampled_arch['keep_prob'], 0.9)
            self.assertGreaterEqual(sampled_arch['input_keep_prob'], 0.8)
            self.assertLessEqual(sampled_arch['input_keep_prob'], 1.0)
            self.assertGreaterEqual(sampled_arch['batch_norm_decay'], 0.9)
            self.assertLessEqual(sampled_arch['batch_norm_decay'], 0.99)

    def test_add_and_train_all_nets(self):
        arch = {
                'no_of_layers': {'lin': (3, 6)},
                'hidden_size': {'lin': (32, 256)},
                'use_batch_norm': True,
                'keep_prob': {'lin': (0.4, 1.0, 2)},
                'input_keep_prob': {'lin': (0.8, 1.0, 2)},
                'batch_norm_decay': {'inv-log': (0.9, 0.99, 2)},
        }
        learning_dict = {
            'learning_rate': 0.001,
            'decay_steps': 20,
            'decay_rate': 0.999
        }
        train_dict = {
            'batch_size': 256,
            'epochs': 1,
            'skip_steps': 20
        }
        ens = self.ens
        data = RepslyData()

        ens.add_nets(RepslyFC, arch=arch, data=data, learning_dict=learning_dict, no_of_nets=2)
        print(ens.nets)

        # you must read data before using it
        data.read_data('data/trial_users_analysis.csv', mode='FC')

        # train
        ens.train_all(train_dict)
