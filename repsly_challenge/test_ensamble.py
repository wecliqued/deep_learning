from unittest import TestCase

import numpy as np

from ensamble import Ensamble

from repsly_nn import RepslyFC
from repsly_data import RepslyData

class TestEnsamble(TestCase):
    def setUp(self):
        self.nets = [{'arch': {'batch_norm_decay': 0.95999999999999996,
                               'hidden_size': 184,
                               'input_keep_prob': 0.85,
                               'keep_prob': 0.8,
                               'no_of_layers': 5,
                               'use_batch_norm': True},
                      'data': None,
                      'global_step': 1099,
                      'learning_dict': {'decay_rate': 0.999,
                                        'decay_steps': 20,
                                        'learning_rate': 0.00020000000000000001},
                      'name': 'RepslyFC/no_of_layers-5/hidden_size-184/use_batch_norm-True/keep_prob-0.8/input_keep_prob-0.85/batch_norm_decay-0.96/lr-0.0002/dr-0.999/ds-20',
                      'net_cls': RepslyFC,
                      'stats': {'accuracy': 0.93093894675925926,
                                'f1_score': 0.63564603240548401,
                                'loss': 0.23691272735595703,
                                'precision': 0.90909090909090917,
                                'recall': 0.495}},
                     {'arch': {'batch_norm_decay': 0.97999999999999998,
                               'hidden_size': 67,
                               'input_keep_prob': 0.88,
                               'keep_prob': 0.72,
                               'no_of_layers': 6,
                               'use_batch_norm': True},
                      'data': None,
                      'global_step': 1099,
                      'learning_dict': {'decay_rate': 0.999,
                                        'decay_steps': 20,
                                        'learning_rate': 0.00029999999999999997},
                      'name': 'RepslyFC/no_of_layers-6/hidden_size-67/use_batch_norm-True/keep_prob-0.72/input_keep_prob-0.88/batch_norm_decay-0.98/lr-0.0003/dr-0.999/ds-20',
                      'net_cls': RepslyFC,
                      'stats': {'accuracy': 0.92235243055555549,
                                'f1_score': 0.57839721254355403,
                                'loss': 0.25847414880990982,
                                'precision': 0.8315412186379928,
                                'recall': 0.44582043343653249}},
                     {'arch': {'batch_norm_decay': 0.94999999999999996,
                               'hidden_size': 132,
                               'input_keep_prob': 0.92,
                               'keep_prob': 0.75,
                               'no_of_layers': 4,
                               'use_batch_norm': True},
                      'data': None,
                      'global_step': 1099,
                      'learning_dict': {'decay_rate': 0.999,
                                        'decay_steps': 20,
                                        'learning_rate': 0.0001},
                      'name': 'RepslyFC/no_of_layers-4/hidden_size-132/use_batch_norm-True/keep_prob-0.75/input_keep_prob-0.92/batch_norm_decay-0.95/lr-0.0001/dr-0.999/ds-20',
                      'net_cls': RepslyFC,
                      'stats': None}
                     ]

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

    def test_sample_range_int(self):
        for mode in ['lin', 'log']:
            for i in range(100):
                x = {mode: (1, 10000)}
                a = self.ens.sample(x)
                self.assertIsInstance(a, int)

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

    def test_nets_by_key_stat(self):
        ens = self.ens
        ens.nets = self.nets

        for key in ['f1_score', 'loss', 'accuracy', 'precision', 'recall']:
            for reverse in [True, False]:
                nets = ens.nets_by_key_stat(key=key, reverse=reverse)
                scores = [net['stats'][key] for net in nets]
                np.testing.assert_array_equal(sorted(scores, reverse=not reverse), scores)

    def test_untrained_nets(self):
        ens = self.ens
        ens.nets = self.nets

        nets = ens.untrained_nets()
        self.assertEqual(len(nets), 1)

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

        self.assertEqual(len(ens.nets_by_key_stat(key='f1_score')), 0)
        self.assertEqual(len(ens.untrained_nets()), 0)
        ens.add_nets(RepslyFC, arch=arch, data=data, learning_dict=learning_dict, no_of_nets=2)
        self.assertEqual(len(ens.nets_by_key_stat(key='f1_score')), 0)
        self.assertEqual(len(ens.untrained_nets()), 2)

        # you must read data before using it
        data.read_data('data/trial_users_analysis.csv', mode='FC')

        # train
        i = ens.train_all(train_dict)
        self.assertEqual(i, 2)
        self.assertEqual(len(ens.nets_by_key_stat(key='f1_score')), 2)
        self.assertEqual(len(ens.untrained_nets()), 0)
        ens.print_stat_by_key('f1_score')

        i = ens.train_untrained(train_dict)
        self.assertEqual(i, 0)
        self.assertEqual(len(ens.nets_by_key_stat(key='f1_score')), 2)
        self.assertEqual(len(ens.untrained_nets()), 0)

        i = ens.train_top_nets_by_key_stat('f1_score', 3, train_dict)
        self.assertEqual(i, 2)
        self.assertEqual(len(ens.nets_by_key_stat(key='f1_score')), 2)
        self.assertEqual(len(ens.untrained_nets()), 0)

        i = ens.train_top_nets_by_key_stat('f1_score', -1, train_dict)
        self.assertEqual(i, 1)
        self.assertEqual(len(ens.nets_by_key_stat(key='f1_score')), 2)
        self.assertEqual(len(ens.untrained_nets()), 0)

