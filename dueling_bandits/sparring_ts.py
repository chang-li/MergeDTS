from abstract_duel import AbstractDuel

import numpy as np
import logging
import argparse

def my_argmax(ay):
    idx = np.nonzero(ay == ay.max())[0]
    return idx[np.random.randint(0, idx.shape[0])]


class SparringTS(AbstractDuel):
    def __init__(self, arms, arg=""):
        super(SparringTS, self).__init__()
        self.arms = arms
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='SparringTS')
        parser.add_argument("--alpha", type=float, default=0.51)
        parser.add_argument("--continue_sampling_experiment", type=str,
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        parser.add_argument("--random_seed", type=int, default=42)

        args = parser.parse_known_args(arg.split())[0]

        ### for random seed
        self.random_seed = args.random_seed
        self.prng = np.random.RandomState(self.random_seed)

        self.alpha = args.alpha
        self.n_arms = len(arms)
        self.loss_l = np.ones(self.n_arms)
        self.loss_r = np.ones(self.n_arms)
        self.win_l = np.ones(self.n_arms)
        self.win_r = np.ones(self.n_arms)
        self.theta_l = 0.5 * np.ones(self.n_arms)
        self.theta_r = 0.5 * np.ones(self.n_arms)
        self.t = 1
        self.iteration = 0
        self.arm_l = 0
        self.arm_r = 0

    def get_ucb(self):
        self.theta_l = self.prng.beta(self.win_l, self.loss_l)
        self.theta_r = self.prng.beta(self.win_r, self.loss_r)


    def get_arms(self):
        self.get_ucb()
        arm_l = my_argmax(self.theta_l)
        arm_r = my_argmax(self.theta_r)
        self.arm_r = arm_r
        self.arm_l = arm_l
        return arm_l, arm_r

    def update_scores(self, winner, loser):
        if winner == loser:
            if self.prng.rand() <= 0.5:
                self.win_r[winner] += 1
                self.loss_l[loser] += 1
            else:
                self.win_l[winner] += 1
                self.loss_r[loser] += 1
        elif winner == self.arm_l:
            self.win_l[winner] += 1
            self.loss_r[loser] += 1
        else:
            self.win_r[winner] += 1
            self.loss_l[loser] += 1
        self.t += 1
        self.iteration += 1
        if self.t % 100000 == 0:
            logging.info('SparringTS, iteration: %d, potential winner is %d.' %(self.t, self.get_winner()))

    def get_winner(self):
        mean_l = self.win_l / (self.win_l + self.loss_l)
        mean_r = self.win_r / (self.win_r + self.loss_r)
        return my_argmax(mean_l) if mean_l.max() >= mean_r.max() else my_argmax(mean_r)

