from sparring import Sparring
import numpy as np
import logging
import argparse


def my_argmax(ay):
    idx = np.nonzero(ay == ay.max())[0]
    return idx[np.random.randint(0, idx.shape[0])]


class SparringEXP3(Sparring):
    def __init__(self, arms, arg=""):
        super(SparringEXP3, self).__init__(arms=arms, arg=arg)
        self.arms = arms
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='SparringEXP3')
        parser.add_argument("--alpha", type=float, default=-1, help='gamma in EXP3, if -1, we use adaptive_gamma')
        parser.add_argument("--continue_sampling_experiment", type=str,
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        parser.add_argument("--random_seed", type=int, default=42)
        args = parser.parse_known_args(arg.split())[0]

        self.name = args.sampler
        ### for random seed
        self.random_seed = args.random_seed
        self.prng = np.random.RandomState(self.random_seed)

        self.w_l = np.ones(self.n_arms)
        self.w_r = np.ones(self.n_arms)
        self.ucb_l = np.ones(self.n_arms) / self.n_arms
        self.ucb_r = np.ones(self.n_arms) / self.n_arms
        self.k = self.n_arms + 0.  # self.n_arms
        self.gamma = self.adaptive_gamma() if self.alpha == -1 else self.alpha

    def adaptive_gamma(self):
        t = max(self.t, 1e4)
        return min(1, np.sqrt(self.k*np.log(self.k)/t))

    def sampling_distribution(self):
        self.gamma = self.adaptive_gamma() if self.alpha == -1 else self.alpha
        self.gamma += 0.0
        ucb_l = self.w_l / np.sum(self.w_l)
        self.ucb_l = (1 - self.gamma) * ucb_l + self.gamma / self.k
        ucb_r = self.w_r / np.sum(self.w_r)
        self.ucb_r = (1 - self.gamma) * ucb_r + self.gamma / self.k

    def get_ucb(self):
        self.sampling_distribution()

    def get_arms(self):
        self.get_ucb()
        arm_l = self.prng.choice(self.n_arms, 1, replace=False, p=self.ucb_l)[0]
        arm_r = self.prng.choice(self.n_arms, 1, replace=False, p=self.ucb_r)[0]
        self.arm_l = arm_l
        self.arm_r = arm_r
        return arm_l, arm_r

    def update_scores(self, winner, loser):
        self.times_l[self.arm_l] += 1
        self.times_r[self.arm_r] += 1
        if winner == loser:
            if 0.5 < self.prng.rand():
                self.w_l[winner] = self.w_l[winner] * np.exp(self.gamma / self.ucb_l[winner] / self.k)
            else:
                self.w_r[winner] = self.w_r[winner] * np.exp(self.gamma / self.ucb_r[winner] / self.k)
        else:
            if winner == self.arm_l:
                self.w_l[winner] = self.w_l[winner] * np.exp(self.gamma / self.ucb_l[winner] / self.k)
            else:
                self.w_r[winner] = self.w_r[winner] * np.exp(self.gamma / self.ucb_r[winner] / self.k)
        self.t += 1
        self.iteration += 1
        if (self.t+1) % self.n_disp == 0:
            logging.info('%s, iteration: %d, potential winner is %d.' % (self.name, self.t, self.get_winner()))





