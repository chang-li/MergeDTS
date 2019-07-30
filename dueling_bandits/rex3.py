from abstract_duel import AbstractDuel
import numpy as np
import logging
import argparse


def my_argmax(ay):
    idx = np.nonzero(ay == ay.max())[0]
    return idx[np.random.randint(0, idx.shape[0])]


class Rex3(AbstractDuel):
    def __init__(self, arms, arg=""):
        super(Rex3, self).__init__( )
        self.arms = arms
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='Rex3')
        parser.add_argument("--alpha", type=float, default=-1, help='gamma in Rex3, if -1, we use adaptive_gamma')
        parser.add_argument("--continue_sampling_experiment", type=str,
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        parser.add_argument("--random_seed", type=int, default=42)
        args = parser.parse_known_args(arg.split())[0]

        self.name = args.sampler
        self.alpha = args.alpha
        self.random_seed = args.random_seed
        self.args = args
        self.n_arms = len(arms)
        self.k = self.n_arms + 0.  # self.n_arms

        self.t = 1.
        self.iteration = 0

        self.arm_l = 0
        self.arm_r = 0

        ### for random seed
        self.random_seed = args.random_seed
        self.prng = np.random.RandomState(self.random_seed)

        self.w = np.ones(self.n_arms)
        self.n = np.ones(self.n_arms)

        self.prob = np.ones(self.n_arms) / self.n_arms
        self.gamma = self.adaptive_gamma() if self.alpha == -1 else self.alpha

    def adaptive_gamma(self):
        t = max(self.t, 1e4)
        return min(0.5, np.sqrt(self.k*np.log(self.k)/t))

    def sampling_distribution(self):
        self.gamma = self.adaptive_gamma() if self.alpha == -1 else self.alpha
        self.gamma += 0.0
        ucb = (self.w + 1e-100) / np.sum(self.w + 1e-100)  # 1e-100 for the precision
        self.prob = (1-self.gamma) * ucb + self.gamma / self.k

    def get_ucb(self):
        self.sampling_distribution()

    def get_arms(self):
        self.get_ucb()
        arms = self.prng.choice(self.n_arms, 2, replace=True, p=self.prob)
        self.arm_l = arms[0]
        self.arm_r = arms[1]
        return arms[0], arms[1]

    def update_scores(self, winner, loser):
        self.n[self.arm_l] += 1
        self.n[self.arm_r] += 1
        if winner != loser:
            self.w[winner] = self.w[winner] * np.exp(self.gamma / self.prob[winner] / self.k)
            self.w[loser] = self.w[loser] * np.exp(-self.gamma / self.prob[loser] / self.k)
        self.t += 1
        self.iteration += 1
        if (self.t+1) % self.n_disp == 0:
            logging.info('%s, iteration: %d, potential winner is %d.' % (self.name, self.t, self.get_winner()))

    def get_winner(self):
        mean = self.w / self.n
        return my_argmax(mean)



