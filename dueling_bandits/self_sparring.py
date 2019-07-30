from abstract_duel import AbstractDuel

import numpy as np
import logging
import argparse

def my_argmax(a):
    idx = np.nonzero(a == a.max())[0]
    return idx[np.random.randint(0, len(idx))]


class SelfSparring(AbstractDuel):
    def __init__(self, arms, arg=""):
        super(SelfSparring, self).__init__()
        self.arms = arms
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='Self-Sparring')
        parser.add_argument("--delta", type=float, default=3.0)
        parser.add_argument("--m", type=int, default=2, help='number of arms chosen per round')
        parser.add_argument("--alpha", type=float, default=0.51)
        parser.add_argument("--continue_sampling_experiment", type=str,
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        parser.add_argument("--random_seed", type=int, default=42)

        args = parser.parse_known_args(arg.split())[0]

        self.alpha = args.alpha
        self.n_arms = len(arms)
        self.w = np.ones(self.n_arms)
        self.l = np.ones(self.n_arms)
        self.theta = 0.5 * self.l
        self.t = 1
        self.iteration=0
        self.m = args.m
        self.delta = args.delta

        ### for random seed
        self.random_seed = args.random_seed
        self.prng = np.random.RandomState(self.random_seed)

    def ts_sample(self):
        self.theta = self.prng.beta(self.w, self.l)
        return my_argmax(self.theta)
        
    def get_arms(self):
        arms = []
        for i in range(self.m):
            arms.append(self.ts_sample())
        return arms

    def update_scores(self, winner, loser):
        if (self.t+1) % self.n_disp == 0:
            logging.info('SelfSparring, iteration: %d, winner: %d' % (self.t, self.get_winner()))
        if type(loser) is list:
            self.w[winner] += 1 * len(loser)
        else:
            self.w[winner] += 1
        self.l[loser] += 1
        self.t += 1
        self.iteration += 1
        
    def get_winner(self):
        stat_win = (self.w / (self.w + self.l) > 0.5)
        return np.argmax(stat_win)

  