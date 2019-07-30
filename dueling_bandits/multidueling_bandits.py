from abstract_duel import AbstractDuel

import numpy as np
import logging
import argparse


def my_argmax(a):
    idx = np.nonzero(a == a.max())[0]
    return idx[np.random.randint(0, len(idx))]


class MultiDuelingBandit(AbstractDuel):

    def __init__(self, arms, arg=""):
        super(MultiDuelingBandit, self).__init__()
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='MultiDueling Bandit')
        parser.add_argument("--alpha", type=float, default=0.51)
        parser.add_argument("--random_seed", type=int, default=42)
        args = parser.parse_known_args(arg.split())[0]
        self.sampler = args.sampler
        self.alpha = args.alpha
        self.arms = arms

        ### for random seed
        self.random_seed = args.random_seed
        self.prng = np.random.RandomState(self.random_seed)

        self.n_arms = len(arms)
        self.w = np.ones((self.n_arms, self.n_arms))
        self.times = self.w + self.w.T
        self.t = 1
        self.ucb = np.ones(self.n_arms)
        self.lcb = np.ones(self.n_arms)

    def sample_tournament(self):
        self.ucb = self.w / self.times + np.sqrt(self.alpha * np.log(self.t) / self.times)
        np.fill_diagonal(self.ucb, 0.5)
        ucb = np.prod(self.ucb > 0.5, axis=1)
        arms = np.where(ucb)[0]
        if len(arms) == 0:
            #return np.arange(self.n_arms)
            return self.prng.choice(self.n_arms, 100)
        else:
            if len(arms) > 100:
                return arms
            return arms

    def get_arms(self):
        return self.sample_tournament()

    def update_scores(self, arms, pref):
        if (self.t+1) % self.n_disp == 0:
            logging.info('SOSM, iteration: %d, winner: %d' % (self.t, self.get_winner()))
        pref = np.sign(pref)
        pref[pref < 0] = 0
        self.w[np.ix_(arms, arms)] += pref
        self.t += 1

    def get_winner(self):
        stat_win = (self.w / (self.w + self.w.T) > 0.5)
        return np.argmax(stat_win.sum(axis=1))

