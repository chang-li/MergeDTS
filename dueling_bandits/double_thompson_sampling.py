from abstract_duel import AbstractDuel

import numpy as np
import logging
import argparse


def my_argmax(a):
    idx = np.nonzero(a == a.max())[0]
    return idx[np.random.randint(0, len(idx))]

class FastBeta:
    def __init__(self, W, depth=100, seed=42):
        # W - a square matrix containing a score sheet
        self.depth = depth
        self.shape = W.shape
        self.W = np.maximum(W, np.ones(W.shape))
        self.prng = np.random.RandomState(seed)
        self.allSamples = np.zeros(self.shape+(depth,))
        self.sampleAllBeta()
        self.depthIndex = 0
        self.upper_or_lower = 'UPPER'

    def sampleAllBeta(self):
        self.allSamples = 0.5*np.ones(self.shape+(self.depth,))
        for i in range(self.depth):
            self.allSamples[:, :, i] = self.prng.beta(self.W, self.W.T)
            np.fill_diagonal(self.allSamples[:, :, i], 0.5)

    def update(self,r,c,w):
        self.W[r,c] = w
        self.allSamples[r,c,:] = self.prng.beta(self.W[r,c],self.W[c,r],self.depth)
        self.allSamples[c,r,:] = self.prng.beta(self.W[c,r],self.W[r,c],self.depth)

    def getSamples(self):
        if self.upper_or_lower == 'UPPER':
            if self.depthIndex > self.depth-1:
                self.depthIndex = 0
                self.sampleAllBeta()
            self.upper_or_lower = 'LOWER'
            return np.triu(self.allSamples[:,:,self.depthIndex])
        elif self.upper_or_lower == 'LOWER':
            self.depthIndex = self.depthIndex + 1
            self.upper_or_lower = 'UPPER'
            return np.tril(self.allSamples[:,:,self.depthIndex-1])


class DoubleThompsonSampling(AbstractDuel):

    def __init__(self, arms, arg=""):
        super(DoubleThompsonSampling, self).__init__()
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='Double Thompson Sampling')
        parser.add_argument("--alpha", type=float, default=0.51)
        parser.add_argument("--continue_sampling_experiment", type=str,
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
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
        self.theta = 0.5 * np.ones((self.n_arms, self.n_arms))
        self.t = 1
        self.ucb = np.ones(self.n_arms)
        self.lcb = np.ones(self.n_arms)
        self.beta = FastBeta(self.w, seed=self.prng.randint(1000))

    def sample_tournament(self):
        self.ucb = self.w / self.times + np.sqrt(self.alpha * np.log(self.t) / self.times)
        np.fill_diagonal(self.ucb, 0.5)
        cope_scores = (self.ucb > 0.5).sum(axis=1)
        wins = np.where(cope_scores == cope_scores.max())[0]
        if len(wins) == 1: 
            return wins[0]
        self.theta = self.beta.getSamples() 
        win_theta = (self.theta > 0.5) + ((self.theta > 0) & (self.theta < 0.5)).T
        cope_wins = win_theta.sum(axis=1)
        return wins[my_argmax(cope_wins[wins])]

    def relative_sample(self, arm_c):
        self.lcb = self.w[:, arm_c]/ self.times[:, arm_c] \
                   - np.sqrt(self.alpha * np.log(self.t)/self.times[:, arm_c])
        self.lcb[arm_c] = 0.5
        lcb_bool = (self.lcb <= 0.5)
        idx = np.where(lcb_bool)[0]
        rel_theta = self.prng.beta(self.w[idx, arm_c], self.w[arm_c, idx])
        rel_theta[np.sum(lcb_bool[0:arm_c])] = 0.5
        return idx[my_argmax(rel_theta)]

    def get_arms(self):
        arm_c = self.sample_tournament()
        arm_d = self.relative_sample(arm_c)
        return arm_c, arm_d

    def update_scores(self, winner, loser):
        if (self.t+1) % self.n_disp == 0:
            logging.info('DTS, iteration: %d, winner: %d' % (self.t, self.get_winner()))
        self.w[winner][loser] += 1
        self.times[winner][loser] += 1
        self.times[loser][winner] += 1
        self.t += 1
        self.beta.update(winner, loser, self.w[winner][loser])

    def get_winner(self):
        stat_win = (self.w / (self.w + self.w.T) > 0.5)
        return np.argmax(stat_win.sum(axis=1))

