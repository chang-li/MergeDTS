from abstract_duel import AbstractDuel

import numpy as np
import logging
import argparse


def my_argmax(a):
    idx = np.nonzero(a == a.max())[0]
    return idx[np.random.randint(0, len(idx))]


class FastBeta:
    def __init__(self, W, depth=100):
        # W - a square matrix containing a score sheet
        self.depth = depth
        self.shape = W.shape
        self.W = np.maximum(W, np.ones(W.shape))
        self.allSamples = np.zeros(self.shape+(depth,))
        self.sampleAllBeta()
        self.depthIndex = 0
        self.upper_or_lower = 'UPPER'

    def sampleAllBeta(self):
        self.allSamples = 0.5*np.ones(self.shape+(self.depth,))
        for r,c in [(row,col) for row in range(self.shape[0])
                              for col in range(self.shape[1]) if row != col]:
            self.allSamples[r,c,:] = np.random.beta(self.W[r,c],self.W[c,r],self.depth)

    def update(self,r,c,w):
        self.W[r,c] = w
        self.allSamples[r,c,:] = np.random.beta(self.W[r,c],self.W[c,r],self.depth)
        self.allSamples[c,r,:] = np.random.beta(self.W[c,r],self.W[r,c],self.depth)

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


class RelativeThompsonSampling(AbstractDuel):

    def __init__(self, arms, arg=""):
        super(RelativeThompsonSampling, self).__init__()
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='Relative Thompson Sampling')
        parser.add_argument("--continue_sampling_experiment", type=str,
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        args = parser.parse_known_args(arg.split())[0]
        self.sampler = args.sampler
        self.arms = arms

        self.n_arms = len(arms)
        self.w = np.ones((self.n_arms, self.n_arms))
        self.times = self.w + self.w.T
        self.theta = 0.5 * np.ones((self.n_arms, self.n_arms))
        self.t = 1
        self.lcb = np.ones(self.n_arms)
        self.beta = FastBeta(self.w)

    def sample_tournament(self):
        self.theta = self.beta.getSamples()
        wins = (self.theta > 0.5) + ((self.theta > 0) & (self.theta < 0.5)).T
        cope_wins = wins.sum(axis=1)
        return my_argmax(cope_wins)

    def relative_sample(self, arm_c):
        rel_theta = np.array([np.random.beta(self.w[i][arm_c], self.w[arm_c][i]) for i in range(self.n_arms)])
        rel_theta[arm_c] = 0.5
        return my_argmax(rel_theta)

    def get_arms(self):
        arm_c = self.sample_tournament()
        arm_d = self.relative_sample(arm_c)
        return arm_c, arm_d

    def update_scores(self, winner, loser):
        if self.t % 100000 == 0:
            logging.info('RTS, iteration: %d, winner: %d' % (self.t, self.get_winner()))
        self.w[winner][loser] += 1
        self.times[winner][loser] += 1
        self.times[loser][winner] += 1
        self.t += 1
        self.beta.update(winner, loser, self.w[winner][loser])

    def get_winner(self):
        stat_win = (self.w / (self.w + self.w.T) > 0.5)
        return np.argmax(stat_win.sum(axis=1))

