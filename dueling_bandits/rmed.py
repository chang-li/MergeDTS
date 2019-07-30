from abstract_duel import AbstractDuel

import numpy as np
import logging
import argparse

DELTA = 1e-7


def my_argmax(ay):
    idx = np.nonzero(ay == ay.max())[0]
    return idx[np.random.randint(0, idx.shape[0])]


def my_argmin(ay):
    idx = np.nonzero(ay == ay.min())[0]
    return idx[np.random.randint(0, idx.shape[0])]


def KLBernoulli(p, q=0.5):
    p += 0.
    q += 0.
    if p > 1 or q > 1:
        return -1
    if p <= DELTA:
        return - np.log(1 - q)
    if p >= 1 - DELTA:
        return - np.log(q)
    return p * np.log(p / q) + (1-p) * np.log((1-p) / (1-q))


class RMED1(AbstractDuel):
    def __init__(self, arms, arg=""):
        super(RMED1, self).__init__()
        self.arms = arms
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='RMED1')
        parser.add_argument("--alpha", type=float, default=0.51)
        parser.add_argument("--continue_sampling_experiment", type=str,
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        parser.add_argument("--random_seed", type=int, default=42)
        parser.add_argument("--T", type=int, default=int(1e8))
        parser.add_argument("--kweight", type=float, default=0.3, help="The weight of f(k)=K^1.01 in Eq. 4")
        args = parser.parse_known_args(arg.split())[0]

        self.name = args.sampler

        ### for random seed
        self.random_seed = args.random_seed
        self.prng = np.random.RandomState(self.random_seed)

        self.alpha = args.alpha
        self.T = args.T
        self.n_arms = len(arms)
        self.func_k = args.kweight * (self.n_arms**1.01)  # f(k)
        self.threshold = self.func_k  # Eq. 4 log(t) + f(k)

        self.w = np.zeros((self.n_arms, self.n_arms))  # pair-wise observation
        np.fill_diagonal(self.w, 1)
        self.n = self.w + self.w.T
        self.u = 0.5 * np.ones((self.n_arms, self.n_arms))  # preference estimator
        self.t = 0  # time step
        self.log_likelihood = np.ones(self.n_arms) # log likelihood

        self.arm_l = 0
        self.arm_r = 0
        self.LC = self.prng.permutation(self.n_arms).tolist()  # arms in current step
        self.LR = set(range(self.n_arms))  # remaining arms
        self.LN = set()   # next step

    def get_relative_arm(self):
        u = self.w[self.arm_l] / (self.w[self.arm_l] + self.w[:, self.arm_l])
        u[self.arm_l] = 1
        u_min = u.min()
        arm_r = self.arm_l if u_min > 0.5 else my_argmin(u)
        return arm_r

    def get_arms(self):
        if self.t < self.n_arms * (self.n_arms -1) / 2.:
            self.arm_r += 1
            if self.arm_r == self.n_arms:
                self.arm_l += 1
                self.arm_r = self.arm_l + 1
        else:
            self.arm_l = self.LC.pop()
            self.arm_r = self.get_relative_arm()
        return self.arm_l, self.arm_r

    def __update_likelihood(self):
        for k in [self.arm_l, self.arm_r]:
            o_hat = np.where(self.u[k] <= 0.5)[0]
            kl = [KLBernoulli(self.u[k][j], 0.5) for j in o_hat]
            self.log_likelihood[k] = np.dot(self.n[k][o_hat], kl)

    def __get_likelihood(self):
        for k in range(self.n_arms):
            o_hat = np.where(self.u[k] <= 0.5)[0]
            kl = [KLBernoulli(self.u[k][j], 0.5) for j in o_hat]
            self.log_likelihood[k] = np.dot(self.n[k][o_hat], kl)

    def update_scores(self, winner, loser):
        self.w[winner][loser] += 1
        self.n[winner][loser] += 1
        self.n[loser][winner] += 1
        self.u[winner][loser] = self.w[winner][loser] / self.n[winner][loser]
        self.u[loser][winner] = 1 - self.u[winner][loser]

        if self.t >= self.n_arms * (self.n_arms - 1) / 2.:
            # update list
            # Line 15
            self.LR.remove(self.arm_l)

            # Line 16
            self.__update_likelihood()
            new_items = np.where(self.log_likelihood <= self.log_likelihood.min() + np.log(self.t) + self.func_k)[0]
            if new_items.shape[0]:
                for item in new_items:
                    if item not in self.LR:
                        self.LN.add(item)

            if not self.LC: # Line 19
                self.LC = list(self.LN)
                self.prng.shuffle(self.LC)
                self.LR, self.LN = self.LN, set()

        self.t += 1
        if self.t == self.n_arms * (self.n_arms - 1) / 2.:
            self.__get_likelihood()

        if (self.t+1) % self.n_disp == 0:
            logging.info('%s, iteration: %d, potential winner is %d.' %(self.name, self.t, self.get_winner()))

    def get_winner(self):
        u = self.w / (self.w + self.w.T)
        return my_argmax(np.sum(u > 0.5, axis=1))


if __name__ == "__main__":
    pm = np.loadtxt('../matrix-backup/ArXiv.txt')
    T = 100000
    regret = np.zeros(T)
    sampler = RMED1(arms=range(pm.shape[0]), arg="--T=%d" % T)
    copeland_score = (pm > 0.5).sum(axis=1) / float(len(pm) - 1)
    winner = copeland_score.argmax()
    gap = pm[winner] - 0.5

    for t in range(T):
        l, r = sampler.get_arms()
        coin = np.random.rand()
        if coin >= pm[l][r]:
            sampler.update_scores(r, l)
        else:
            sampler.update_scores(l, r)

        regret[t] = 0.5 * (gap[l] + gap[r])

    print(np.cumsum(regret)[[0, 9, 100-1, 1000-1, 10000-1, 100000-1]])