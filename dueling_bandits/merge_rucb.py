from abstract_duel import AbstractDuel

import numpy as np
import logging
import argparse

def my_argmin(A):
    # A is assumed to be a 1D array
    bottomInds = np.nonzero(A==A.min())[0]
    return bottomInds[np.random.randint(0,bottomInds.shape[0])]


def my_argmax(A):
    # A is assumed to be a 1D array
    topInds = np.nonzero(A==A.max())[0]
    return topInds[np.random.randint(0,topInds.shape[0])]


class ArmTree(object):

    def __init__(self, idx_arm, batch_size=4):
        self.batch_size = batch_size
        self.batches = []
        np.random.shuffle(idx_arm)
        num_sets = np.int(np.ceil(np.float(len(idx_arm))/batch_size))
        idx_set = batch_size * np.arange(num_sets+1)
        idx_set[-1] = len(idx_arm)
        for i in range(num_sets):
            self.batches.append(idx_arm[idx_set[i]:idx_set[i+1]])

    def prune_item(self, batch_idx, ucb):
        # remove an arm once we find is dominated by any other
        batch = self.batches[batch_idx]
        if len(batch) == 1:
            return False
        l, u = np.nonzero(ucb < 0.5)
        n_l = len(l)
        if n_l == 0:
            return False
        self.batches[batch_idx].pop(l[np.random.randint(n_l)])
        return True
        
    def merge_batches(self):
        old_batches = self.batches[:]
        np.random.shuffle(old_batches)
        old_batches.sort(cmp=lambda x, y: cmp(len(x), len(y)))
        self.batches = []
        i = 0
        j = len(old_batches) - 1
        while i <= j:
            if i == j:
                self.batches.append(old_batches[i])
                break
            elif len(old_batches[i]) + len(old_batches[j]) > self.batch_size * 1.5:
                self.batches.append(old_batches[j])
                j -= 1
            else:
                self.batches.append(old_batches[i] + old_batches[j])
                i += 1
                j -= 1

    def merge_batch_pair(self, i, j):
        # i = (j + 1) % len(self.batches)
        if i != 0: 
            self.batches[j] = self.batches[i] + self.batches.pop(j)
        else:  
            self.batches[i] = self.batches[i] + self.batches.pop(j)

    def num_arms(self):
        return np.sum([len(ag) for ag in self.batches])

    def index(self, batch):
        return self.batches.index(batch)

    def current_batch_size(self, bs):
        return len(self.batches[bs])

    def survivors(self):
        return sorted([s_arm for s_arm in batch] for batch in self.batches)
        
    def __getitem__(self, key):
        return self.batches[key % len(self.batches)]

    def __len__(self):
        return len(self.batches)


class MergeRUCB(AbstractDuel):
    """docstring for MergeRCS"""
    def __init__(self, arms=[], arg=""):
        super(MergeRUCB, self).__init__()
        parser = argparse.ArgumentParser(prog=self.__class__.__name__)
        parser.add_argument("--sampler", type=str, default='MergeRUCB')
        parser.add_argument("--alpha", type=float, default=1.01)
        parser.add_argument("--epsilon", type=float, default=0.01)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--continue_sampling_experiment", type=str,
                            default="No")
        parser.add_argument("--old_output_dir", type=str, default="")
        parser.add_argument("--old_output_prefix", type=str, default="")
        parser.add_argument('--C', type=int,default=-1)
        args = parser.parse_known_args(arg.split())[0]

        self.continue_sampling_experiment = args.continue_sampling_experiment
        self.old_output_prefix = args.old_output_prefix
        self.old_output_dir = args.old_output_dir
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        
        self.arms = arms
        self.n_arms = len(arms)
        self.i_arms = range(self.n_arms)
        self.arm_tree = ArmTree(self.i_arms, self.batch_size)
        self.w = np.ones((self.n_arms, self.n_arms))
        self.times = self.w + self.w.T
        self.mean = self.w / self.times
        self.stage = 1
        if args.C == -1:
            self.C = ((4*self.alpha - 1)*self.n_arms**2 / ((2*self.alpha - 1)*self.epsilon))**(1.0/(2*self.alpha - 1))
        else:
            self.C = args.C 
        self.t = np.int(self.C + 1)
        self.iteration = 0
        self.stage = 1
        self.current_batch = self.t % len(self.arm_tree)
        self.current_abs_idx = self.arm_tree[self.current_batch]
        self.batch_w = np.ones((self.batch_size, self.batch_size))

        self.ucb = 0.5 * np.ones((self.batch_size, self.batch_size))
        self.mean = 0.5 * np.ones((self.batch_size, self.batch_size))
        self.theta = 0.5 * np.ones((self.batch_size, self.batch_size))
    
    def compute_cb(self):
        self.current_abs_idx = self.arm_tree[self.current_batch]
        w_idx = np.ix_(self.current_abs_idx, self.current_abs_idx)
        self.mean = self.w[w_idx] / self.times[w_idx]
        cb = np.sqrt(self.alpha * np.log(self.t)) / np.sqrt(self.times[w_idx])
        self.ucb = self.mean + cb
        np.fill_diagonal(self.ucb, 0.5)
        self.batch_w = self.w[w_idx]

    
    def sample_tournament(self):
        self.compute_cb()
        n_arms = len(self.mean)
        while self.arm_tree.prune_item(self.current_batch, self.ucb):
            self.compute_cb()
            n_arms = len(self.mean)
        arm_c_relative_idx = np.random.randint(0, n_arms)
        return arm_c_relative_idx

    def relative_sample(self, relative_c):
        n_arms = len(self.mean)
        relative_ucb = self.ucb[:, relative_c]
        relative_ucb[relative_c] = 0.0
        arm_d_relative_idx = my_argmax(relative_ucb)
        return arm_d_relative_idx

    def get_arms(self):
        if self.arm_tree.current_batch_size(self.current_batch) == 1:
            return self.current_abs_idx[0], self.current_abs_idx[0]
        arm_c_rel = self.sample_tournament()
        arm_d_rel = self.relative_sample(arm_c_rel)
        return self.current_abs_idx[arm_c_rel], self.current_abs_idx[arm_d_rel]

    def update_scores(self, winner, loser):
        # winner and loser are the absolute index
        self.w[winner][loser] += 1
        self.times[winner][loser] += 1
        self.times[loser][winner] += 1
        self.t += 1
        self.iteration += 1 
        if (self.iteration+1) % self.n_disp == 0:
            survivors = [sorted([s_arm for s_arm in batch] for batch in self.arm_tree.batches)]
            logging.info('MergeRUCB, iteration: %d, number of survivors: %d, survivors: %s' %(self.iteration, self.arm_tree.num_arms(), survivors))
            
        if self.arm_tree.num_arms() <= self.n_arms / (2 ** self.stage) + 1 \
                and len(self.arm_tree) > 1:
            self.arm_tree.merge_batches()
            if min([len(a) for a in self.arm_tree.batches]) <= 0.5 * self.batch_size:
                self.arm_tree.merge_batches()
            self.stage += 1
            self.t = np.int(self.C + 1)
            logging.info("%d- Iteration %d" % (self.t, self.stage))
        self.current_batch = self.t % len(self.arm_tree)
        self.current_abs_idx = self.arm_tree[self.current_batch]

    def get_winner(self):
        self.times = self.w + self.w.T
        mean = (0.0 + self.w) / self.times
        champ = my_argmax((mean > 0.5).sum(axis=1))
        logging.info("MergeRUCB.get_winner() was called!")
        return champ
