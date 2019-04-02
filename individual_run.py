from dueling_bandits import *

import time
import timeit
import datetime
import argparse
import numpy as np
import os
import logging

FLAGS = None
START_TIME = str(datetime.datetime.now()).replace(' ', '-').replace(':', '').split('.')[0]

METHOD_DICT = ['MergeDTSBEST',  'MergeDTS', 'MergeRUCBBEST', 'MergeRUCB', 'SelfSparring', 'DTSBEST', 'DTS']

METHOD_INIT = {
               'MergeRUCB': lambda ARMS: MergeRUCB(ARMS[:], arg='--alpha=1.01'),
               'MergeDTS': lambda ARMS: MergeDTS(ARMS[:], '--alpha=%f --C=%d --batch_size=%d' % (1.01, -1, 4)),
               'MergeRUCBBEST': lambda ARMS: MergeRUCB(ARMS[:], arg='--alpha=%f --C=%d --batch_size=%d' % (np.power(0.8, 6), 400000, 8)),
               'MergeDTSBEST': lambda ARMS: MergeDTS(ARMS[:], '--alpha=%f --C=%d --batch_size=%d' % (np.power(0.8, 6), 400000, 16)),
               'DTS': lambda ARMS: DTS(ARMS[:], arg='--alpha=0.51'),
               'DTSBEST': lambda ARMS: DTS(ARMS[:], arg='--alpha=%f' % np.power(0.8, 7)),
               'SelfSparring': lambda ARMS: SelfSparring(ARMS[:]),
               }
SEED = []


def init_bandits(method_name):
    return METHOD_INIT[method_name]


def save_results(regrets, dataset_name, method_name, winner_type, suffix):
    prefix_dict = './results/' + str(FLAGS.iterations) + '/'
    if not os.path.exists(prefix_dict + dataset_name):
        os.makedirs(prefix_dict + dataset_name)
    f = open(prefix_dict + dataset_name + '/' + method_name + '-' + winner_type + '-' + suffix + '.npy', 'wb')
    np.save(f, regrets)
    f.close()


def run_bandits(rep, pm, method_name, dataset_name, FLAGS):
    # time.sleep(rep*10)
    logging.info('Job %d has been submitted.' % rep)
    np.random.seed(SEED[rep])
    # Init
    suffix = 'rep-' + str(rep) + '-' + str(datetime.datetime.now()).replace(' ', '-').replace(':', '').split('.')[0]
    regrets_condorcet = np.zeros(FLAGS.iterations/FLAGS.scale)
    n_arms = len(pm)
    rdp = np.random.permutation(n_arms)
    pref_matrix = pm[np.ix_(rdp, rdp)]
    copeland_score = (pref_matrix > 0.5).sum(axis=1) / float(len(pref_matrix) - 1)
    winner = copeland_score.argmax()
    gap = pref_matrix[winner] - 0.5
    bandits_inst = init_bandits(method_name)(range(len(pm)))
    f_bandits = 'objs/' + dataset_name + '/'
    if not os.path.exists(f_bandits):
        if not os.path.exists('objs'):
            os.mkdir('objs')
        elif not os.path.exists('objs/' + dataset_name):
            os.mkdir('objs/' + dataset_name)
        else:
            logging.info('Invalid path' + f_bandits)
            return 0
    # Running
    s = timeit.default_timer()
    j_step = 10000
    for epoch_i in range(FLAGS.iterations/j_step):
        for epoch_j in range(j_step):
            epoch = epoch_j + j_step*epoch_i
            l_arm, r_arm = bandits_inst.get_arms()
            coin = np.random.rand()
            if coin <= pref_matrix[l_arm][r_arm]:
                bandits_inst.update_scores(l_arm, r_arm)
            else:
                bandits_inst.update_scores(r_arm, l_arm)
            regrets_condorcet[epoch / FLAGS.scale] += 0.5 * (gap[l_arm] + gap[r_arm])
            if not (epoch + 1) % 100000000:
                bandits_inst.save(f_bandits + method_name + '-' + suffix + '.bandit')
            if not (epoch + 1) % 1000000:
                logging.info('Repeat%d finished %dth iteration' % (rep, epoch))
    time_run = timeit.default_timer() - s
    logging.info('finished %d repeat' % rep)
    # save results
    save_results(np.cumsum(regrets_condorcet), dataset_name, method_name, 'condorcet', suffix)
    # save_results(time_run, dataset_name, method_name, 'time', suffix)
    return np.cumsum(regrets_condorcet), time_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_idx', default=3, type=int)
    parser.add_argument('--iterations', default=int(1e8), type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--method_idx', default=0, type=int)
    parser.add_argument('--scale', default=int(1e4), type=int)
    FLAGS = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
    files = os.listdir('./matrix')
    filename = files[FLAGS.data_idx]
    method_name = METHOD_DICT[FLAGS.method_idx]  
    pref_matrix = np.loadtxt('./matrix/' + filename)
    copeland_score = (pref_matrix > 0.5).sum(axis=1) / float(len(pref_matrix) - 1)
    logging.info('winner: %d, score: %f' % (copeland_score.argmax(), copeland_score.max()))

    if len(pref_matrix) < 100:
        FLAGS.repeat = 100
        FLAGS.iterations = int(1e7)
    FLAGS.scale = FLAGS.iterations / 100000

    for key, value in vars(FLAGS).items():
        if key is 'data_idx':
            logging.info('dataset' + ' : ' + filename)
        elif key is 'method_idx':
            logging.info('method : ' + method_name)
        else:
            logging.info(key + ' : ' + str(value))

  
    SEED = np.random.choice(10000, FLAGS.repeat)

    for rep in range(FLAGS.repeat):
      _, t_item = run_bandits(rep, pm=pref_matrix[:], method_name=method_name, dataset_name=filename[:-4], FLAGS=FLAGS)
    logging.info('finished the run of ' + method_name + ' on ' + filename + 'with time of %f' % t_item)




