import os
import pickle

prefix = './article_pickles/'
content = [pickle.load(open(prefix + fname, 'rb')) for fname in sorted(os.listdir(prefix))]
content = [[w for w in doc if len(w) < 20] for doc in content]

import pandas as pd
def to_seconds_from_epoch(fname):
    return (pd.to_datetime(fname.split('_')[0]) - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
timestamps = [to_seconds_from_epoch(fname) for fname in sorted(os.listdir(prefix))]

import importlib
import topics_over_time
from topics_over_time.tot import TopicsOverTime

tot = topics_over_time.tot.TopicsOverTime()
vocab = set()
for doc in content:
    for word in doc:
        if word not in vocab:
            vocab.add(word)
timestamp_range = timestamps[-1] - timestamps[0]
timestamps_scaled = [(t - timestamps[0]) / timestamp_range for t in timestamps]
vocab = list(vocab)

import scipy
import random
import numpy as np

par = {}						# dictionary of all parameters
par['max_iterations'] = 100		# max number of iterations in gibbs sampling
par['T'] = 20					# number of topics
par['D'] = len(content)
par['V'] = len(vocab)
par['N'] = [len(doc) for doc in content]
par['alpha'] = [50.0/par['T'] for _ in range(par['T'])]
par['beta'] = [0.1 for _ in range(par['V'])]
par['beta_sum'] = sum(par['beta'])
par['psi'] = [[1 for _ in range(2)] for _ in range(par['T'])]
par['betafunc_psi'] = [scipy.special.beta( par['psi'][t][0], par['psi'][t][1] ) for t in range(par['T'])]
par['word_id'] = {word: idx for idx, word in enumerate(vocab)}
par['word_token'] = vocab
par['z'] = [np.random.randint(0, par['T'], par['N'][d]) for d in range(par['D'])]
par['t'] = [timestamps[d] * np.ones(par['N'][d]) for d in range(par['D'])]
par['w'] = [[par['word_id'][content[d][i]] for i in range(par['N'][d])] for d in range(par['D'])]
par['m'] = [np.zeros(par['T']) for d in range(par['D'])]
par['n'] = [np.zeros(par['V']) for t in range(par['T'])]
par['n_sum'] = np.zeros(par['T'])
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore', invalid='ignore')
tot.CalculateCounts(par)

theta, phi, psi = tot.TopicsOverTimeGibbsSampling(par)

import pickle
pickle.dump((theta, phi, psi), open('theta_phi_psi_run_1.pkl', 'wb'))
