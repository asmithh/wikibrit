import os
import pickle

prefix = "./article_pickles/"
content = [
    pickle.load(open(prefix + fname, "rb")) for fname in sorted(os.listdir(prefix))
]
content = [[w for w in doc if len(w) < 20] for doc in content]

import pandas as pd


def to_seconds_from_epoch(fname):
    return (
        pd.to_datetime(fname.split("_")[0]) - pd.Timestamp("1970-01-01", tz="UTC")
    ) // pd.Timedelta("1s")


timestamps = [to_seconds_from_epoch(fname) for fname in sorted(os.listdir(prefix))]

import importlib
import topics_over_time
from topics_over_time.tot import TopicsOverTime

vocab = set()
for doc in content:
    for word in doc:
        if word not in vocab:
            vocab.add(word)
timestamp_range = timestamps[-1] - timestamps[0]
timestamps_scaled = [(t - timestamps[0]) / (1 + timestamp_range) for t in timestamps]
vocab = list(vocab)

import scipy
import random
import numpy as np

tot = topics_over_time.tot.TopicsOverTime(content, timestamps, dictionary)

theta, phi, psi = tot.TopicsOverTimeGibbsSampling()

import pickle

pickle.dump((theta, phi, psi), open("theta_phi_psi_run_1.pkl", "wb"))
