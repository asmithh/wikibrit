import os
import importlib
import topics_over_time
from topics_over_time.tot import TopicsOverTime
import pickle
import pandas as pd

import numpy as np
import scipy
import random

prefix = "./article_pickles/"
content = [
    pickle.load(open(prefix + fname, "rb")) for fname in sorted(os.listdir(prefix))
]
content = [[w for w in doc if len(w) < 20] for doc in content]



def to_seconds_from_epoch(fname):
    return (
        pd.to_datetime(fname.split("_")[0]) - pd.Timestamp("1970-01-01", tz="UTC")
    ) // pd.Timedelta("1s")


timestamps = [to_seconds_from_epoch(fname) for fname in sorted(os.listdir(prefix))]
min_timestamps = min(timestamps)
delta_timestamps = max(timestamps) - min_timestamps
timestamps = [(t - min_timestamps) / delta_timestamps for t in timestamps]
# for testing purposes
content = content[::50]
timestamps = timestamps[::50]

vocab = {}
for doc in content:
    for word in doc:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] = vocab[word] + 1

timestamp_range = timestamps[-1] - timestamps[0]
timestamps_scaled = [(t - timestamps[0]) / (1 + timestamp_range) for t in timestamps]

vocab = [word for word, quantity in vocab.items() if quantity < len(content) * .75 and quantity > len(content) * .15]
vocab_set = set(vocab)
content = [[w for w in doc if w in vocab_set] for doc in content]


tot = topics_over_time.tot.TopicsOverTime(content, timestamps, vocab)

theta, phi, psi = tot.TopicsOverTimeGibbsSampling()


pickle.dump((theta, phi, psi), open("theta_phi_psi_run_1.pkl", "wb"))
