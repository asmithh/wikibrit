import os
import importlib
import topics_over_time
from topics_over_time.tot import TopicsOverTime
import pickle
import pandas as pd

import numpy as np
import scipy
import random

data = "wiki"
wiki_to_remove = {
    "publisherpeoplecom",
    "namefoxnews",
    "aligncenter",
    "rowspan",
    "mdash",
    "ndash",
    "publishermtvcom",
    "postscriptnoneref",
    "formatogg",
}

wiki_substring_to_remove = ["publisher", "nbsp", "author", "date", "year"]


def wiki_substring_safe(word):
    for w in wiki_substring_to_remove:
        if w in word:
            return False
    return True


if data == "wiki":
    prefix = "./article_pickles/"
    content = [
        pickle.load(open(prefix + fname, "rb")) for fname in sorted(os.listdir(prefix))
    ]

    def to_seconds_from_epoch(fname):
        return (
            pd.to_datetime(fname.split("_")[0]) - pd.Timestamp("1970-01-01", tz="UTC")
        ) // pd.Timedelta("1s")

    timestamps = [to_seconds_from_epoch(fname) for fname in sorted(os.listdir(prefix))]

    content = [[w for w in doc if len(w) < 20 and len(w) > 3] for doc in content]
    content = [
        [w for w in doc if w not in wiki_to_remove and wiki_substring_safe(w)]
        for doc in content
    ]
elif data == "tmz":
    pickles_to_load = ["articles_scraped.pkl", "articles_scraped_run_2.pkl"]
    articles = []
    for pkl in pickles_to_load:
        articles.extend(pickle.load(open(pkl, "rb")))
    content = [
        [
            w.strip().lower()
            for w in a[0].split("see also")[0].split("Share on Facebook")[0].split()
        ]
        for a in articles
    ]

    def to_seconds_from_epoch(ts):
        return (pd.to_datetime(ts) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    timestamps = [to_seconds_from_epoch(a[1]) for a in articles]

    content = [[w for w in doc if len(w) < 20 and len(w) > 3] for doc in content]
print(timestamps)


min_timestamps = min(timestamps)
delta_timestamps = max(timestamps) - min_timestamps
timestamps = [(t - min_timestamps) / delta_timestamps for t in timestamps]
# for testing purposes
content = content[::5]
timestamps = timestamps[::5]

vocab = {}
for doc in content:
    for word in list(set(doc)):
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] = vocab[word] + 1


vocab = [
    word
    for word, quantity in vocab.items()
    if quantity < len(content) * 0.5 and quantity > len(content) * 0.05
]
vocab_set = set(vocab)
content = [[w for w in doc if w in vocab_set] for doc in content]

tot = topics_over_time.tot.TopicsOverTime(content, timestamps, vocab)

theta, phi, psi = tot.TopicsOverTimeGibbsSampling()


pickle.dump((theta, phi, psi), open("theta_phi_psi_run_1.pkl", "wb"))
