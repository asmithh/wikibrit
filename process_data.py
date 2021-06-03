import collections
import os
import numpy as np
import pandas as pd
import pickle
import re
import string

from gensim import models, corpora
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# crappy attempt at getting LDA to work on these documents. takes a long time to run.

# best_n_clusters is a total guess. feel free to change.
best_n_clusters = 10

# these tend to work pretty well as presets
stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# feel free to mess with the punctuation we remove/don't remove
punctuation_no_underscore = set(string.punctuation)
punctuation_no_underscore.add("’")
punctuation_no_underscore.add("”")
punctuation_no_underscore.remove("_")

my_dict = corpora.Dictionary()
pickles = []
filter_keys = {"="}

# ./revisions is the set of all revisions as .txt files
L = len(os.listdir("./revisions"))
tot = 0
for wiki_version in os.listdir("./revisions"):
    if tot % 500 == 0:
        print(tot / L)
        # just to keep track of our progress. there are a bunch of revisions!!
    tot += 1
    words = []
    with open(os.path.join("./revisions", wiki_version)) as f:
        for line in f.readlines():
            line = re.sub("[^a-zA-Z ]+", "", line)
            # we're getting rid of anything that isn't a letter - this is a bit severe and we can mess with it
            words.extend(line.split())
    content_no_punctuation = [word.lower() for word in words]
    # here we remove stopwords
    content_no_stopwords = [
        word for word in content_no_punctuation if word not in stopwords
    ]
    # lemmatizing (removing suffixes/verb forms to get at the root forms of words iirc)
    content_lemmatized = [lemmatizer.lemmatize(word) for word in content_no_stopwords]
    # getting rid of small words
    content_not_short = [word for word in content_lemmatized if len(word) > 2]
    # pickling bag-of-words files for easier access later/keyword search
    pickle_filename = "article_pickles/" + wiki_version[:-4] + ".pkl"
    pickle.dump(content_not_short, open(pickle_filename, "wb"))
    pickles.append(pickle_filename)
    # adding bag-of-words document to dictionary
    my_dict.add_documents([content_not_short])

my_dict.filter_extremes(no_below=int(len(os.listdir("./revisions")) / 50), no_above=0.9)
# get rid of common/rare words

corpus = []
for pickle_file in pickles:
    text = pickle.load(open(pickle_file, "rb"))
    # bag-of-words-ifying documents
    corpus.append(my_dict.doc2bow(text))

# creating & training LDA (latent dirichlet allocation) model for document clustering
lda_model = models.LdaModel(
    corpus=corpus,
    num_topics=best_n_clusters,
    id2word=my_dict,
)

topic_keywords = []
pickle.dump(lda_model, open("lda_model.pkl", "wb"))
# try to distinguish topics by keywords (pretty crappy right now)
for idx, topic in lda_model.show_topics(num_topics=10, num_words=20, formatted=False):
    print("Topic: {} \nWords: {}".format(idx, [w[0] for w in topic]))
    for my_word in [w[0] for w in topic]:
        print(my_word)
    topic_keywords.append([w[0] for w in topic])

pickle.dump(topic_keywords, open("topic_keywords.pkl", "wb"))
