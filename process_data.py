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


best_n_clusters = 10
stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

punctuation_no_underscore = set(string.punctuation)
punctuation_no_underscore.add("’")
punctuation_no_underscore.add("”")
punctuation_no_underscore.remove("_")

my_dict = corpora.Dictionary()
pickles = []
filter_keys = {'='}

for wiki_version in os.listdir('./revisions'):
	words = []
	with open(os.path.join('./revisions', wiki_version)) as f:
		for line in f.readlines():
			words.extend(line.split())
	content_no_punctuation = [word for word in words if re.match("[a-zA-Z0-9]+", word)]
    # here we remove stopwords
	content_no_stopwords = [word for word in content_no_punctuation if word not in stopwords]
	content_lemmatized = [lemmatizer.lemmatize(word) for word in content_no_stopwords]
	content_not_short = [word for word in content_lemmatized if len(word) > 2]
	content_no_weird = [word for word in content_not_short if set([l for l in word]) & filter_keys != {}]
	pickle_filename = wiki_version[:-4] + '.pkl'
	pickle.dump(content_not_short, open(pickle_filename, 'wb'))
	pickles.append(pickle_filename)
	my_dict.add_documents([content_not_weird])

my_dict.filter_extremes(no_below=int(len(os.listdir('./revisions')) / 50), no_above=0.5)

corpus = []
for pickle_file in pickles:
	text = pickle.load(open(pickle_file, 'rb'))
	corpus.append(my_dict.doc2bow(text))

lda_model = models.LdaModel(
    corpus=corpus,
    num_topics=best_n_clusters,
    id2word=my_dict,
)

topic_keywords = []	
for idx, topic in lda_model.show_topics(
    num_topics=10, num_words=20, formatted=False
):
    print("Topic: {} \nWords: {}".format(idx, [w[0] for w in topic]))
    for my_word in [w[0] for w in topic]:
        print(my_word)
    topic_keywords.append([w[0] for w in topic])

topics_by_doc = lda_model.load_document_topics()
print(topics_by_doc)
