# Copyright 2015 Abhinav Maurya

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import fileinput
import random
import scipy.special
import math
import numpy as np
import scipy.stats
import pickle
from math import log
from multiprocessing import Pool


class TopicsOverTime:
    def __init__(self, documents, timestamps, dictionary, n_topics=20, n_iter=100):
        self.document_chunk_size = 500
        self.max_iterations = n_iter  # max number of iterations in gibbs sampling
        self.n_topics = n_topics  # previously T     # number of topics
        self.n_documents = len(documents)  # previously D
        self.vocab_size = len(dictionary)  # previously V
        self.document_lengths = [len(doc) for doc in documents]  # previously N

        self.alpha = (50.0 / self.n_topics) * np.ones(self.n_topics)
        self.beta = 0.1 * np.ones(self.vocab_size)
        self.beta_sum = np.sum(self.beta)

        self.psi = np.ones((2, self.n_topics))
        self.beta_function_psi = [
            scipy.special.beta(self.psi[t][0], self.psi[t][1])
            for t in range(self.n_topics)
        ]
        self.word_id = {word: idx for idx, word in enumerate(dictionary)}
        self.word_token = dictionary

        # timestamps[d][i] is the timestamp of the ith term in document d.
        # previously t
        self.timestamps = [
            timestamps[d] * np.ones(self.document_lengths[d])
            for d in range(self.n_documents)
        ]

        # previously z
        # topic_by_doc_tokens[d][i] is the topic associated with the ith term in document d.
        self.topic_by_doc_tokens = [
            np.random.randint(0, self.n_topics, self.document_lengths[d])
            for d in range(self.n_documents)
        ]

        # words_by_doc_index[d][i] is the word ID of the ith word in document d
        # previously w
        self.words_by_doc_index = [
            [self.word_id[documents[d][i]] for i in range(self.document_lengths[d])]
            for d in range(self.n_documents)
        ]

        # previously m
        # tokens_assigned_to_topics_by_doc[t][d] is the number of tokens assigned to topic t in document d
        self.tokens_assigned_to_topics_by_doc = np.zeros(
            (self, n_documents, self.n_topics)
        )

        # previously n
        # word_topic_assignments[w][t] is the number of times word w is assigned to topic t
        self.word_topic_assignments = np.zeros((self.n_topics, self.vocab_size))

        # previously n_sum
        self.words_to_topic_total = np.zeros(self.n_topics)
        print("initialized")
        np.set_printoptions(threshold=np.inf)
        np.seterr(divide="ignore", invalid="ignore")
        self.CalculateCounts()

    def CalculateCounts(self, par):
        """
        For each document in all documents:
            For each word in each document:
                topic_di is the assigned topic to the ith word in document d.
                word_di is the word ID of that word (int ID).
                the word count of words belonging to that topic in the document is incremented by one.
                the number of times that word is assigned to that topic is incremented by one.
                the overall number of words in that topic is incremented by one.
        """
        document_indices = [
            (left_index, min(left_index + 500, self.n_documents))
            for left_index in range(0, self.n_documents, self.document_chunk_size)
        ]

        def calculate_counts(args):
            tokens_assigned_to_topics_by_doc = args["tattbd"]
            topic_by_doc_tokens = args["tbdt"]
            words_by_doc_index = args["wbdi"]
            word_topic_assignments = args["wta"]
            words_to_topic_total = args["wttt"]
            doc_lengths = args["dl"]

            for d in range(args["left_index"], args["right_index"]):
                for i in range(doc_lengths[d]):
                    topic_di = topics_by_doc_tokens[d][
                        i
                    ]  # topic in doc d at position i
                    word_di = words_by_doc_index[d][i]  # word ID in doc d at position i
                    tokens_assigned_to_topics_by_doc[d][topic_di] += 1
                    word_topic_assignments[topic_di][word_di] += 1
                    words_to_topic_total[topic_di] += 1

            return_package = {
                "tattbd": tokens_assigned_to_topics_by_doc,
                "wta": word_topic_assignments,
                "wttt": words_to_topic_total,
            }

            return return_package

        def make_arg_package(indices):
            args = {}
            args["tattbd"] = self.tokens_assigned_to_topics_by_doc
            args["tbdt"] = self.topic_by_doc_tokens
            args["wbdi"] = self.words_by_doc_index
            args["wta"] = self.word_topic_assignments
            args["wttt"] = self.words_to_topic_total
            args["dl"] = doc_lengths
            args["left_index"] = indices[0]
            args["right_index"] = indices[1]

            return args

        args_list = [make_arg_package(idx) for idx in document_indices]

        with Pool(processes=8) as pool:
            res = pool.imap_unordered(calculate_counts, args_list)

        self.tokens_assigned_to_topics_by_doc = np.sum(
            [r["tattbd"] for r in res], axis=0
        )
        self.word_topic_assignments = np.sum([r["wta"] for r in res], axis=0)
        self.word_to_topic_total = np.sum([r["wttt"] for r in res], axis=0)

    def GetTopicTimestamps(self):
        topic_timestamps = []
        for topic in range(self.n_topics):
            current_topic_timestamps = []
            current_topic_doc_timestamps = [
                [
                    (self.topic_by_doc_tokens[d][i] == topic) * self.timestamps[d][i]
                    for i in range(self.document_lengths[d])
                ]
                for d in range(self.n_documents)
            ]
            for d in range(self.n_documents):
                current_topic_doc_timestamps[d] = filter(
                    lambda x: x != 0, current_topic_doc_timestamps[d]
                )
            for timestamps in current_topic_doc_timestamps:
                current_topic_timestamps.extend(timestamps)
            assert current_topic_timestamps != []
            topic_timestamps.append(current_topic_timestamps)
        return topic_timestamps

    def GetMethodOfMomentsEstimatesForPsi(self):
        topic_timestamps = self.GetTopicTimestamps(par)
        psi = [[1 for _ in range(2)] for _ in range(len(topic_timestamps))]
        for i in range(len(topic_timestamps)):
            current_topic_timestamps = topic_timestamps[i]
            timestamp_mean = np.mean(current_topic_timestamps)
            timestamp_var = np.var(current_topic_timestamps)
            if timestamp_var == 0:
                timestamp_var = 1e-6
            common_factor = timestamp_mean * (1 - timestamp_mean) / timestamp_var - 1
            psi[i][0] = 1 + timestamp_mean * common_factor
            psi[i][1] = 1 + (1 - timestamp_mean) * common_factor
        return psi

    def ComputePosteriorEstimatesOfThetaAndPhi(self):
        theta = deepcopy(self.tokens_assigned_to_topics_by_doc)
        phi = deepcopy(self.word_topic_assignments)

        for d in range(self.n_documents):
            if sum(theta[d]) == 0:
                theta[d] = np.asarray(
                    [1.0 / len(theta[d]) for _ in range(len(theta[d]))]
                )
            else:
                theta[d] = np.asarray(theta[d])
                theta[d] = 1.0 * theta[d] / sum(theta[d])
        theta = np.asarray(theta)

        for t in range(self.n_topics):
            if sum(phi[t]) == 0:
                phi[t] = np.asarray([1.0 / len(phi[t]) for _ in range(len(phi[t]))])
            else:
                phi[t] = np.asarray(phi[t])
                phi[t] = 1.0 * phi[t] / sum(phi[t])
        phi = np.asarray(phi)

        return theta, phi

    def ComputePosteriorEstimatesOfTheta(self):
        theta = deepcopy(self.tokens_assigned_to_topics_by_doc)

        for d in range(self.n_documents):
            if sum(theta[d]) == 0:
                theta[d] = np.asarray(
                    [1.0 / len(theta[d]) for _ in range(len(theta[d]))]
                )
            else:
                theta[d] = np.asarray(theta[d])
                theta[d] = 1.0 * theta[d] / sum(theta[d])

        return np.matrix(theta)

    def ComputePosteriorEstimateOfPhi(self):
        phi = deepcopy(self.word_topic_assignments)

        for t in range(self.n_topics):
            if sum(phi[t]) == 0:
                phi[t] = np.asarray([1.0 / len(phi[t]) for _ in range(len(phi[t]))])
            else:
                phi[t] = np.asarray(phi[t])
                phi[t] = 1.0 * phi[t] / sum(phi[t])

        return np.matrix(phi)

    def TopicsOverTimeGibbsSampling(self):
        for iteration in range(self.max_iterations):
            for d in range(self.n_documents):
                for i in range(self.document_lengths[d]):
                    word_di = self.words_by_doc_index[d][i]
                    t_di = self.timestamps[d][i]

                    old_topic = self.topic_by_doc_tokens[d][i]
                    self.tokens_assigned_to_topics_by_doc[d][old_topic] -= 1
                    self.word_topic_assignments[old_topic][word_di] -= 1
                    self.words_to_topic_total[old_topic] -= 1

                    topic_probabilities = []
                    for topic_di in range(self.n_topics):
                        psi_di = self.psi[topic_di]
                        topic_probability = 1.0 * (
                            self.tokens_assigned_to_topics_by_doc[d][topic_di]
                            + self.alpha[topic_di]
                        )
                        topic_probability *= ((1 - t_di) ** (psi_di[0] - 1)) * (
                            (t_di) ** (psi_di[1] - 1)
                        )
                        topic_probability /= self.beta_function_psi[topic_di]
                        topic_probability *= (
                            self.word_topic_assignments[topic_di][word_di]
                            + self.beta[word_di]
                        )
                        topic_probability /= (
                            self.words_to_topic_total[topic_di] + self.beta_sum
                        )
                        topic_probabilities.append(topic_probability)
                    sum_topic_probabilities = sum(topic_probabilities)
                    if sum_topic_probabilities == 0:
                        topic_probabilities = [
                            1.0 / self.n_topics for _ in range(self.n_topics)
                        ]
                    else:
                        topic_probabilities = [
                            p / sum_topic_probabilities for p in topic_probabilities
                        ]

                    new_topic = list(
                        np.random.multinomial(1, topic_probabilities, size=1)[0]
                    ).index(1)
                    self.topic_by_doc_tokens[d][i] = new_topic
                    self.tokens_assigned_to_topics_by_doc[d][new_topic] += 1
                    self.word_topic_assignments[new_topic][word_di] += 1
                    self.words_to_topic_total[new_topic] += 1

                if d % 1000 == 0:
                    print(
                        "Done with iteration {iteration} and document {document}".format(
                            iteration=iteration, document=d
                        )
                    )
            self.psi = self.GetMethodOfMomentsEstimatesForPsi(par)
            self.beta_function_psi = [
                scipy.special.beta(self.psi[t][0], self.psi[t][1])
                for t in range(self.n_topics)
            ]
        (
            self.tokens_assigned_to_topics_by_doc,
            self.word_topic_assignments,
        ) = self.ComputePosteriorEstimatesOfThetaAndPhi(par)
        return (
            self.tokens_assigned_to_topics_by_doc,
            self.word_topic_assignments,
            self.psi,
        )
