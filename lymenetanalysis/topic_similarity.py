# -*- coding: utf-8 -*-
"""
Updated on Thu Apr 23 3:43:47 2020

@author: Aditya Jayanti
"""

import csv
import warnings
from queue import PriorityQueue

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")


# Reading Data from file
def read_topic_distribution(topic_model_distribution):
    topic_model_values = []
    TM = []
    with open(topic_model_distribution, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            x = np.array(row)
            topic_model_values.append(x.astype(np.float64))

    for i in topic_model_values:
        count = 0
        r = []
        for j in i:
            r.append((count, j))
            count += 1
        TM.append(r)
    return TM


def measuring_similarity(TopicModel1, TopicModel2):
    # Variable
    Q = PriorityQueue()
    # for i in TM:
    #   print(i)
    # TopicModel1 = TM[0]
    # TopicModel2 = TM[1]
    seen_topics1 = []
    seen_topics2 = []
    Dx = 0
    T = []

    # Ordering Topic Models by Topic Distribution
    TopicModel1.sort(key=lambda tup: tup[1])
    TopicModel2.sort(key=lambda tup: tup[1])

    # print(TopicModel1)
    # print(TopicModel2)
    for t1 in TopicModel1:
        for t2 in TopicModel2:
            Dx = jensenshannon(t1, t2)
            Q.put((Dx, [t1, t2]))

    while not Q.empty():
        row = Q.get()
        topic1 = row[1][0]
        topic2 = row[1][1]

        if topic1 not in seen_topics1 and topic2 not in seen_topics2:
            seen_topics1.append(topic1)
            seen_topics2.append(topic2)
            T.append([topic1, topic2])

    # sorting the topics by topic distribution for each of the two topic models
    seen_topics1.sort(key=lambda tup: tup[1])
    seen_topics2.sort(key=lambda tup: tup[1])

    # print(seen_topics1)
    # print(seen_topics2)
    t1 = [k[0] for k in seen_topics1]
    t2 = [k[0] for k in seen_topics2]
    return stats.kendalltau(t1, t2)


def generate_kendall_tau_similarity(topic_model_distribution):
    corr = 0
    p = 0
    correl = []
    result = []
    TM = read_topic_distribution(topic_model_distribution)
    for TopicModeli in range(len(TM)):
        correlation = []
        for TopicModelj in range(len(TM)):
            if TopicModeli != TopicModelj:
                corr, p = measuring_similarity(TM[TopicModeli], TM[TopicModelj])
            correlation.append(corr)
            correl.append(corr)
        temp = sum(correlation) / len(correlation)
        result.append(temp)
        print("Topic Model" + str(TopicModeli), "corr = " + str(temp))
    print("Average Kendall-Tau score over all pairs = ", str(sum(correl) / len(correl)))
    return result
