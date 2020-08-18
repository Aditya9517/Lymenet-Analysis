# -*- coding: utf-8 -*-
# Takes a csv file as input with only one column containing text.
# to specify Column change the variable columnNumber to the required
# LDA_lymeDisease.py outfile_content.csv col_number num_topics num_passes

"""
Updated on Thu Apr 26 3:43:47 2020

@author: Aditya Jayanti
"""

import csv
import logging
import os
import re
import string
import sys
import time
from collections import defaultdict

import emoji
import gensim
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.utils import lemmatize
from matplotlib.pyplot import hist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pymongo import MongoClient
from scipy.stats import entropy

from topic_distribution import generate_topic_distribution
from topic_similarity import generate_kendall_tau_similarity

# sampleSize = 16000
grid = {}

stop = set(stopwords.words('english') + ["lyme"])
punctuations = string.punctuation
add_stop = set('br/ I ... -- n\'t \'s'.split())
stemmer = PorterStemmer()

# Database under consideration is Medical Question Data
collection_name = 'medicalQuestion'
generatedFiles = "generatedFiles"

perplexity, bound, coherence, postList, ids = [], [], [], [], []

# initializing the parameter list

# TEST 1
# parameter_list = [1]
# parameter_list += [n for n in range(1, 101) if n % 5 == 0]

# TEST 2
# parameter_list = [n for n in range(40, 141) if n % 5 == 0]

# TEST 3 - Generating 10 Topic Models for testing similarity
low, high, mod = 20, 100, 8
parameter_list = [n for n in range(low, high) if n % mod == 0]

# testing topic size
topics_generated = {}

plt.switch_backend('agg')


# parse the file into a list
def read_posts_from_database(db):
    global postList, ids
    medicalQuestionData = db[collection_name + "Data"]  # db.medicalQuestionData
    cursor = medicalQuestionData.find(no_cursor_timeout=True)
    counts = defaultdict(int)

    for lymeRecord in cursor:
        counts[lymeRecord["topic"]] += 1
        currentText = lymeRecord["text"]
        postList.append(currentText)
        ids.append(lymeRecord['_id'])


# generates a histogram containing number of words in text
def get_word_count_histogram(texts):
    counts = [len(t) for t in texts]
    plt.clf()
    plt.ylabel("Frequency of Posts or Comments")
    plt.xlabel("Number of Words")
    hist(counts, log=True)
    plt.savefig("graphsAndFigures/words_by_doc__hist_" + collection_name + ".pdf")


# tokenize a post
def tokenize(post):
    for currPunct in punctuations:
        post = post.replace(currPunct, "")
    if bool(emoji.get_emoji_regexp().search(post)):
        post = emoji.demojize(post)
    tokens = lemmatize(post)
    tokens = [str(x).split("/")[0].split('\'')[1] for x in tokens]
    tokens = [item for item in tokens if not item in stop and item not in add_stop]
    return tokens


def plot_it(x, y, xname, yname, topic=""):
    plt.clf()
    plt.plot(x, y, color='blue')
    if topic:
        plt.title(topic)
        plt.grid()
    plt.xlabel(xname, fontsize=12)
    plt.ylabel(yname, fontsize=12)
    plt.savefig(generatedFiles + '/' + yname.replace(" ", "_") + ".pdf")


def print_topics(ldamodel, parameter_value):

    topics_brief = ldamodel.show_topics(
        num_topics=parameter_value, num_words=15, log=False, formatted=False)
    print('Printing Topics...\n\n')

    count = 0
    for topic in topics_brief:
        print(topic)
        print("%d: %s" % (count, " ".join([x for x, y in topic[1]])))
        # topics_generated[count] = "%s" % (" ".join([x for x, y in topic[1]]))
        count += 1


# save text to file
def write_to_file(texts):
    thefile = open(generatedFiles + '/frequentWords.txt', 'w')
    for text in texts:
        thefile.write(" ".join(text) + "\n")


def __getLDA__(cp_train, cp_test, dictionary, parameter_value, number_of_words, db):
    # print "starting pass for num_topic = %d" % num_topics_value
    print("starting pass for parameter_value = %.3f" % parameter_value)
    print("Training set size: %d" % len(cp_train))
    start_time = time.time()

    # parameter value set to parameter_val leads to inconsistent results
    ldamodel = models.LdaModel(corpus=cp_train, id2word=dictionary, num_topics=parameter_value, eval_every=10, chunksize=3000,
                               passes=40, eta=None)

    print_topics(ldamodel, parameter_value)
    # show elapsed time for model
    elapsed = time.time() - start_time
    print("Elapsed time: %s" % elapsed)

    ldamodel.save(generatedFiles + '/LDA_MODEL_' + "%d" % parameter_value)

    varbound = ldamodel.bound(cp_test)
    bound.append(varbound)
    perplex = ldamodel.log_perplexity(cp_test)

    cm = models.coherencemodel.CoherenceModel(model=ldamodel, corpus=cp_test, coherence='u_mass')
    coheren = cm.get_coherence()
    coherence.append(coheren)

    per_word_perplex = -perplex / number_of_words
    print("Per-word Perplexity: %s" % per_word_perplex)
    perplexity.append(per_word_perplex)

    if not istest:
        coll = db[collection_name + "Data"]
        for (i, doc) in zip(ids, cp_train):
            dist = ldamodel.get_document_topics(doc)

            # initializing new list as it throws non-compatible error
            new_dist = []
            for x, y in dist:
                new_dist.append((x, float(y)))

            coll.update_one({'_id': {'$eq': i}},
                            {'$set': {'document_topics': new_dist,
                                      'topic_entropy': entropy([float(y) for x, y in dist])}})

    return varbound, per_word_perplex, coheren


# main
def main():
    global collection_name
    global generatedFiles
    global istest
    global postList
    logging.basicConfig(filename='gensim.log',
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        level=logging.INFO)

    client = MongoClient('127.0.0.1', 27017)
    lyme_disease_database = client.lymeDiseaseDB

    if len(sys.argv) > 1:
        collection_name = sys.argv[1]

    t = time.localtime(time.time())
    time_stamp = time.strftime("%Y:%m:%d:%H:%M:%S", t)

    model_fit_file = open("model_fitness.csv", "a")

    if "--test" in sys.argv:
        istest = True
        print("Test mode enabled")
    else:
        istest = False

    generatedFiles = generatedFiles + '_' + collection_name
    if not os.path.exists(generatedFiles):
        os.makedirs(generatedFiles)

    try:
        corpus = gensim.corpora.MmCorpus(
            collection_name + '.mm')
        dictionary = gensim.corpora.Dictionary.load(
            collection_name + '.dict')
        print('Found corpus and dictionary for the input file')
    except IOError:
        print("Corpora or dictionary for the input file not found, creating one...")
        read_posts_from_database(lyme_disease_database)
        # for sampling to select model
        texts = [text.lower() for text in postList]

        print('tokenizing')
        texts = [tokenize(text) for text in postList]

        print('tokenized')
        # remove words that appear only once
        write_to_file(texts)

        # generates a histogram on the number of words in the text
        get_word_count_histogram(texts)

        # get_word_count_histogram(texts)
        dictionary = corpora.Dictionary(texts)
        print("Number of distinct words: %d" % len(dictionary))
        dictionary.filter_extremes(no_below=2, no_above=1.0)
        print("Number of distinct words appearing more than once: %d" % len(dictionary))
        dictionary.save(collection_name + '.dict')
        print('new dictionary saved')
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize(collection_name + '.mm', corpus)
        print('new corpora stored')
    cp = list(corpus)

    # train and test assigned the same value as we do not perform prediction
    cp_train = cp
    cp_test = cp

    number_of_words = sum(
        cnt for document in cp_test for _, cnt in document)
    for parameter_value in parameter_list:
        varbound, per_word_perplex, coheren = __getLDA__(cp_train, cp_test, dictionary,
                                                         parameter_value, number_of_words, lyme_disease_database)
        # import pdb
        # pdb.set_trace()

        # Coherence Generated
        # Greater coherence score = Better quality of Topics
        print("Coherence Score Generated = ", coheren)

        # Perplexity Generated
        # Lower coherence score = Better quality of Topics
        print("Perplextity Score Generated = ", per_word_perplex)

        model_fit_file.write(
            "%s\t%d\t%E\t%E\t%E\n" % (time_stamp, parameter_value, per_word_perplex, varbound, coheren))
        model_fit_file.flush()
    model_fit_file.close()


if __name__ == '__main__':
    main()
    # print(grid)

    with open('LDA_perplexity_' + collection_name + "_.csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in zip(parameter_list, perplexity):
            writer.writerow([key, value])

    # Plot for perplexity, coherence
    plot_it(parameter_list, perplexity, 'Number of Topics', "Perplexity")
    plot_it(parameter_list, bound, 'Number of Topics', "Variational bound")
    plot_it(parameter_list, coherence, 'Number of Topics', "Coherence")

    # Find convergence parameters
    pattern = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [pattern.findall(p) for p in open('gensim.log')]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    perplexity = [float(t[1]) for t in tuples]
    liklihood = [float(t[0]) for t in tuples]
    iter_loop = list(range(0, len(tuples) * 10, 10))

    # Plot for convergence
    plot_it(iter_loop, liklihood, 'Iteration', "Log Likelihood", "Topic Model Convergence")

    # Generate topic distribution
    # All LDA generated models are saved in generatedFiles_medicalQuestion/LDA_MODEL_"
    lda_model_path = "/Users/adityakalyanjayanti/PycharmProjects/lymenetanalysis/generatedFiles_" + collection_name + "/LDA_MODEL_"

    # Document path are the posts/comments extracted from MongoDB
    document_path = "/Users/adityakalyanjayanti/PycharmProjects/lymenetanalysis/Data/documents.json"

    # This file is generated by topic_distribution.py, calculates the topic distribution in each topic model
    topic_model_distribution = "/Users/adityakalyanjayanti/PycharmProjects/lymenetanalysis/Data/topic_model_distribution.csv"

    # Method determines the topic distribution in each topic model and saves the data in a CSV file
    generate_topic_distribution(document_path, lda_model_path, parameter_list, topic_model_distribution)

    # Obtain Kendall-Tau distance pairwise using topic distribution values
    kendall_tau = generate_kendall_tau_similarity(topic_model_distribution)

    # plot kendall-tau similarity
    plot_it(parameter_list, kendall_tau, "Number of Topics", "Kendall-Tau Similarity")

