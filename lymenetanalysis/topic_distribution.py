# -*- coding: utf-8 -*-
"""
Updated on Thu Apr 23 3:43:47 2020

@author: Aditya Jayanti
"""

import csv
import json

from gensim import models


# Iterating over all posts and computing the probability
def iterate_posts(topic_model_number, lda_models, data):
    topic_distribution = []
    for k, post in data.items():
        if post != []:
            # Converting each post/comment to BoW (Bag of Words)
            bow = topic_model_number.id2word.doc2bow(post)
            topic_distribution.append(lda_models[0].get_document_topics(bow, minimum_probability=0.0))
    return topic_distribution


# Create Map of probability values associated with topic
def map_probability_topics(topic_model_number, lda_models, data):
    map_probability = {}
    for l in iterate_posts(topic_model_number, lda_models, data):
        for i in l:
            if i[0] not in map_probability:
                map_probability[i[0]] = [i[1]]
            else:
                map_probability[i[0]].append(i[1])
    # Computing average across all posts/comments
    for k, v in map_probability.items():
        map_probability[k] = sum(v) / len(v)
    return map_probability


# Iterate over all the topic models and determining topic distribution
def generate_topic_distribution(document_path, lda_model_path, parameter_list, topic_model_distribution):
    with open(document_path, 'r') as f:
        data = json.load(f)

    # Loading LDA MODELS
    # list holds all the topic models
    lda_models = []

    for num in parameter_list:
        # Load LDA model and store the model in a list
        lda_models.append(models.LdaModel.load(lda_model_path + str(num)))

    # Holds the resulting topic distribution for each topic model
    model_probability = []

    for model in range(len(lda_models)):
        # model indicates topic model number
        model_probability.append(map_probability_topics(lda_models[model], lda_models, data))

    print(model_probability)
    keys = model_probability[0].keys()
    with open(topic_model_distribution, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(model_probability)
