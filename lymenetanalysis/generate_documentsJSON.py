# -*- coding: utf-8 -*-
"""
Updated on Thu Apr 29 3:43:47 2020

@author: Aditya Jayanti
"""

import string
from pymongo import MongoClient
from gensim.utils import lemmatize
from nltk.stem.porter import PorterStemmer
import emoji
import json
from nltk.corpus import stopwords
stop = set(stopwords.words('english') + ["lyme"])
punctuations = string.punctuation
add_stop = set('br/ I ... -- n\'t \'s'.split())
stemmer = PorterStemmer()


def tokenize(post):
    for currPunct in punctuations:
        post = post.replace(currPunct, "")
    if bool(emoji.get_emoji_regexp().search(post)):
        post = emoji.demojize(post)
    tokens = lemmatize(post)
    tokens = [str(x).split("/")[0].split('\'')[1] for x in tokens]
    tokens = [item for item in tokens if not item in stop and item not in add_stop]
    return tokens


def read_posts_db():
    documents = {}
    # Create an instance of Mongo client
    client = MongoClient('127.0.0.1', 27017)

    # Connect to database
    lyme_disease_database = client.lymeDiseaseDB

    # Read collection
    collection = lyme_disease_database.medicalQuestionData

    # Find all documents in database
    details = collection.find()

    document_count = 0
    for doc in details:
        # tokenize documents
        documents[document_count] = list(set(tokenize(doc['text'])))
        document_count += 1

    # Save documents in a JSON file
    with open("Data/documents.json", 'w') as f:
        json.dump(documents, f)


def read_file(path):
    # read date from JSON file
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def main():
    read_posts_db()
    path = "/Data/documents.json"
    data = read_file(path)


if __name__ == '__main__':
    main()



