################################################################
# Imports
################################################################

# System imports
import os
import sys
import codecs
import argparse
import json
import gzip
import re
import string
from collections import Counter

# Helper imports
# - Verbose info for debugging
from traceback_with_variables import activate_by_import
# - Progress bars
from tqdm import tqdm # Runtime progress bar

# Machine learning imports
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

# NLP imports
from spacy.lang.en import English

# Other imports
from assignment1_fns import *

################################################################
# Functions from previous assignments
################################################################

# Read from speeches file, returning parallel lists of documents and their labels
def read_and_clean_lines(infile, chamber='Senate'):
    print("\nReading and cleaning text from {}".format(infile))
    lines = []
    parties = []
    with gzip.open(infile,'rt') as f:
        for line in tqdm(f):
            j = json.loads(line)
            if (j['chamber'] == chamber):
                party      = j['party']
                text       = j['text']
                clean_text = re.sub(r"\s+"," ",text)
                lines.append(clean_text)
                parties.append(party)
    print("Read {} documents".format(len(lines)))
    print("Read {} labels".format(len(parties)))
    return lines, parties


# Read a set of stoplist words from filename, assuming it contains one word per line
# Return a python Set data structure (https://www.w3schools.com/python/python_sets.asp)
def load_stopwords(filename):
    stopwords = []
    with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
        stopwords = fp.read().split('\n')
    return set(stopwords)


# Call sklearn's train_test_split function to split the dataset into training items/labels
# and test items/labels.  See https://realpython.com/train-test-split-python-data/
# (or Google train_test_split) for how to make this call.
#
# Note that the train_test_split function returns four sequences: X_train, X_test, y_train, y_test
# X_train and y_train  are the training items and labels, respectively
# X_test  and y_test   are the test items and labels, respectively
#
# This function should return those four values
def split_training_set(lines, labels, test_size=0.3, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=test_size, random_state=random_seed, stratify=labels)
    print("Training set label counts: {}".format(Counter(y_train)))
    print("Test set     label counts: {}".format(Counter(y_test)))
    return X_train, X_test, y_train, y_test

# Converting text into features.
# Inputs:
#    X - a sequence of raw text strings to be processed
#    analyzefn - either built-in (see CountVectorizer documentation), or a function we provide from strings to feature-lists
#
#    Arguments used by the words analyzer
#      stopwords - set of stopwords (used by "word" analyzer")
#      lowercase - true if normalizing by lowercasing
#      ngram_range - (N,M) for using ngrams of sizes N up to M as features, e.g. (1,2) for unigrams and bigrams
#
#  Outputs:
#     X_features - corresponding feature vector for each raw text item in X
#     training_vectorizer - vectorizer object that can now be applied to some new X', e.g. containing test texts
#    
# You can find documentation at https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# and there's a nice, readable discussion at https://medium.com/swlh/understanding-count-vectorizer-5dd71530c1b
#
def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,2)):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer

# Input:
#    lines     - a raw text corpus, where each element in the list is a string
#    stopwords - a set of strings that are stopwords
#    remove_stopword_bigrams = True or False
#
# Output:  a corresponding list converting the raw strings to space-separated features
#
# The features extracted should include non-stopword, non-punctuation unigrams,
# plus the bigram features that were counted in collect_bigram_counts from the previous assignment
# represented as underscore_separated tokens.
# Example:
#   Input:  ["This is Remy's dinner.",
#            "Remy will eat it."]
#   Output: ["remy 's dinner remy_'s 's_dinner",
#            "remy eat"]
def convert_lines_to_feature_strings(lines, stopwords, remove_stopword_bigrams=True):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")
        
    print(" Initializing")
    nlp          = English(parser=False)
    all_features = []
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):

        # Get spacy tokenization and normalize the tokens
        spacy_analysis    = nlp(line)
        spacy_tokens      = [token.orth_ for token in spacy_analysis]
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Collect unigram tokens as features
        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams          = [token   for token in normalized_tokens
                                 if token not in stopwords and token not in string.punctuation]

        # Collect string bigram tokens as features
        bigrams           = ngrams(normalized_tokens, 2) 
        bigrams           = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'

        # TO DO: replace this line with your code
        feature_list   = unigrams + bigram_tokens
        feature_string = " ".join(feature_list)

        # Add this feature string to the output
        all_features.append(feature_string)


    # print(" Feature string for first document: '{}'".format(all_features[0]))
        
    return all_features

# For both classes, print the n most heavily weighted features in this classifier.
def most_informative_features(vectorizer, classifier, n=20):
    # Adapted from https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers#11116960
    feature_names       = vectorizer.get_feature_names()
    coefs_with_features = sorted(zip(classifier.coef_[0], feature_names))
    top                 = zip(coefs_with_features[:n], coefs_with_features[:-(n + 1):-1])
    for (coef_1, feature_1), (coef_2, feature_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, feature_1, coef_2, feature_2))

# Split on whitespace, e.g. "a    b_c  d" returns tokens ['a','b_c','d']
def whitespace_tokenizer(line):
    return line.split()


