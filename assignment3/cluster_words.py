import sys
import spacy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# Word clustering using k-means, using spacy's vector representations of words,
# which are 300-dimensional GloVe vectors trained on the Common Crawl; see https://spacy.io/models/en
# and https://nlp.stanford.edu/projects/glove/. This function:
#   Converts a list of words into a list of vectors using the spacy nlp object
#   Does k-means clustering to generate numclusters clusters
#   Returns a dictionary that maps from numeric cluster label to the list of words in that cluster
#
# Note: this function silently ignores words that don't have a vector representation in spacy.
#
# See https://github.com/danielwilentz/Cuisine-Classifier/blob/master/topic_modeling/clustering.ipynb
# for an example of clustering that uses word2vec vectors, which you can find online by searching
# for GoogleNews-vectors-negative300.bin. That example also illustrates a common manual method for
# deciding on how many clusters to use.
def cluster_words(nlp, words, numclusters):
    
    # Convert words into spacy tokens and from there into vectors
    # Only include tokens that actually have a vector in spacy
    tokens     = [nlp(word) for word in words]
    vectors    = [token.vector for token in tokens if token.has_vector]
    wordlist   = [token.text   for token in tokens if token.has_vector]

    # Do k-means clustering on the set of vectors
    # Result of clustering is item_labels, a numpy array with one entry per item that was clustered
    # That is, item_labels[0] is the cluster label for the first vector,
    #          item_labels[1] is the cluster label for the next vector, etc.
    sys.stderr.write("Running KMeans...\n")
    df_vectors = pd.DataFrame(vectors)
    km         = KMeans(n_clusters=numclusters, init='k-means++', random_state=10, n_init=1)
    km.fit(df_vectors)
    item_labels = km.predict(df_vectors)  

    # Convert k-means result into a dictionary (clusterdict) mapping from
    # a cluster label to list of the words in that cluster
    clusterdict = {}
    item_number = 0
    for item_label in item_labels:
        if item_label in clusterdict:
            clusterdict[item_label].append(wordlist[item_number])
        else:
            clusterdict[item_label] = [wordlist[item_number]]
        item_number +=1
    
    return clusterdict



if __name__ == "__main__":
    
    # Illustrate k-means clustering using spacy's vector representations.
    # The result, clusterdict, is a mapping from integer cluster labels (0, 1, etc.)
    # to words in the cluster.  What do you expect the words to be in the two clusters?
    
    print("Initializing spaCy")
    nlp = spacy.load('en_core_web_sm')
    
    print("Clustering")
    wordlist    = ['eat', 'drink', 'consume', 'dog', 'cat', 'bear', 'pig']
    clusterdict = cluster_words(nlp, wordlist, 2)

    print("Clustering result")
    print(clusterdict)
