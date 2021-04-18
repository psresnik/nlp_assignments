import sys
import numpy as np
import argparse


# Import functions from other assignments
from assignment4_fns import *
 
# Convenient for debugging but feel free to comment out
from traceback_with_variables import activate_by_import


###################################################################################################################
# Adaptation of the main logistic regression experiment from Assignment 2, to use cross-validation.
###################################################################################################################
def run_experiment(input_speechfile, stopwords_file, use_sklearn_feature_extraction, test_size, num_folds, stratify, random_seed):

    # Load stopwords
    stop_words = load_stopwords(stopwords_file)

    # Read the dataset in and split it into training documents/labels (X) and test documents/labels (y)
    # Note that for this assignment, we are then going to ignore X_test and y_test.
    # This simulates real-world experimentation where one might successively improve the system
    # using cross-validation on the training set, e.g. trying different features or doing hyperparameter tuning,
    # while being careful not to test on the test data until the end.
    X, y                              = read_and_clean_lines(input_speechfile)
    X_train, X_test, y_train, y_test  = split_training_set(X, y, test_size)

    # Feature extraction is the same as done previously
    if use_sklearn_feature_extraction:
        # Use sklearn CountVectorizer's built-in tokenization to get unigrams and bigrams as features
        X_features_train, training_vectorizer = convert_text_into_features(X_train, stop_words, "word", range=(1,2))
        X_test_documents = X_test
    else:
        # Roll your own feature extraction.
        # Call convert_lines_to_feature_strings() to get your features
        # as a whitespace-separated string that will now represent the document.
        print("Creating feature strings for training data")
        X_train_feature_strings = convert_lines_to_feature_strings(X_train, stop_words)

        # Commenting out feature extraction for final test data, since we're not going to use it
        # (See earlier comment about X_test and y_test.)
        #  print("Creating feature strings for final test data")
        #  X_test_documents        = convert_lines_to_feature_strings(X_test,  stop_words)
        
        # Call CountVectorizer with whitespace-based tokenization as the analyzer, so that it uses exactly your features,
        # but without doing any of its own analysis/feature-extraction.
        # Again, this is the same as previously
        X_features_train, training_vectorizer = convert_text_into_features(X_train_feature_strings, stop_words, whitespace_tokenizer)

    # Create a k-fold cross validation object.
    # Use the shuffle parameter (shuffle before splitting) and pass in random_seed.
    # Use either Kfold or StratifiedKFold - the latter makes sure the ratio of labels in each fold is the same as the original data.
    # See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    # and https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    # Note that if you use stratified split() it needs to include y_train (i.e. labels) so it knows the proportion of labels in the original training set.
    print("Doing cross-validation splitting with stratify={}. Showing 10 indexes for items in train/test splits in {} folds.".format(stratify,num_folds))
    kfold = None # Replace this with appropriate function call to create cross-validation object
        
    # Create the classifier object
    classifier = LogisticRegression(solver='liblinear')

    # Do cross-validation and look at mean/stdev of scores by calling cross_val_score()
    # See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
    # Nice tutorial: https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/
    # The cross_val_score expects classifier, feature-vectorized docs, labels, evaluation score to use ('accuracy'), and  the kfold object
    # Use accuracy, but for other evaluation scores you can see https://scikit-learn.org/stable/modules/model_evaluation.html
    print("Running {}-fold cross-validation on {}% of the data, still holding out the rest for final testing.".format(num_folds,(1-test_size)*100))
    accuracy_scores = [0]*num_folds # Replace this line with your call to cross_val_score()
    print("accuracy scores = {}, mean = {}, stdev = {}".format(accuracy_scores, np.mean(accuracy_scores), np.std(accuracy_scores)))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Classification with cross validation')
    parser.add_argument('--use_sklearn_features', default=False, action='store_true', help="Use sklearn's feature extraction")
    parser.add_argument('--test_size',            default=0.3,   action='store',      help="Proportion (from 0 to 1) of items held out for final testing")
    parser.add_argument('--num_folds',            default=5,     action='store',      help="Number of folds for cross-validation (use 2 for just a train/test split)")
    parser.add_argument('--stratify',             default=False, action='store_true', help="Use stratified rather than plain cross-validation")
    parser.add_argument('--seed',                 default=13,    action='store',      help="Random seed")
    parser.add_argument('--stopwords',            default="./mallet_en_stoplist.txt",  action='store',      help="Stopwords file")
    parser.add_argument('--infile',               default=None,  action='store',      help="Input jsonlines.gz file")
    args = parser.parse_args()
    if args.infile == None:
        print("Argument --infile is required.")
        sys.exit(1)
    run_experiment(args.infile,
                       args.stopwords,
                       args.use_sklearn_features,
                       float(args.test_size),
                       int(args.num_folds),
                       args.stratify,
                       int(args.seed))

