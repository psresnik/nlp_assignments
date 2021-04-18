## Evaluating NLP

The on-paper assignment is [here](evaluation.pdf).  No programming is required, although if you're solid in python I do recommend you consider exercising your skills in question 2(b) rather than doing it by hand!

This assignment is due at the start of class on April 26th at 6pm ET. 


## Optional extra credit (up to 40%)

You are also welcome to do optional extra credit, which makes use of code in this repository.  This is a modification of the logistic regression assignment you've already done, making it a big step more realistic by adding cross-validation.

### Background info

The code skeleton is in [assignment4.py](assignment4.py). You will see that the code still splits the dataset into training and test data, like in Assignment 2. However, instead of then training a classifier on the training data, and testing on the test data, here we do cross-validation on *just* the training data. (You can think of the test data as being held out for later summative evaluation, which is what you'd typically do in the real world also.)

I've provided a small input file ([small.json.gz](small.json.gz)) for development/debugging, since that'll be faster than working with the whole set of speeches. If you run this with test_size 0.1 (so that cross-validation on the training set is using 90% of the available data), what you get looks like this, though of course where there's anything random what you get may be slightly different.

```
% python assignment4.py --test_size 0.1 --num_folds 5 --stratify --seed 42 --infile small.jsonl.gz 
Reading and cleaning text from small.jsonl.gz
Read 47 documents
Read 47 labels
Training set label counts: Counter({'Republican': 34, 'Democrat': 8})
Test set     label counts: Counter({'Republican': 4, 'Democrat': 1})
Creating feature strings for training data
 Converting from raw text to unigram and bigram features
 Includes filtering stopword bigrams
 Initializing
 Iterating through documents extracting unigram and bigram features
Doing cross-validation splitting with stratify=True. Showing 10 indexes for items in train/test splits in 5 folds.
Fold 1 - train: [ 2  3  4  5  6  7  8  9 10] test [12 13 16 19 20 23 31 41]
Fold 2 - train: [ 1  2  3  4  6  7  8 10 12] test [ 9 11 17 18 32 36 37 38]
Fold 3 - train: [ 3  4  5  6  7  8  9 10 11] test [ 2 14 15 22 28 34 39]
Fold 4 - train: [ 1  2  5  8  9 10 11 12 13] test [ 4  6  7 25 27 29 30]
Fold 5 - train: [ 1  2  3  4  5  6  7  9 11] test [10 21 24 26 33 35 40]
Running 5-fold cross-validation on 90.0% of the data, still holding out the rest for final testing.
accuracy scores = [0.77777778 0.55555556 0.875      0.875      0.75      ], mean = 0.7666666666666667, stdev = 0.11699688715918159
```

Most of that should be pretty familiar, except that now at the end it's showing you the cross-validation folds (with the first 10 train/test items, respectively, for each fold), the accuracy for each fold, and then the mean and standard deviation.  It shouldn't be surprising to you that the variance is so high across folds, given how tiny the training set is!

### What you need to do

If you want to implement cross-validation from scratch, that's ok. However, the code skeleton I've given you makes it very easy to use the cross-validation capabilities in scikit-learn with just a few lines of code and I very strongly recommend doing that; just look for the places where a comment says '# Replace...'.  For all the credit below, use the full data in `speeches2020_jan_to_jun.jsonl.gz`.


- Baseline (10%): Modify the code in [assignment4.py](assignment4.py) so that after doing the train/test split, it uses [(non-stratified) k-fold cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) on the training items and their labels.
- +5%: Add the option to use [*stratified* cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) if the `--stratify` flag is provided. With stratified k-fold cross-validation, the sampling will ensure that each fold has the same distribution of labels as in the original dataset, in this case the training side of the top-level train/test split.
- +10%: Show how cross-validation mean accuracy and standard deviation change as you move from smaller to larger quantities of data.  Start with a very small training set (`--test_size=.9`, so that you're doing cross-validation on only 10% of the available data) and proceed by increments of 10% up to `--test_size=.1`.  You can show this as a table, but if you have the time the best practice would be to show learning curves, e.g. like [this](https://i.stack.imgur.com/haGpo.png). If you've done both stratified and non-stratified cross-validation show both curves.
- +5%: Add another classifier (e.g. SVM), and show how its performance compares with logistic regression. If you've done the previous parts show the learning curves also.
- +10%: Train at least two different classifiers (e.g. previous bullet), select the one with the best cross-validation accuracy, and do summative evaluation by running that classifier on the held-out test set.  (Alternatively, use one type of classifier, but use cross-validation to assess different parameter settings and do summative evaluation using that classifier with the best parameters.)


