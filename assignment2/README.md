## Coding up a logistic regression classifier

The goal of this part of the assignment is *not* to write a lot of code, or re-invent the wheel, but to put together a simple NLP classification pipeline,  analyze results, and make small modifications to see how things change.

You will build a simple logistic regression classifier using scikit-learn that takes the Congressional speeches we worked with from Assignment 1, and tries to predict whether the speaker is from the Democratic party or the Republican party.  You'll use a "bag of words" representation of the documents (speeches) where the features are unigrams and bigrams. 

Specifically:

- You'll be working with the code in `assignment1.py`.  Note that `assignment1_fns.py` contains some of the functions from assignment 1 in case you prefer them to your own (or had trouble writing them correctly).
- You'll make a small modification to the `read_and_clean_lines` function from assignment 1 so that it's returning not only a list of the documents, like before, but also a corresponding list of the labels for those documents ('Democrat' or 'Republican'). 
- You'll split these documents and labels into training set and a held-out test set, using a 70/30 split.  You're going to use scikit-learn's `train_test_split` function for this in function `split_training_set`. It's a one-line call, and in a comment in the code we're pointing you to a tutorial example of how to do it.
- You'll transform the raw text of speeches (the 'documents') into features in `convert_lines_to_feature_strings`.  This is basically the same extraction of unigrams and bigrams you did in assignment 1 for purposes of counting things, except now for each document you're representing that document as a whitespace-separated string of the unigrams and bigrams you found in it.
- After doing some experimentation with Senate speeches, you'll modify your program so that it can do the same thing with House speeches, which means one more update to `read_and_clean_lines` where it can use 'House' instead of 'Senate'.

That's it for code you're writing.  Everything else for training and running/testing the logistic regression classifier is provided for you in the code.

## How you know your code is working

- Once `read_and_clean_lines` is working correctly, the print statements should give you the same number of documents and labels to return: 

```
Reading and cleaning text from ./speeches2020_jan_to_jun.jsonl.gz
[progress bar]
Read 4535 documents
Read 4535 labels
```

- Once `split_training_set` is working for the Senate speeches, the split of training and test data should give you these numbers:

```
Training set label counts: Counter({'Republican': 2126, 'Democrat': 1048})
Test set     label counts: Counter({'Republican': 912, 'Democrat': 449})
```

- Before you go further, you can run `python assignment.py --use_sklearn_features` and the program will run end to end using its own built-in unigram and bigram tokenization in scikit-learn's CountVectorizer (see [here](https://medium.com/swlh/understanding-count-vectorizer-5dd71530c1b) for a nice tutorial discussion).  

- Once `convert_lines_to_feature_strings` is working you should see results like the following:

```
Creating feature strings for training data
 Converting from raw text to unigram and bigram features
 Includes filtering stopword bigrams
 Initializing
 Iterating through documents extracting unigram and bigram features
 [progress bar]
 Feature string for first document: 'madam president yesterday gave details historic police reform passed unanimously houses republican controlled legislature surely democrats iowa work republicans find unanimity n't problems u.s. congress passing unanimously 's perfect enemy good morning senate majority leader mcconnell senator tim scott members republican task force unveiled piece legislation title unifying solutions invigorate communities acronym justice short applaud leadership issue democrats stop partisan attacks spend time working find solutions pretty simple iowa legislature bipartisan unanimous madam_president historic_police police_reform passed_unanimously controlled_legislature find_unanimity u.s._congress congress_passing senate_majority majority_leader leader_mcconnell senator_tim tim_scott republican_task task_force force_unveiled unifying_solutions invigorate_communities acronym_justice stop_partisan partisan_attacks find_solutions pretty_simple iowa_legislature'
Creating feature strings for test data
 Converting from raw text to unigram and bigram features
 Includes filtering stopword bigrams
 Initializing
 Iterating through documents extracting unigram and bigram features
 [progress bar]
 Feature string for first document: 'mr. president talk thing 5:30 night grant program strive policing bias count concept civilian review process prosecutorial decisions n't understand senator lee asked questions bill animosity object time hope part broader agenda june 16 hearing things related police race make part package time object committee mr._president grant_program civilian_review review_process prosecutorial_decisions senator_lee broader_agenda june_16 things_related'
```

- At this point, the program should run end to end. By default it will show you the 10 most informative features for each class used in the logistic regression, e.g.

```
	-1.6608	speak          		1.3086	break          
	-1.3796	reserving      		1.1832	cloture_motion 
	-1.1198	announce       		1.1760	counsel        
	-1.0694	rescinded      		1.0707	suggest        
	-1.0151	passed         		1.0534	minute         
	-0.9792	objection      		1.0195	calendar       
	-0.8969	massachusetts  		0.9680	senators       
	-0.8920	house_managers 		0.9652	permitted      
	-0.8685	managers       		0.9639	recognized     
	-0.8569	statement      		0.9498	motion 
```

and then performance metrics, e.g.:

```
Accuracy  = 0.8148420279206466
Precision for label Republican = 0.9111842105263158
Recall    for label Republican = 0.8293413173652695
Precision for label Democrat = 0.6191536748329621
Recall    for label Democrat = 0.7743732590529248
```

- If you use the `--plot_metrics` flag when running the program, it will generate (a) a [confusion matrix,](https://en.wikipedia.org/wiki/Confusion_matrix) and (b) an [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) showing the tradeoff between true positives and false positives for the *Republican* label as the threshold for assigning that label varies. For example, running with sklearn's features, the ROC curve shows that if you want to achieve a *true positive rate* of about .8 (that is, to successfully label 80% of the true Republicans as 'Republican', also known as *recall* or *sensitivity), with this classifier you'd need to be willing to accept about a 20% false positive rate (that is, 20% of Democrats would be mistakenly labeled 'Republican', also known as "false alarms").  Note that on your terminal it may pop the windows up one right on top of the other, so you have to move one to see the other.


## After you've gotten your code working

A. In your writeup answer the following:

1. Show the evaluation numbers you obtain when running the code with your feature extraction.
1. Show the confusion matrix that you obtain.
3. How does this performance compare to the simple baseline of simply picking the most frequent label in the training set for every test item?

B. Modify your code to repeat the experiment with speeches from the House rather than the Senate.  Answer the same questions (1,2,3) above.

C. Modify your code one more time with the House and Senate speeches combined.  Again answer the same questions (1,2,3).

D. Fill in the table below. 


| Dataset       | # Train examples |  # Test examples | Overall Test Accuracy | 
| -----------   | -------------- | -------------- | -------------  |
| House         |                |                |         |
| Senate        |                |                |         |
| All        |                |                |         |

## Extra credit (up to 20%)


See if you can change your code in a way that improves accuracy of logistic regression classification in the Senate by 2% or more. Here are some ideas for things to try.  You'll get at least partial extra credit for all reasonable effort that you report on in a thorough, thoughtful way.


- See whether documents or even features involving procedural language (e.g. "I yield back the balance of my time") are helping you or hurting performance and modify accordingly. The file `new_legis_proc_jargon_stopwords.txt` contains terms likely to be found in procedural language.
- Try balancing the training data so that the minority label is as well represented as the majority label.  One good way to do this would be to use the [imbalanced-learn package](http://glemaitre.github.io/imbalanced-learn/index.html), in particular [over-sampling](https://imbalanced-learn.org/stable/over_sampling.html) with naive random over-sampling or with SMOTE.
- Try automatically identifying features that are particularly likely to be useful, e.g. you could consider using a statistical method to find features that are strongly associated with one label or the other in the training data by computing statistics like the log-likelihood ratio ([code](https://github.com/tdunning/python-llr)) for the following contingency table:

	
	|           |  Democrat  |  Republican  |
	| ---       |  -------- |  ----------  |
	|  **feature**      |  count(feature, Democrat) | count(feature, Republican) |
	| **not feature** |  sum over f != feature [ count(f, Democrat) ] | sum over f != feature [ count(f, Republican) ]|

	Then you could modify your feature extraction to limit itself to useful features as a subset of the ones you extracted before.
	
- Perform error analysis: look at actual documents classified mistakenly with a strong probability from the classifier. Look at false positives separately from false negatives, and see if you can identify any likely properties of the text, or of your features that seem to be characteristically misleading. Talk about them, or, if you want to go truly end to end, modify your code to try to do better.

	Now, of course, if you do this, you're improving your system based on looking at what happened on test data, which is cheating.  There are two things you could do here to avoid that.  One is to do a train/devtest/test split for the whole first round above.  That is, instead of 70% train, 30% test, you could, for example, do 70% train, 15% devtest, 15% held-out test. (Do a 70-30 split and divide the latter in half randomly.) You would do everything, all the way up to and including error analysis and modifying your code, on the 70% train/15% *devtest* split, holding the remaining 15% test set out, unexamined, until the very end.  After you've gotten your code doing the very best you can via error analysis and everything else, evaluation performance of (a) the scikit-learn version, (b) your original code, and (c) your improved code, all on the true 15% test data.

