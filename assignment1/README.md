
## Normalizing text and exploring a corpus

### Overview
In this assignment, you'll get experience with:

- Starting with a typical "raw" dataset 
	- We'll be using a dataset of speeches from the U.S. Congressional Record during 2020, acquired using code at [https://github.com/ahoho/congressional-record](https://github.com/ahoho/congressional-record).  This is publicly available material.
- Extracting relevant text to create one or more corpora
	- We'll restrict ourselves to the Senate, and create subcorpora of speeches by Democrats and Republicans.
- Tokenizing text
	- We'll use the spaCy tokenizer
- Normalizing text
	- Well use case folding and also a stopword list
- Extracting potentially useful ngrams
	- In this assignment we'll focus on bigrams


### The files you'll be working with  
You'll be working with the following:

- Files in [jsonlines](https://jsonlines.org/) format containing raw data
	- `test_speeches.jsonl.gz` - small example data for testing
	- `speeches2020_jan_to_jun.jsonl.gz` - main data you'll run on
- Files containing code
	- `assignment.py` - code skeleton that you'll fill in
	- `public_tests_obj.py` - code to run for unit testing
- Other resources
	- `mallet_en_stoplist.txt` - the stopword list from the widely used [Mallet](http://mallet.cs.umass.edu/) toolkit


### What you should do

- Check out this repo


- Execute `python assignment.py`
	- It should run successfully from end to end with progress messages on the output
	- If it does not, most likely it's because it is using packages you don't have installed. Install them (see: requirements.txt)
		- If you use conda, we recommend installing a fresh conda env and putting your classwork dependencies there.
		- Execute 

				conda create --name YOURCONDAENVIRONMENT python=3.8
				conda activate YOURCONDAENVIRONMENT
				which pip
				
		- Ensure that your `pip` lives in its own env, like: `/anaconda3/envs/YOURCONDANEVIRONMENT/bin/pip`
		- Execute `pip install -r requirements.txt`

- Execute `python public_tests_obj.py -v`
	- The code should run, but will report on tests that have failed.

- Read and modify `assignment.py`.  
	- Each function has a detailed comment about input, output, and what it does.
	- You can look at `public_tests_obj.py` for examples of the function calls.
	- You will find a comment like `# ASSIGNMENT: replace this with your code` everywhere you have work to do.
	- Keep working until all the tests pass when you run `public_tests_obj.py`.

- **Code to be graded.** Once all tests pass, submit `assignment.py` to the autograder. This will be the basis for grading your code.

- **Analysis to be graded.** For the analysis part of the assignment, look at the output of `assignment.py` and submit a brief but clear written response as PDF.  (Note that, particularly if you are not very familiar with U.S. politics, you are welcome to discuss the data you're looking at with other other people -- as long as you state explicitly in your writeup that you have done so, and of course you need to write your answers in your own words.)

	-- *Looking at frequency.* The first set of outputs are lists of the top Democratic and Republican bigrams by frequency.  Looking at these lists, how similar or different are the most-frequent bigrams used by members of the two parties?  Are there any generalizations you can make about the two parties, at least during this time period, based on this information? If yes, discuss. If you think the answer is no, clearly explain why. Support your answer with examples.
		
	




