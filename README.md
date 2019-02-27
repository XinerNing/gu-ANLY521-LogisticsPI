# gu-ANLY521-LogisticsPI
Logistic Regression Model on Paraphrase Identification
---------------------------------------------------

This project examines string similarity metrics for paraphrase identification. It converts semantic textual similarity data to paraphrase identification data using threshholds. Though semantics go beyond the surface representations seen in strings, some of these metrics constitute a good benchmark system for detecting paraphrase.



Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

## Homework: pi_logreg.py
Train a logistic regression model predicting whether one sentence is paraphase of the other using similarity of features: "NIST", "BLEU", "Word Error Rate", "Longest common substring", "Levenshtein distance". The model is trained using train.csv, then being applied in dev.csv. The script prints three things. First, length of training pairs. Second, length of dev pari. Finally, accuracy score on dev.

## Discription of each metrics used
* NIST: a method for evaulating the quality of text which has been translated using machine translation.
* BLEU: a algorithm for evaulating the quality of text which has been machine-translated from one natural language to another.
* Word Error Rate: is a measure of the performance of an automatic speech recognition, machine translation etc.
* Longest common substring: given two strings, find the lenght of their longest common substring.
* Levenshtein distance: is a string metric for measuring difference between two sequences. It is the minumum number of single character edits, including insertions, deletions and substitutions required to change one word into the other.


# Example usage:
`python pi_logreg.py`

