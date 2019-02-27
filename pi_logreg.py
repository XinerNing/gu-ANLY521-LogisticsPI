#! /usr/bin/env python

import argparse
import numpy as np
import nltk
from nltk.metrics.distance import edit_distance
from nltk.translate import nist_score, bleu_score
#from scipy.stats.stats import pearsonr  
from difflib import SequenceMatcher  #longest common substring
from sklearn.linear_model import LogisticRegression


def load_sts(sts_data):
    """Read a dataset from a file in STS benchmark format"""
    texts = []
    labels = []
    with open(sts_data, 'r',encoding="utf-8") as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1,t2))

    labels = np.asarray(labels)

    return texts, labels


def sts_to_pi(texts, labels, min_paraphrase=4.0, max_nonparaphrase=3.0):
    """Convert a dataset from semantic textual similarity to paraphrase.
    Remove any examples that are > max_nonparaphrase and < min_nonparaphrase.
    labels must have shape [m,1] for sklearn models, where m is the number of examples"""
    pi_rows = np.where(np.logical_or(labels>=min_paraphrase, labels<=max_nonparaphrase))[0] #gives the index
    # using indexing to get the right rows out of texts
    texts = [texts[i] for i in pi_rows]
    # using indexing to get the right rows out of labels
    pi_y = labels[pi_rows]
    # convert to binary using threshold
    labels = pi_y > max_nonparaphrase #pi_y >3, so pi_y is 1 if paraphase, 0 if nonparaphase
    
    return texts, labels

def nist_func(x,y):
    try:
        return nist_score.sentence_nist([nltk.word_tokenize(x)],nltk.word_tokenize(y))
    except ZeroDivisionError:
        return 0

## function to get the longest common substring
def find_LCS(x,y):
    match = SequenceMatcher(None, x,y).find_longest_match(0, len(x), 0, len(y))
    common_str=x[match.a: match.a + match.size]
    return (len(common_str))

def load_X(sent_pairs):
    """Create a matrix where every row is a pair of sentences and every column in a feature.
    Feature (column) order is not important to the algorithm."""

    features = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Levenshtein distance"]

    X = np.zeros((len(sent_pairs), len(features)))
    lev_dist=[]
    wer_score=[]
    mynist_score=[]
    mybleu_score=[]
    myLCS_score=[]
    for pair in sent_pairs:
        t1,t2=pair
        token1=nltk.word_tokenize(t1)
        token2=nltk.word_tokenize(t2)
        dist=edit_distance(t1,t2)
        dist_new=edit_distance(token1,token2)
        #t1=t1.split()
        #t2=t2.split()
        #mywer=wer(t1.split(),t2.split())
        mynist=nist_func(t1,t2)
        mybleu=bleu_score.sentence_bleu([token1],token2)
        mywer=((dist_new)/len(token1))+((dist_new)/len(token2))
        myLCS = find_LCS(t1,t2)
        lev_dist.append(dist)
        wer_score.append(mywer)
        mynist_score.append(mynist)
        mybleu_score.append(mybleu)
        myLCS_score.append(myLCS)
   
    X = np.zeros((len(sent_pairs), len(features)))
    X[:,0]=mynist_score
    X[:,1]=mybleu_score
    X[:,2]=wer_score
    X[:,3]=myLCS_score
    X[:,4]=lev_dist
    return X


def main(sts_train_file, sts_dev_file):
    """Fits a logistic regression for paraphrase identification, using string similarity metrics as features.
    Prints accuracy on held-out data. Data is formatted as in the STS benchmark"""

    min_paraphrase = 4.0
    max_nonparaphrase = 3.0

    # loading train
    train_texts_sts, train_y_sts = load_sts(sts_train_file)
    train_texts, train_y = sts_to_pi(train_texts_sts, train_y_sts,
      min_paraphrase=min_paraphrase, max_nonparaphrase=max_nonparaphrase)

    train_X = load_X(train_texts)

    # loading dev
    dev_texts_sts, dev_y_sts = load_sts(sts_dev_file)
    dev_texts, dev_y = sts_to_pi(dev_texts_sts, dev_y_sts,
      min_paraphrase=min_paraphrase, max_nonparaphrase=max_nonparaphrase)

    dev_X = load_X(dev_texts)
    
    # train a logistic model using train
    logisticRegr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    #logisticRegr = LogisticRegression()
    logisticRegr.fit(train_X, train_y)
    
    # apply the model in dev and get the accuracy score
    score=logisticRegr.score(dev_X, dev_y)

    print(f"Found {len(train_texts)} training pairs")
    print(f"Found {len(dev_texts)} dev pairs")

    print(f"Fitting and evaluating model. This system scores {score} on dev set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_dev_file", type=str, default="sts-dev.csv",
                        help="dev file")
    parser.add_argument("--sts_train_file", type=str, default="sts-train.csv",
                        help="train file")
    args = parser.parse_args()

    main(args.sts_train_file, args.sts_dev_file)
