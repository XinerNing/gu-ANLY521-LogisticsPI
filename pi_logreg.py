# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:37:00 2019

@author: shera
"""

import argparse
import numpy as np


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


def load_X(sent_pairs):
    """Create a matrix where every row is a pair of sentences and every column in a feature.
    Feature (column) order is not important to the algorithm."""

    features = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Levenshtein distance"]

    X = np.zeros((len(sent_pairs), len(features)))
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

    print(f"Found {len(train_texts)} training pairs")
    print(f"Found {len(dev_texts)} dev pairs")

    print("Fitting and evaluating model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_dev_file", type=str, default="sts-dev.csv",
                        help="dev file")
    parser.add_argument("--sts_train_file", type=str, default="sts-train.csv",
                        help="train file")
    args = parser.parse_args()

    main(args.sts_train_file, args.sts_dev_file)