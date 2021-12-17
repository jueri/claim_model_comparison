# -*- coding: utf-8 -*-
"""This module contains functions related to the dataset on which 
the models are trained and tested.

Example:
    The dataset can simply be loaded with a singe function:
        $ from dataset import load_dataset
        $
        $ X_train, X_test, y_train, y_test = load_dataset(dataset_path=DATASET_2014_PATH, false_class_balance=FALSE_CLASS_BALANCE)

"""
import os
import re

import nltk  # type: ignore
from nltk import sent_tokenize

import pandas as pd  # type: ignore
from config import DATASET_2018_DIR, CLAIMS_PATH, ARTICLE_PATH, NLTK_DATA_PATH, DATASET_2018_PATH, DATASET_2014_PATH
from sklearn.model_selection import train_test_split  # type: ignore


nltk.data.path.append(NLTK_DATA_PATH)


def load_dataset(dataset_path: str, split_size: float=0.2, false_class_balance: float=1.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Array, pd.Array]:
    """Import a prpared dataset, downsample the non claim class and return train and test splits.

    Args:
        dataset_path (str): path to the prepared dataset.
        split_size (float, optional): Relative size of the test data split compared to the train split. Defaults to 0.2.
        false_class_balance (float, optional): Relative size of the non claim class compared to the claim class. Defaults to 1.0.

    Returns:
        tuple[pd.DataFrame, pd.Array]: train and test data frames and label arrays
    """

    data = pd.read_csv(os.path.join(dataset_path))  # load Data

    claims = data[data["Claim"] == True]

    n_samples = int(len(claims) * false_class_balance)
    no_claims = data[data["Claim"] == False].sample(n=n_samples, random_state=42)
    data_sample = pd.concat([claims, no_claims])

    X_train, X_test, y_train, y_test = train_test_split(
        data_sample, data_sample["Claim"], test_size=split_size, random_state=0
    )
    return X_train, X_test, y_train, y_test




def preprocess_dataset_2014():
    """Prepare the dataset IBM_Debater_(R)_CE-ACL-2014.v0 for model training."""

    original_claims = pd.read_excel(CLAIMS_PATH)[["Article", "Claim"]]
    articles = os.listdir(ARTICLE_PATH)  # list of all articles
    frames = []

    for article in articles:
        article_name = article.replace("_", " ")

        # Load and escape claim passages to find
        claim_sentences = original_claims[original_claims["Article"] == article_name]["Claim"]
        claim_sentences = claim_sentences.apply(lambda x: re.escape(x)).to_list()

        # Prepare table
        df = pd.read_csv(os.path.join(ARTICLE_PATH, article), delimiter="\t", names=["Text"], quoting=3)  # load article to table
        df["Article"] = article_name  # add article name
        df["Sentence"] = df.apply(lambda x: sent_tokenize(x["Text"]), axis=1)  # split text to sentences
        df = df.explode("Sentence")  # each sentence one row

        # Clean text
        df["Sentence"] = df["Sentence"].str.replace("[REF]", "", regex=False) 
        df["Sentence"] = df["Sentence"].str.replace("[REF", "", regex=False)
        df["Sentence"] = df["Sentence"].str.replace(" .", ".", regex=False)
        df = df[df["Sentence"].str.len()>5]  # delete "sentences" with only 5 chars

        df = df.drop(["Text"], axis=1)  # drop original text column

        # Add label if topic has claims
        if claim_sentences:  # skip articles without claims
            df["Claim"] = df["Sentence"].str.contains("|".join(claim_sentences))  # Regex search for claims
        else:
            print("No claims in article:", article_name)
            df["Claim"] = False

        frames.append(df)

    sentences = pd.concat(frames).reset_index()[["Article", "Sentence", "Claim"]]
    sentences.to_csv(DATASET_2014_PATH, index=False)


def preprocess_dataset_2018():
    """Prepare the dataset IBM_Debater_(R)_claim_sentences_search for model training."""
    
    names = ["id", "Article", "mc", "Sentence", "query_pattern", "score", "Claim", "url"]
    data = pd.read_csv(os.path.join(DATASET_2018_DIR, "test_set.csv"), names=names)
    data["Claim"] = data["Claim"].apply(lambda x: True if x==1 else False)

        # Clean text
    data["Sentence"] = data["Sentence"].str.replace("[REF]", "", regex=False) 
    data["Sentence"] = data["Sentence"].str.replace("[REF", "", regex=False)
    data["Sentence"] = data["Sentence"].str.replace(" .", ".", regex=False)
    sentences = data[["Article", "Sentence", "Claim"]]
    sentences.to_csv(DATASET_2018_PATH, index=False)    