# -*- coding: utf-8 -*-
"""This module contains functions related to the dataset on which 
the models are trained and tested.

Example:
    The dataset can simply be loaded with a singe function:
        $ from dataset import load_dataset
        $
        $ load_dataset()

"""
import os
import re

import nltk  # type: ignore
from nltk import sent_tokenize

import pandas as pd  # type: ignore
from config import DATA_PATH, CLAIMS_PATH, ARTICLE_PATH, BASE_PATH, NLTK_DATA_PATH
from sklearn.model_selection import train_test_split  # type: ignore


nltk.data.path.append(NLTK_DATA_PATH)


def load_dataset(test_size: float=0.2) -> pd.DataFrame:
    """Function to load the dataset.

    Returns:
        X_train (DatFrame): Train data
        X_test (DatFrame): Test data
        y_train (DatFrame): Train label
        y_test (DatFrame): Test label
    """
    data = pd.read_csv(os.path.join(DATA_PATH))  # load Data

    claims = data[data["Claim"] == True]
    no_claims = data[data["Claim"] == False].sample(n=len(claims), random_state=42)
    data_sample = pd.concat([claims, no_claims])

    X_train, X_test, y_train, y_test = train_test_split(
        data_sample, data_sample["Claim"], test_size=test_size, random_state=0
    )
    return X_train, X_test, y_train, y_test




def preprocess_dataset():
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
    sentences.to_csv(os.path.join(BASE_PATH, "CE-ACL_processed.csv"), index=False)