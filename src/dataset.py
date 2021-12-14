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

import nltk
from nltk import sent_tokenize

import pandas as pd
from config import DATA_PATH, CLAIMS_PATH, ARTICLE_PATH, BASE_PATH
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_dataset() -> pd.DataFrame:
    """Function to load the dataset.

    Returns:
        pd.DatFrame: Dataframe with all data.
    """
    data = pd.read_csv(os.path.join(DATA_PATH))  # load Data

    claims = data[data["Claim"] == True]
    no_claims = data[data["Claim"] == False].sample(n=len(claims), random_state=42)
    data_sample = pd.concat([claims, no_claims])
    return data_sample


stemmer = WordNetLemmatizer()

def fasttext_preprocessing(document):
    """Preprocessing pipeline from: https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/"""
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    en_stop = set(stopwords.words('english'))
    
    # Lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text



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