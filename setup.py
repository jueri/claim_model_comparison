# -*- coding: utf-8 -*-
"""Setup the necessary tools for this repository. 

This script will set up Spacy and NLTK for feature processing. 
The Spacy model `en_core_web_sm` will be downloaded and saved to 
the specified path. Further, NLTK data like the `vader_lexicon` 
and `punct` will be downloaded.

Example:
    Run this setup script with:

        $ python -m setup.py

"""
import os
import subprocess

import nltk  # type: ignore
import spacy  # type: ignore

from config import NLTK_DATA_PATH, SPACY_DATA_PATH, SPACY_MODEL_NAME, FASTTEXT_PATH, DATASET_2014_DIR, DATASET_2018_DIR
from src.dataset import preprocess_dataset_2014, preprocess_dataset_2018

# setup Spacy
if not os.path.exists(SPACY_DATA_PATH):
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])  # download model 
    os.makedirs(SPACY_DATA_PATH)  # create foleder 
    nlp = spacy.load(SPACY_MODEL_NAME)  # load model 
    nlp.to_disk(SPACY_DATA_PATH)  # save model to created dir

# setup NLTK
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
    nltk.download("vader_lexicon", NLTK_DATA_PATH)
    nltk.download("wordnet", NLTK_DATA_PATH)
    nltk.download("stopwords", NLTK_DATA_PATH)
    nltk.download("punkt", NLTK_DATA_PATH)
    nltk.data.path.append(NLTK_DATA_PATH)

if not os.path.exists(FASTTEXT_PATH):
    os.makedirs(FASTTEXT_PATH)

if "CE-ACL_processed.csv" not in os.listdir(DATASET_2014_DIR):
    preprocess_dataset_2014()
    
if "claim_sentence_search.csv" not in os.listdir(DATASET_2018_DIR):
    preprocess_dataset_2018()