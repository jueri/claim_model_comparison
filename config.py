# -*- coding: utf-8 -*-
"""Config file to specify some setup variables. 

This file contains setup variables the models and features rely on.
"""
import os

# Dataset 2014
DATASET_2014_DIR = os.path.join("data", "IBM_Debater_(R)_CE-ACL-2014.v0")
CLAIMS_PATH = os.path.join(DATASET_2014_DIR, "2014_7_18_ibm_CDCdata.xls")
ARTICLE_PATH = os.path.join(DATASET_2014_DIR, "wiki12_articles")
DATASET_2014_NAME = "IBM_Debater_(R)_CE-ACL-2014.v0"
DATASET_2014_PATH = os.path.join(DATASET_2014_DIR, "CE-ACL_processed.csv")

# Dataset 2018
DATASET_2018_DIR = os.path.join("data", "IBM_Debater_(R)_claim_sentences_search")
DATASET_2018_NAME = "IBM_Debater_(R)_claim_sentences_search"
DATASET_2018_PATH = os.path.join(DATASET_2018_DIR, "claim_sentence_search.csv")

# NLTK
NLTK_DATA_PATH = os.path.join("data", "nltk_data")

# Spacy
SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_DATA_PATH = os.path.join("data", "spacy_data", SPACY_MODEL_NAME)

# Pyserini
PYSERINI_PATH = os.path.join("data", "pyserini")
INDEX_PATH = os.path.join(PYSERINI_PATH, "index")
CLAIM_LEXICON_PATH = os.path.join("data", "claim_lexicon.txt")

# Fasttext
FASTTEXT_PATH = os.path.join("data", "fasttext")
FASTTEXT_BIN_MODEL_PATH = os.path.join("data", "fasttext", "ce.bin")

# WandB 
PROJECT_NAME = "Claim detection models"
