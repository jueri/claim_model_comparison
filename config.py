# -*- coding: utf-8 -*-
"""Config file to specify setup variables. 

This file contains setup variables the models and features rely on.
"""
import os

DATASETS = {
    "dataset_2014": {
        "base_path": os.path.join("data", "IBM_Debater_(R)_CE-ACL-2014.v0"),
        "claim_file": "2014_7_18_ibm_CDCdata.xls",
        "articles_file": "2014_7_18_ibm_CDCdata.xls",
        "name": "IBM_Debater_(R)_CE-ACL-2014.v0",
        "data": "CE-ACL_processed.csv",
    },
    "dataset_2018": {
        "base_path": os.path.join("data", "IBM_Debater_(R)_claim_sentences_search"),
        "name": "IBM_Debater_(R)_claim_sentences_search",
        "data": "claim_sentence_search.csv",
    },
    "dataset_2014_de": {
        "base_path": "data",
        "name": "IBM_Debater_(R)_CE-ACL-2014.v0_translated",
        "data": "CE-ACL_processed_de.csv",
    },
    "dataset_2018_de": {
        "base_path": "data",
        "name": "IBM_Debater_(R)_claim_sentences_search_translated",
        "data": "claim_sentence_search_de.csv",
    },
    "SMC_2000": {
        "base_path": "data",
        "name": "SMC_CDC_2000",
        "data": "SMC_CDC_2000.csv",
    },
    "SMC_1000": {
        "base_path": "data",
        "name": "SMC_CDC_1000",
        "data": "SMC_CDC_1000.csv",
    },
}

# NLTK
NLTK_DATA_PATH = os.path.join("data", "nltk_data")

# Spacy
SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_DATA_PATH = os.path.join("data", "spacy_data", SPACY_MODEL_NAME)

# Pyserini
PYSERINI_PATH = os.path.join("data", "pyserini")
INDEX_PATH = os.path.join(PYSERINI_PATH, "index")
CLAIM_LEXICON_PATH = os.path.join("data", "claim_lexicon.txt")

# FastText
FASTTEXT_PATH = os.path.join("data", "fasttext")
FASTTEXT_BIN_MODEL_PATH = os.path.join("data", "fasttext", "ce.bin")

# WandB
PROJECT_NAME = "Claim detection models"
