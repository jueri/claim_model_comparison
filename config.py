# -*- coding: utf-8 -*-
"""Config file to specify some setup variables. 

This file contains setup variables the models and features rely on.
"""
import os

# Dataset
BASE_PATH = os.path.join("data", "IBM_Debater_(R)_CE-ACL-2014.v0")
CLAIMS_PATH = os.path.join(BASE_PATH, "2014_7_18_ibm_CDCdata.xls")
ARTICLE_PATH = os.path.join(BASE_PATH, "wiki12_articles")
DATA_PATH = os.path.join(BASE_PATH, "CE-ACL_processed.csv")

# NLTK
NLTK_DATA_PATH = os.path.join("data", "nltk_data")

# Spacy
SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_DATA_PATH = os.path.join("data", "spacy_data", SPACY_MODEL_NAME)

# Pyserini
PYSERINI_PATH = os.path.join("data", "pyserini")
INDEX_PATH = os.path.join(PYSERINI_PATH, "index")
CLAIM_LEXICON_PATH = os.path.join("data", "claim_lexicon.txt")
