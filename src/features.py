# -*- coding: utf-8 -*-
"""Different features for claim detection models. 

This model contains various features used in different claim detection 
models to detect if a sentence contains a claim or not.


Example:
	The features are implemented as `sklearn` estimator classes so that
    they can easily be used in pipelines like that:

        $ from sklearn.compose import ColumnTransformer
        $ from features import POSTagDistribution
        $ 
        $ column_transformer = ColumnTransformer(
        $    [("POS_tag", POSTagDistribution(), "Sentence")]
        $   )

"""
import os
import re

import numpy as np
import spacy
import nltk  # type: ignore
from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore

from sklearn.base import BaseEstimator  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore
from textblob import TextBlob  # type: ignore
import fasttext

from config import NLTK_DATA_PATH, SPACY_DATA_PATH, FASTTEXT_PATH, FASTTEXT_BIN_MODEL_PATH
from src.dataset import preprocess_dataset

nltk.data.path.append(NLTK_DATA_PATH)

class FastTextPreprocessing(BaseEstimator):
    """Prepare the dataset for fasttext"""

    def get_feature_names(self):
        return [self.__class__.__name__]

    def fasttext_preprocessing(self, document):
        """Preprocessing pipeline from: https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/"""
        document = re.sub(r'\W', ' ', str(document))  # Remove all the special characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)  # remove all single characters
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)  # Remove single characters from the start
        document = re.sub(r'\s+', ' ', document, flags=re.I)  # Substituting multiple spaces with single space
        document = re.sub(r'^b\s+', '', document)  # Removing prefixed 'b'
        document = document.lower()  # Converting to Lowercase

        en_stop = set(stopwords.words('english'))
        
        # Lemmatization
        tokens = document.split()
        tokens = [self.stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    def fit(self, X, y):
        self.stemmer = WordNetLemmatizer()
        return self

    def transform(self, X, y, name):
        path = os.path.join(FASTTEXT_PATH, "dataset_" + name + ".txt")
        with open(path, 'w', encoding='utf-8') as outFile:
            for sentence, label in zip(X, y):
                preprcessed_sentence = self.fasttext_preprocessing(sentence)
                preprcessed_label = "__label__claim" if label == True else "__label__no_claim"

                processed_data = preprcessed_label + " " + preprcessed_sentence

                outFile.write(processed_data)
                outFile.write("\n")

        return path


class FasttextSentenceVector(BaseEstimator):
    """Sentence Vector encoding"""

    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        self.ft = fasttext.load_model(FASTTEXT_BIN_MODEL_PATH)
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            sentence_vector = self.ft.get_sentence_vector(sentence)
            results.append(sentence_vector)
        return np.array(results)


class ThatToken(BaseEstimator):
    """THAT encoding"""

    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            if "that" in sentence.lower():
                results.append([1])
            else:
                results.append([0])
        return np.array(results)


class Sentiment(BaseEstimator):
    """Sentiment encoding"""

    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        self.sia = SentimentIntensityAnalyzer()
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            sentiments = self.sia.polarity_scores(sentence)
            results.append(list(sentiments.values()))
        return np.array(results)


class Subjectivity(BaseEstimator):
    """Subjectivity encoding"""

    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            claim = TextBlob(sentence)  # 0.0 is very objective and 1.0 is very subjective
            subjectivity = claim.sentiment.subjectivity
            results.append([subjectivity])
        return np.array(results)


class SentenceTopicSimilarity(BaseEstimator):
    """Topic, Sentence similarity encoding"""

    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        collumns = [X[column] for column in X.columns]
        sentence_topic = collumns[0] + collumns[1]  # generalize for more collumns
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(sentence_topic)
        return self

    def transform(self, X):
        results = []
        collumns = [X[column] for column in X.columns]
        for sentence, topic in zip(collumns[0], collumns[1]):
            sentence_vec = self.tfidf.transform([sentence])
            topic_vec = self.tfidf.transform([topic])
            similarity = cosine_similarity(sentence_vec, topic_vec)
            results.append([similarity[0][0]])
        return np.array(results)


"""
Liebeck et al. (2016) -> SVM
- Unigrams
- L2 Normalized POS Tag distribution of STTS
- L2 Normalized POS Tag dependencies TIGER Schema
"""
# unigrams

# POS Tags distribution
class POSTagDistribution(BaseEstimator):
    """POS Distribution encoding"""

    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        self.empty_distribution = {
            k: 0 for k in range(len(spacy.glossary.GLOSSARY.items()))
        }  # count all possible tags
        self.nlp = spacy.load(SPACY_DATA_PATH)
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            doc = self.nlp(sentence)
            distribution = self.empty_distribution.copy()

            for k, v in doc.count_by(spacy.attrs.POS).items():
                distribution[k] = v

            vector = list(distribution.values())
            results.append(vector)
        normalized = normalize(results)  # l2 normalization
        return normalized


# POS Tag dependencies
class POSDependencyDistribution(BaseEstimator):
    """POS Distribution encoding"""

    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        all_dep = [
            "ROOT",
            "acl",
            "acomp",
            "advcl",
            "advmod",
            "agent",
            "amod",
            "appos",
            "attr",
            "aux",
            "auxpass",
            "case",
            "cc",
            "ccomp",
            "compound",
            "conj",
            "csubj",
            "csubjpass",
            "dative",
            "dep",
            "det",
            "dobj",
            "expl",
            "intj",
            "mark",
            "meta",
            "neg",
            "nmod",
            "npadvmod",
            "nsubj",
            "nsubjpass",
            "nummod",
            "oprd",
            "parataxis",
            "pcomp",
            "pobj",
            "poss",
            "preconj",
            "predet",
            "prep",
            "prt",
            "punct",
            "quantmod",
            "relcl",
            "xcomp",
        ]
        self.empty_distribution = {k: 0 for k in all_dep}  # count all possible tags
        self.nlp = spacy.load(SPACY_DATA_PATH)
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            doc = self.nlp(sentence)
            distribution = self.empty_distribution.copy()

            for k, v in doc.count_by(spacy.attrs.DEP).items():
                name = doc.vocab[k].text
                distribution[name] = v

            vector = list(distribution.values())
            results.append(vector)
        normalized = normalize(results)  # l2 normalization
        return normalized


"""
Levy et al. (2017) -> Claim Sentence Query (CSQ)
- Keyword that
- main concept
- Lexicon
"""


"""
Chakrabarty et al. (2019) -> LSTM
- sentence (LSTM)
"""


"""
Toledo-Ronen et al. (2020) -> mBERT
- BERT embedding
"""
