import nltk
import numpy as np
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from sklearn.metrics.pairwise import cosine_similarity

NLTK_DATA_PATH = os.path.join("data", "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)
nltk.download("vader_lexicon", NLTK_DATA_PATH)


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
Hassan et al. (2017) -> SVM
- TF-IDF
- POS
- NER
"""


def create_tfidf():
    german_stop_words = stopwords.words("german")
    tfidf = TfidfVectorizer(
        min_df=10, ngram_range=(1, 2), stop_words=german_stop_words
    )  # prepare vectorizer
    # tfidf.fit_transform(df["post"])
    return tfidf


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
