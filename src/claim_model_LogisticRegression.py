"""
Levy et al. (2014) -> Logistic regression classifyer
- sentence-topic similarity
- Linguistic expansion
- Keyword that
- sentiment
- subjectivity
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from features import *

DATA_PATH = os.path.join("data", "IBM_Debater_(R)_CE-ACL-2014.v0", "CE-ACL_processed.csv")

data = pd.read_csv(os.path.join(DATA_PATH))

claims = data[data["Claim"] == True]
no_claims = data[data["Claim"] == False].sample(n=len(claims), random_state=42)
data_sample = pd.concat([claims, no_claims])

text_features = FeatureUnion(transformer_list=[("tf-idf", TfidfVectorizer())])

column_trans = ColumnTransformer(
    [
        ("tf-idf", text_features, "Sentence"),
        ("that", ThatToken(), "Sentence"),
        ("sentiment", Sentiment(), "Sentence"),
        ("subjectivity", Subjectivity(), "Sentence"),
        ("similarity", SentenceTopicSimilarity(), ["Sentence", "Article"]),
    ],
    remainder="drop",
    verbose=True,
)

pipe = Pipeline(
    [
        ("preprocessing", column_trans),
        ("scaler", StandardScaler(with_mean=False)),
        ("classify", LogisticRegression(max_iter=200)),
    ],
    verbose=True,
)

X_train, X_test, y_train, y_test = train_test_split(
    data_sample, data_sample["Claim"], test_size=0.2, random_state=0
)

pipe.fit(X_train, y_train)

Y_pred = pipe.predict(X_test)
print(classification_report(y_test, Y_pred))
