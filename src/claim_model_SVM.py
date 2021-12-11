"""
Liebeck et al. (2016) -> SVM
- Unigrams
- L2 Normalized POS Tag distribution of STTS
- L2 Normalized POS Tag dependencies TIGER Schema

              precision    recall  f1-score   support

       False       0.69      0.74      0.71       235
        True       0.74      0.69      0.72       259

    accuracy                           0.71       494
   macro avg       0.72      0.72      0.71       494
weighted avg       0.72      0.71      0.71       494
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from features import *

DATA_PATH = os.path.join("data", "IBM_Debater_(R)_CE-ACL-2014.v0", "CE-ACL_processed.csv")

data = pd.read_csv(os.path.join(DATA_PATH))

claims = data[data["Claim"] == True]
no_claims = data[data["Claim"] == False].sample(n=len(claims), random_state=42)
data_sample = pd.concat([claims, no_claims])

text_features = FeatureUnion(transformer_list=[("unigrams", CountVectorizer())])

column_trans = ColumnTransformer(
    [
        ("unigrams", text_features, "Sentence"),
        ("POS_tag", POSTagDistribution(), "Sentence"),
        ("POS_dep", POSDependencyDistribution(), "Sentence"),
    ],
    remainder="drop",
    verbose=True,
)

pipe = Pipeline(
    [
        ("preprocessing", column_trans),
        ("scaler", StandardScaler(with_mean=False)),
        ("classify", LinearSVC()),
    ],
    verbose=True,
)

X_train, X_test, y_train, y_test = train_test_split(
    data_sample, data_sample["Claim"], test_size=0.2, random_state=0
)

pipe.fit(X_train, y_train)

Y_pred = pipe.predict(X_test)
print(classification_report(y_test, Y_pred))
