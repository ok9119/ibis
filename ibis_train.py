# -*- coding: utf-8 -*-

import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier

from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

df_train = pd.read_csv('./data/ibis_train/train_tokens.csv', sep = ',')

print(df_train.head())

X_train, X_test, y_train, y_test = train_test_split(
                 df_train['tokens'].astype('str'), df_train['TF'],
                 test_size = 0.2, random_state = 42)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

estimators = [('lr', LogisticRegression(class_weight= 'balanced', solver = 'newton-cg')),
             ('knn', KNeighborsClassifier(n_neighbors=5, weights = 'distance'))]


clf = Pipeline([('vect', TfidfVectorizer(analyzer = 'word', ngram_range=(1,3), max_df = 0.7, min_df = 0.001)),
                ('stack', StackingClassifier(estimators = estimators, final_estimator=CatBoostClassifier(), n_jobs=28))])
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))

filename = '/models/g2a_tfidf_lr_knn_cb.joblib.pkl'
_ = joblib.dump(clf, filename, compress=9)
