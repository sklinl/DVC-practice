# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 18:27:10 2021

@author: Tracy
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
import yaml
from sklearn.ensemble import RandomForestClassifier
import joblib
params = yaml.safe_load(open("params.yaml"))["train"]

n_est = params["n_est"]
min_split = params["min_split"]

train = pd.read_csv('./data/train.csv')
# test = pd.read_csv('./data/test.csv')

X_train, Y_train = train[['Pclass', 'Sex','SibSp','Parch','AgeBand']], train['Survived']
# X_test, Y_test = test[['Pclass', 'Sex','SibSp','Parch','AgeBand']], test['Survived']

clf = RandomForestClassifier(
    n_estimators=n_est, min_samples_split=min_split
)
clf.fit(X_train, Y_train)


joblib.dump(clf, 'model.pkl')

# dvc stage add -n train -p train.n_est,train.min_split -d src/train.py -d data/train.csv -o model.pkl python src/train.py
# dvc exp run --set-param train.min_split=30
# dvc exp run --queue -S train.n_est=200 -S train.min_split=50
