# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 22:41:09 2021

@author: Tracy
"""
import json
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, average_precision_score,accuracy_score, precision_recall_curve
import math

test = pd.read_csv('./data/test.csv')
clf = joblib.load('model.pkl')

X_test, Y_test = test[['Pclass', 'Sex','SibSp','Parch','AgeBand']], test['Survived']
pred = clf.predict(X_test)
acc = accuracy_score(Y_test, pred)
print(acc)


precision, recall, prc_thresholds = precision_recall_curve(Y_test, pred)
fpr, tpr, roc_thresholds = roc_curve(Y_test, pred)

avg_prec = average_precision_score(Y_test, pred)
roc_auc = roc_auc_score(Y_test, pred)

with open('scores.json', "w") as fd:
    json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc, 'accuracy': acc}, fd)



# dvc stage add -n evaluate -d src/evaluate.py -d data/test.csv -d model.pkl -M scores.json python src/evaluate.py

# nth_point = math.ceil(len(prc_thresholds) / 1000)
# prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
# with open('prc.json', "w") as fd:
#     json.dump(
#         {
#             "prc": [
#                 {"precision": p, "recall": r, "threshold": t}
#                 for p, r, t in prc_points
#             ]
#         },
#         fd,
#         indent=4,
#     )

# with open('roc.json', "w") as fd:
#     json.dump(
#         {
#             "roc": [
#                 {"fpr": fp, "tpr": tp, "threshold": t}
#                 for fp, tp, t in zip(fpr, tpr, roc_thresholds)
#             ]
#         },
#         fd,
#         indent=4,
#     )

