# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 18:19:35 2021

@author: Tracy
"""


import pandas as pd
import yaml
# params = yaml.safe_load(open("params.yaml"))["prepare"]

# raw_data = params["raw_data"]

dataset = pd.read_csv('./data/titanic.csv')

dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
dataset['AgeBand'] = pd.qcut(dataset['Age'], 5, labels=[0, 1, 2, 3, 4])
dataset = dataset.drop(['Age'], axis=1)
dataset['AgeBand'] = dataset['AgeBand'].fillna(0)
dataset = dataset.drop(['Fare', 'Ticket', 'Cabin', 'Name', 'PassengerId', 'Embarked'], axis=1)

dataset.to_csv('./data/prepared.csv', index=False)

# dvc stage add -n prepare -d src/prepare.py -d data/titanic.csv -o data/prepared.csv python src/prepare.py