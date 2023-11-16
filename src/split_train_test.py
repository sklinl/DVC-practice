# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 18:26:19 2021

@author: Tracy
"""

import pandas as pd
dataset = pd.read_csv('./data/prepared.csv')
train = dataset.sample(frac = 0.9)
print(train)
test = dataset.drop(train.index)

train.to_csv('./data/train.csv', index=False)
test.to_csv('./data/test.csv', index=False)


# dvc run -n split_train_test -d src/split_train_test.py -d data/preprocessed.csv -o data/train.csv -o data/test.csv python src/split_train_test.py