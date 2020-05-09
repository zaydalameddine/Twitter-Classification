# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:39:20 2020

@author: Zayd Alameddine
"""

import pandas as pd

all_tweets = pd.read_json("random_tweets.json", lines=True)

print(len(all_tweets))
print(all_tweets.columns)
print(all_tweets.loc[0]['text'])

