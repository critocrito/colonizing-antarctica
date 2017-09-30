#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


dataframe = pd.read_csv("nasa_sea_level_clean.csv")

dataframe.drop(['altimeter-type', 'GMSL-no-gia', 'std-dev-alongtrack-no-gia', 'smoothed-GMSL-no-gia'], 1, inplace=True)

print(dataframe.head())
X0 = np.array(dataframe.drop(['mgd-file-cycle', 'noOfObs', 'noOfWObs','year-fraction', 'std-dev-gia', 'smoothed-GMSL-gia', 'smoothed-GMSL-ann-semiann'], 1))
y0 = np.array(dataframe["year-fraction"])
X1 = np.array(dataframe.drop(['mgd-file-cycle', 'noOfObs', 'noOfWObs','GMSL-gia', 'std-dev-gia', 'smoothed-GMSL-gia', 'smoothed-GMSL-ann-semiann'], 1))
y1 = np.array(dataframe["GMSL-gia"])


X0_train, X0_test, y0_train, y0_test = model_selection.train_test_split(X0, y0, test_size=0.2)
X1_train, X1_test, y1_train, y1_test = model_selection.train_test_split(X1, y1, test_size=0.2)


lrm_y = LinearRegression(n_jobs=-1)
lrm_y.fit(X0_train, y0_train)
lrm_sl = LinearRegression(n_jobs=-1)
lrm_sl.fit(X1_train, y1_train)
accuracy_y = lrm_y.score(X0_test, y0_test)
accuracy_sl = lrm_sl.score(X1_test, y1_test)

print(accuracy_y, accuracy_sl)

print(lrm_y.predict(1000)) # 2284
print(lrm_sl.predict(100000)) # 336842

