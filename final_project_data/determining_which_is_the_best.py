# -*- coding: utf-8 -*-
"""Determining which is the best.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16GPxNkAKtNdn-Aa7ckaFWvf-d3BsIeHu

So the plan is to try to create a classifier using
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# Upload the file from your computer to the colab runtime

#from google.colab import files
#
#uploaded = files.upload()
#
#for fn in uploaded.keys():
#  print('User uploaded file "{name}" with length {length} bytes'.format(
#      name=fn, length=len(uploaded[fn])))
#
## Read the dataset into a pandas dataframe

dataset_filename = "Desktop/final_project_data/cleaneddata.csv"

data = pd.read_csv(dataset_filename, encoding='latin-1')

df = data


#dropped Unnamed: 0 because it has no important information in it
#dropped State, Metro, and County, zipcode
#Keeping City
#dropped median income
if 'zipcode' in df.columns:
  df.drop(columns = ['Unnamed: 0','zipcode','State','Metro', 'County','Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Mean household income (dollars)'],
                  inplace = True)


#categorize the data that contains strings
string_col = ['City']
for category in string_col:
  df[category] = df[category].astype('category').cat.codes
  

# The following code makes income into a categorical variable
# Cuts income into six leves

df['Income_level'] = pd.cut(df['Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)'], 
                            (0, 25000, 50000, 75000, 100000, 200000, 1000000)).cat.codes

from sklearn.model_selection import train_test_split
remove = 'Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)'

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = remove), 
                                                   df['Income_level'],
                                                   test_size = .2, 
                                                   random_state = 42)
outfile = pd.DataFrame(index = names, columns = ['score'])
counter = 0

import time

for name, clf in zip(names, classifiers):
  b4 = time.time()
  clf.fit(X_train, y_train)
  aft = time.time()
  score = clf.score(X_test, y_test)
  print(name)
  print(score)
  time1 = aft-b4
  print(time1)
  outfile.at[name, 'score'] = score
  outfile.at[name, 'time'] = time1
  counter +=1
  
outfile.to_csv(path_or_buf = "Desktop/final_project_data/classificationoutrd2.csv")  

