# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:09:18 2019

@author: Natal
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:30:48 2019

@author: Natal



## Clustering Example: Classification of Digits

Clustering for data exploration purposes can lead to interesting insights into your data, but clustering can also be used for classification purposes.

In the example below we'll try to use k-means clustering to predict handwritten digits.

### Load the data

We'll load the digits dataset packaged with Scikit Learn.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


#### try 6 clusters on income bcause of IRS breakdown





data = pd.read_csv("final_project_data/cleaneddata.csv")

data.head()
datalab = data.columns

estimatedat = data.loc[:,["Zhvi","Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"]]

scaled = scale(estimatedat)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=6)
model = model.fit(scaled)


predicted_clusters = model.predict(scaled)

predicted_clusters

# Create a dictionary with the keys 0 through 9 with 0 values
label_counts = {i:0 for i in range(6)}
data['income_level'] = pd.cut(data['Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)'], (0, 25000, 50000, 75000, 100000, 200000, 1000000)).astype('category').cat.codes
labels = data['income_level']

df = pd.DataFrame()
df['modeled_cluster'] = predicted_clusters
df['labels'] = labels



clusterassign = df.groupby('modeled_cluster')['labels'].agg(lambda x: x.value_counts().index[0])
#these are the states that the cluster sees. color  by this


clusterdict = { i : clusterassign[i] for i in range(0, len(clusterassign) ) }
df["zi_irsbrac_cluster"] = df["modeled_cluster"].map(clusterdict)





from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

homogeneity = homogeneity_score(labels, model.labels_)
completeness = completeness_score(labels, model.labels_)
homogeneity, completeness

scaled = pd.DataFrame(scale(estimatedat))
scaled.columns = estimatedat.columns
plt.scatter(scaled.loc[:,"Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"], scaled.loc[:,"Zhvi"], c=predicted_clusters, s=50, cmap='viridis')



outp = pd.concat([data, df], axis = 1)

outp.to_csv(path_or_buf = "final_project_data/zhvi_income_irs_bracket_dat.csv")
