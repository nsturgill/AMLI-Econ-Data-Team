# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:47:03 2019

@author: Natal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale





data = pd.read_csv("final_project_data/cleaneddata.csv")

data.head()
datalab = data.columns

estimatedat = data.loc[:,["Zhvi","Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"]]

scaled = scale(estimatedat)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model = model.fit(scaled)


predicted_clusters = model.predict(scaled)

predicted_clusters


q75, q25 = np.percentile(data['Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)'], [75,25])


data['income_quartile'] = pd.cut(data['Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)'], (0, q25, q75, 1000000)).astype('category').cat.codes
labels = data['income_quartile']

df = pd.DataFrame()
df['modeled_cluster'] = predicted_clusters
df['labels'] = labels



clusterassign = df.groupby('modeled_cluster')['labels'].agg(lambda x: x.value_counts().index[0])
#these are the states that the cluster sees. color  by this


clusterdict = { i : clusterassign[i] for i in range(0, len(clusterassign) ) }
df["zi_quart_cluster"] = df["modeled_cluster"].map(clusterdict)





from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

homogeneity = homogeneity_score(labels, model.labels_)
completeness = completeness_score(labels, model.labels_)
homogeneity, completeness

scaled = pd.DataFrame(scale(estimatedat))
scaled.columns = estimatedat.columns
plt.scatter(scaled.loc[:,"Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"], scaled.loc[:,"Zhvi"], c=predicted_clusters, s=50, cmap='viridis')



outp = pd.concat([data, df], axis = 1)

outp.to_csv(path_or_buf = "final_project_data/zhvi_income_quartile_dat.csv")
