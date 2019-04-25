# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:11:38 2019

@author: Natal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale




data = pd.read_csv("Dropbox/AMLI/final_project_data/cleaneddata.csv")
featdata = pd.read_csv("Dropbox/AMLI/final_project_data/features_and_response.csv")
state_group = pd.read_csv("Dropbox/AMLI/final_project_data/us census bureau regions and divisions.csv")
state_group.columns = ["State", "Code", "Region", "Division"]
data.head()
datalab = data.columns

scaled = scale(featdata)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=9)
model = model.fit(scaled)


predicted_clusters = model.predict(scaled)

predicted_clusters

groupdict = pd.Series(state_group.Division.values,index=state_group.Code).to_dict()
data["Group"] = data["State"].map(groupdict)




df = pd.DataFrame()
df['modeled_cluster'] = predicted_clusters
df['labels'] = data["Group"]



clusterassign = df.groupby('modeled_cluster')['labels'].agg(lambda x: x.value_counts().index[0])
#these are the states that the cluster sees. color  by this


clusterdict = { i : clusterassign[i] for i in range(0, len(clusterassign) ) }
df["full_sgroup_cluster"] = df["modeled_cluster"].map(clusterdict)





from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

homogeneity = homogeneity_score(labels, model.labels_)
completeness = completeness_score(labels, model.labels_)
homogeneity, completeness

scaled = pd.DataFrame(scale(featdata))
scaled.columns = featdata.columns
plt.scatter(scaled.loc[:,"Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"], scaled.loc[:,"Zhvi"], c=predicted_clusters, s=50, cmap='viridis')



outp = pd.concat([data, df], axis = 1)

outp.to_csv(path_or_buf = "Dropbox/AMLI/final_project_data/full_cluster_state_group_dat.csv")
