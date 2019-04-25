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
#### cluster by zipcode-state (state as answer?), zhvi, income brackets (above)


###THIS IS ZIP CODE



data = pd.read_csv("final_project_data/cleaneddata.csv")
featdata = pd.read_csv("final_project_data/features_and_response.csv")
data.head()
datalab = data.columns

#estimatedat = data.iloc[:,np.r_[6:51]] # pulled out non-geographic features including income 


scaled = scale(featdata)

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans

model = KMeans(n_clusters=51)
model = model.fit(scaled)    # fit model using scaled data



predicted_clusters = model.predict(scaled) #predictions from model 

predicted_clusters


#label_counts = {i:0 for i in range(50)}
 #convert state into numerical categorical variables
data["Statecat"] = data["State"].astype('category') 
data["Statecat"] = data["Statecat"].cat.codes    #actual converting of columns to number
labels = data.loc[:,data.columns == "Statecat"]    #pull labels out of dataframe
labels = labels["Statecat"].tolist()    #convert series to list

#created a dataframe that shows the "modeled state" and the original state in numeric form
df = pd.DataFrame()
df['modeled_cluster'] = predicted_clusters
df['state'] = labels

#these are the states that the cluster sees. color  by this
clusterassign = df.groupby('modeled_cluster')['state'].agg(lambda x: x.value_counts().index[0])


#transform clusterassign into a dictionary
clusterdict = { i : clusterassign[i] for i in range(0, len(clusterassign) ) }
df["eststate"] = df["modeled_cluster"].map(clusterdict)

stateassign = data.groupby('State').agg('count')

statedict= {i: stateassign.index[i] for i in range(51)}

df['full_state_cluster']=df['eststate'].map(statedict)


from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

#creating clustering metrics
homogeneity = homogeneity_score(labels, model.labels_)
completeness = completeness_score(labels, model.labels_)
homogeneity, completeness


scaled = pd.DataFrame(scale(featdata))
scaled.columns = featdata.columns
plt.scatter(scaled.loc[:,"Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"], scaled.loc[:,"Zhvi"], c=predicted_clusters, s=50, cmap='viridis')


outp = pd.concat([data, df], axis = 1)

outp.to_csv(path_or_buf = "final_project_data/full_cluster_state_dat.csv")
