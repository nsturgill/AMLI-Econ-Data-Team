# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:18:28 2019

@author: Natal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt



#### try 6 clusters on income bcause of IRS breakdown
#### cluster by zipcode-state (state as answer?), income brackets (above)





data = pd.read_csv("Desktop/final_project_data/cleaneddata.csv")

data.head()
datalab = data.columns

estimatedat = data.iloc[:,np.r_[6:51]]

scaled = scale(estimatedat)




for i in range(len(estimatedat.columns.values)):
    newdf = pd.DataFrame()
    newdf["income"] = scale(estimatedat["Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"])
    newdf = pd.concat([newdf,pd.DataFrame(scaled).iloc[:,i]], axis = 1)
    

    model = KMeans(n_clusters=50)
    model = model.fit(newdf)




    predicted_clusters = model.predict(newdf)

    predicted_clusters


    label_counts = {i:0 for i in range(50)}
    data["Statecat"] = data["State"].astype('category')
    data["Statecat"] = data["Statecat"].cat.codes
    labels = data.loc[:,data.columns == "Statecat"]
    labels = labels["Statecat"].tolist()
    
    
    df = pd.DataFrame()
    df['modeled_cluster'] = predicted_clusters
    df['state'] = labels
    df["real_state"] = 0
    
    print(df.groupby('modeled_cluster')['state'].agg(lambda x: x.value_counts().index[0]))
       #ASK ABOUT THIS
       #USe tableau to show  hwo zips are actually categorized by state - neeed to do with full model
       #and then how the algorithm classifies
       #then go look at the internal relationships of the "states" to see what is different about them
    
    
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score
    
    homogeneity = homogeneity_score(labels, model.labels_)
    completeness = completeness_score(labels, model.labels_)
    homogeneity, completeness
    
    scaled = pd.DataFrame(scale(estimatedat))
    scaled.columns = estimatedat.columns
    
    plt.subplot(5,9,i+1)
    plt.scatter(scaled.loc[:,"Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"], scaled.iloc[:,i], c=predicted_clusters, s=50, cmap='viridis')
    
