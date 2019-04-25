
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


#### try 6 clusters on income bcause of IRS breakdown





data = pd.read_csv("Dropbox/AMLI/final_project_data/cleaneddata.csv")

data.head()
datalab = data.columns

estimatedat = data.loc[:,["Zhvi","Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"]]

scaled = scale(estimatedat)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=51)
model = model.fit(scaled)


predicted_clusters = model.predict(scaled)

predicted_clusters

# Create a dictionary with the keys 0 through 9 with 0 values
data["Statecat"] = data["State"].astype('category')
data["Statecat"] = data["Statecat"].cat.codes
labels = data.loc[:,data.columns == "Statecat"]
labels = labels["Statecat"].tolist()

df = pd.DataFrame()
df['modeled_cluster'] = predicted_clusters
df['state'] = labels



clusterassign = df.groupby('modeled_cluster')['state'].agg(lambda x: x.value_counts().index[0])
#these are the states that the cluster sees. color  by this


clusterdict = { i : clusterassign[i] for i in range(0, len(clusterassign) ) }
df["eststate"] = df["modeled_cluster"].map(clusterdict)

stateassign = data.groupby('State').agg('count')

statedict= {i: stateassign.index[i] for i in range(51)}

df['zi_state_cluster']=df['eststate'].map(statedict)

from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

homogeneity = homogeneity_score(labels, model.labels_)
completeness = completeness_score(labels, model.labels_)
homogeneity, completeness

scaled = pd.DataFrame(scale(estimatedat))
scaled.columns = estimatedat.columns
plt.scatter(scaled.loc[:,"Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)"], scaled.loc[:,"Zhvi"], c=predicted_clusters, s=50, cmap='viridis')



outp = pd.concat([data, df], axis = 1)

outp.to_csv(path_or_buf = "Dropbox/AMLI/final_project_data/zhvi_income_state_dat.csv")
