# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:14:38 2019

@author: Natal
"""

# -*- coding: utf-8 -*-
"""Sturgill_3/4 Colab: Clustering with K-Means and Scikit Learn

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ABRAffmJDOXo37pbovKEIs3wc2kdQeZo

#### Copyright 2018 Google LLC.
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# Clustering with K-Means

K-Means is an *unsupervised* machine learning algorithm that can be used to group items into clusters.

So far we have only worked with supervised algorithms. Supervised algorithms have training data with labels that identify the numeric value or class for each item. These algorithms use labeled data to build a model that can be used to make predictions.

K-Means clustering is different. The training data is not labeled. Unlabeled training data is fed into the model. The model attempts to find relationships in the data and create clusters based on those relationships. Once these clusters are formed, predictions can be made about which cluster new data items belong to.

The clusters can't easily be labeled in many cases. The clusters are "emergent clusters" which are created by the algorithm and don't always map to groupings that fit our mental model.
"""
#%% [read in data]
import pandas as pd
import numpy as np

data = pd.read_csv("Desktop/final_project_data/cleaneddata.csv")

data.head()

"""### Examine and clean the data

There are many useful columns of data in this data file that could be used for segmenting customers. In this case we'll ignore the common segmentation attributes (gender, age, etc.) and instead focus solely on the product categories and purchase amounts of the customers. The hope is that we can find clusters of customers based on their purchases so that we can fine tune our marketing for each cluster.

For this we'll be looking at the 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', and 'Purchase' fields.

Let's peek at the data types.
"""

print(data.dtypes)
estimatedat = data.iloc[:,np.r_[1,6:50]]

"""'Purchase' is a numeric value that contains the amount spent. The product category fields seem to be encoding a category as numeric value. 'Product_Category_1' holds ints, which fits that theory."""
#%% [look at data]


"""### Perform Clustering

We now have a nice data format containing purchasing information for each customer. To run k-means clustering on the data we simply load [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) from Scikit Learn and ask the model to find a specific number of clusters for us.

Notice that we are scaling the data. Our purchase total and category counts are very different in magnitude. In order not to give the purchase total too much weight we scale the values.
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

model = KMeans(n_clusters=10)
model.fit(scale(estimatedat))

print(model.inertia_)

#%% []
"""Above we asked Scikit Learn to create 10 clusters for us and we then printed out the *inertia* for the resultant clusters. Inertia is the sum of the squared distances of samples to their closest cluster center. Typically the smaller the inertia the better.

But why did we choose 10 clusters? And is the inertia that we received reasonable?

### Find the optimal number of clusters

With just one run of the algorithm it is difficult to tell. k-means is trying to discover things about your data that you do not know. Picking a number of clusters at random isn't the best way to use k-means.

Instead, you should experiment with a few different cluster values and measure the inertia of each. As you increase the number of clusters, your inertia should decrease.
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

#tried 10, 560, 50 and saw elbow between 50 and 200, rerunning 50,200,10
clusters = list(range(50, 200, 10))
inertias = []

scaled_data = scale(estimatedat)

for c in clusters:
  print(c)
  model = KMeans(n_clusters=c)
  model = model.fit(scaled_data)
  inertias.append(model.inertia_)

plt.plot(clusters, inertias)
plt.show()
#%%[]
#80 showed clear dip in both graphs

model = KMeans(n_clusters=80)
model.fit(scale(estimatedat))

print(model.inertia_)


#%% []
"""The resulting graph should start high and to the left and curve down as the number of clusters grows. The initial slope is steep, but begins to level off. Your optimal number of clusters is somewhere in the "elbow" of the graph, as the slope levels.

Once you have this number, you need to then check to see if the number is reasonable for your use case. Say that the 'optimal' number of clusters for our customer segmentation is 20. Is it reasonable to ask our marketing department to market to 20 distinct segments?

And what makes the segments distinct? We only know that specific customers clustered together. Was it due to age? Purchase price? Maybe the groups are formed on unexpected combinations such as "Bought snack food and cosmetics and spent between 100 and 150 USD". What is a good name for that segment?

Clustering the data is often just the start of your journey. Once you have clusters you'll need to look at each group and try to determine what makes them similar. What patterns did the clustering find and will it be useful to you?

## Clustering Example: Classification of Digits