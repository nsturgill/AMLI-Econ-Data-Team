# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:00:34 2019

@author: Natal
"""


import pandas as pd
import numpy as np


fullclustdat = pd.read_csv("Dropbox/AMLI/final_project_data/full_cluster_quartile_dat.csv")

target_list = fullclustdat["full_quart_cluster"]
feature_list = fullclustdat.iloc[:,np.r_[7:52]]
feature_list["SizeRank"] = pd.to_numeric(feature_list["SizeRank"])

target_name = "full_quart_cluster"
feature_names = feature_list.columns

from sklearn import tree

dt = tree.DecisionTreeClassifier(min_samples_split=1000)

dt.fit(
    feature_list,
    target_list
)

import pydotplus

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import graphviz

#dot_data = StringIO()  

dot_data = tree.export_graphviz(
    dt,
    out_file=None,  
    feature_names=feature_names
)  
graph = graphviz.Source(dot_data) 
#graph = pydotplus.graph_from_dot_data(dot_data)  

graph.render(filename = "full_quartile_dt", directory = "Dropbox/AMLI/final_project_data/") 

#from sklearn import tree
#
#dt = tree.DecisionTreeClassifier(max_depth=2)
#
#dt.fit(
#    iris_df[feature_names],
#    iris_df[target_name]
#)
#
#import pydotplus
#
#from IPython.display import Image  
#from sklearn.externals.six import StringIO  
#
#dot_data = StringIO()  
#
#tree.export_graphviz(
#    dt,
#    out_file=dot_data,  
#    feature_names=feature_names
#)  
#
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#
#Image(graph.create_png())  