# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:00:04 2019

@author: Natal
"""
import pandas as pd
import numpy as np
data = pd.read_csv("Desktop/final_project_data/cleaneddata.csv")

locationdat = data.iloc[:,np.r_[1:6]]
featuredat = data.iloc[:,np.r_[3,6:51]]


featuredat = featuredat.drop(['Estimate; INCOME AND BENEFITS (IN 2017 INFLATION-ADJUSTED DOLLARS) - Total households - Mean household income (dollars)'], axis = 1)

locationdat.to_csv(path_or_buf = "Desktop/final_project_data/location_only.csv")
featuredat.to_csv(path_or_buf = "Desktop/final_project_data/features_and_response.csv")