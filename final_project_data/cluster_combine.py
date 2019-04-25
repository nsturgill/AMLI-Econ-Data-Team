# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:01:32 2019

@author: Natal
"""
import pandas as pd

full_state = pd.read_csv("Dropbox/AMLI/final_project_data/full_cluster_state_dat.csv", )
zi_state = pd.read_csv("Dropbox/AMLI/final_project_data/zhvi_income_state_dat.csv")
full_irs = pd.read_csv("Dropbox/AMLI/final_project_data/full_cluster_irs_bracket_dat.csv")
zi_irs = pd.read_csv("Dropbox/AMLI/final_project_data/zhvi_income_irs_bracket_dat.csv")
full_quart = pd.read_csv("Dropbox/AMLI/final_project_data/full_cluster_quartile_dat.csv")
zi_quart = pd.read_csv("Dropbox/AMLI/final_project_data/zhvi_income_quartile_dat.csv")
full_group = pd.read_csv("Dropbox/AMLI/final_project_data/full_cluster_state_group_dat.csv")
zi_group = pd.read_csv("Dropbox/AMLI/final_project_data/zhvi_income_state_group_dat.csv")

if "Unnamed: 0" in full_state.columns:
    full_state.drop(["Unnamed: 0", "Unnamed: 0.1", "eststate", "state", "Statecat", "modeled_cluster"], axis = 1, inplace = True)
    zi_state.drop(["Unnamed: 0", "Unnamed: 0.1", "eststate", "state", "Statecat", "modeled_cluster"], axis = 1, inplace = True)
    full_irs.drop(["Unnamed: 0", "Unnamed: 0.1", "labels", "modeled_cluster"], axis = 1, inplace = True)
    zi_irs.drop(["Unnamed: 0", "Unnamed: 0.1", "labels", "modeled_cluster"], axis = 1, inplace = True)
    full_quart.drop(["Unnamed: 0", "Unnamed: 0.1", "labels", "modeled_cluster"], axis = 1, inplace = True)
    zi_quart.drop(["Unnamed: 0", "Unnamed: 0.1", "labels", "modeled_cluster"], axis = 1, inplace = True)

df = pd.concat([full_state, full_irs["full_irsbrac_cluster"],full_irs["income_level"]], axis = 1)
df = pd.concat([df, zi_irs["zi_irsbrac_cluster"],full_quart["full_quart_cluster"], full_quart["income_quartile"],
                zi_quart["zi_quart_cluster"], zi_state["zi_state_cluster"], full_group["full_sgroup_cluster"], 
                zi_group["zi_sgroup_cluster"], full_group["Group"]], axis = 1)

df.to_csv(path_or_buf = "Dropbox/AMLI/final_project_data/cluster_combined_dat.csv")
