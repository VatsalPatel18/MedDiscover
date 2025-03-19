import pandas as pd
from scipy.stats import ks_2samp, shapiro, mannwhitneyu
import json

def perform_ks_test(data, model1, model2, metric):
    group1 = data[data["Model"] == model1][metric].dropna()
    group2 = data[data["Model"] == model2][metric].dropna()
    ks_stat, ks_p = ks_2samp(group1, group2)
    return ks_stat, ks_p

def perform_shapiro_test(data, model, metric):
    group = data[data["Model"] == model][metric].dropna()
    if len(group) >= 3:
        stat, p = shapiro(group)
        return stat, p
    else:
        return None, None

def perform_mannwhitney_test(data, model1, model2, metric):
    group1 = data[data["Model"] == model1][metric].dropna()
    group2 = data[data["Model"] == model2][metric].dropna()
    stat, p = mannwhitneyu(group1, group2)
    return stat, p

def save_statistical_results(results, filename="statistical_analysis_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
