"""Plot results of model generalisation performance."""

import os
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



if __name__ == '__main__':

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49]

    results_file = os.path.join('results', 'prediction_tests_diff-train-test.csv')
    gen_results = pd.read_csv(results_file)
    mae_results = gen_results[gen_results['Metric']=='gmnMAE']

    correlations_dir = os.path.join('data','analysis','correlations')
    wass_corrs = pd.read_csv(os.path.join(correlations_dir,'wasserstein.csv'),index_col=0)

    models = ['TFT','analysis\linear_0',r'analysis\resmlp_0','analysis\conv_0']
    metric_scores = {}
    for model in models:
        metric_scores[model] = [[float(mae_results[(mae_results['Model Name'] == model) & (mae_results['Train Building'] == t_id)]['L%s'%p_id]) for p_id in UCam_ids] for t_id in UCam_ids]

    # Plot distribution of generalisation scores
    fig,ax = plt.subplots()
    for i,model in enumerate(models):
        ax.violinplot(np.array(metric_scores[model]).flatten(),[i],showmeans=True,showmedians=True)
    ax.set_ylim(0,3)
    ax.set_ylabel('Prediction Quality (gmnMAE)')
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels(models)
    plt.show()

    # Plot correlation between generalisation scores and wasserstein distance
    fig,ax = plt.subplots()
    for model in reversed(models):
        ax.scatter(wass_corrs.to_numpy().flatten(),np.array(metric_scores[model]).flatten())
    ax.set_ylim(0,3)
    ax.set_ylabel('Prediction Quality (gmnMAE)')
    ax.set_xlabel('Wasserstein Similarity Metric')
    plt.show()