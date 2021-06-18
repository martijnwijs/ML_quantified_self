from data_preprocessing.imputation_methods import *
import pandas as pd
import numpy as np
from pathlib import Path
from VisualizeDataset import *
import sys
import copy
import argparse
import pyclust
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
#from sklearn_extra.cluster import KMedoids
DATA_PATH = Path('./data/aggregated/')
#DATASET_FNAMES = ('aggregated_1s_outliers_imputations_slopes', 'aggregated_250ms_outliers_imputations_slopes', 'aggregated_500ms_outliers_imputations_slopes')
DataViz = VisualizeDataset(__file__)

cluster_dimensionslist = [['attr_x', 'attr_y', 'attr_z' ],  ['attr_azimuth', 'attr_pitch', 'attr_roll']]

def k_medoids_over_instances(dataset, cols, k, distance_metric, max_iters, n_inits=5, p=1, name=''):
    # If we set it to default we use the pyclust package...
    temp_dataset = dataset[cols]

    km = pyclust.KMedoids(n_clusters=k, n_trials=n_inits)
    #km = KMedoids(n_clusters=8, metric='euclidean', method='alternate', init='heuristic', max_iter=300, random_state=None)
    km.fit(temp_dataset.values)
    cluster_assignment = km.labels_

    # And add the clusters and silhouette scores to the dataset.
    dataset['cluster'+ name] = cluster_assignment
    silhouette_avg = silhouette_score(temp_dataset, np.array(cluster_assignment))
    silhouette_per_inst = silhouette_samples(temp_dataset, np.array(cluster_assignment))
    dataset['silhouette'+ name] = silhouette_per_inst

    return dataset

def plotting_pos(df, name=''):
    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot(111, projection='3d')
    for cluster in df["cluster"+name].unique():
        ax.scatter(df[df["cluster"+name] == cluster]['attr_x'],df[df["cluster"+name] == cluster]['attr_y'],df[df["cluster"+name] == cluster]['attr_z'])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return

def plotting_rot(df, name=''):
    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot(111, projection='3d')
    for cluster in df["cluster"+name].unique():
        ax.scatter(df[df["cluster"+name] == cluster]['attr_azimuth'],df[df["cluster"+name] == cluster]['attr_pitch'],df[df["cluster"+name] == cluster]['attr_roll'])

    ax.set_xlabel('azimuth')
    ax.set_ylabel('pitch')
    ax.set_zlabel('roll')
    plt.show()
    return
def main():
    '''
    for i in range(len(DATASET_FNAMES)): # loop over the datasets
        dataset_fname = DATASET_FNAMES[i]
    '''
    dataset_fname = 'aggregated_1s_outliers_imputations_slopes'
    dataset = pd.read_csv(Path(DATA_PATH / f"{dataset_fname}.csv"), index_col=0)
    #dataset = dataset.sample(n=4000)
    dataset_try = dataset.sample(n=6000)  #because of the long running time, a smaller part of the dataset is used for silhouttes
    names = ['_pos', '_rot']
    index= 0

    # clustering twice, once over x y z, once over the rotations
    for cluster_dimensions in cluster_dimensionslist:

        # Do some initial runs to determine the right number for k
        k_values = range(2, 6)
        silhouette_values = []
        print('===== k medoids clustering =====')
        
        for k in k_values:
            print(f'k = {k}')
            
            dataset_cluster = k_medoids_over_instances(copy.deepcopy(
                dataset_try), cluster_dimensions, k, 'default', 20, n_inits=10)

            
            silhouette_score = dataset_cluster['silhouette'].mean()
            print(f'silhouette = {silhouette_score}')
            silhouette_values.append(silhouette_score)

        plot = DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                        ylim=[0, 1], line_styles=['b-'])
        

        # And run k medoids with the highest silhouette score
        k = k_values[np.argmax(silhouette_values)]
        print(f'Highest K-Medoids silhouette score: k = {k}')

        dataset = k_medoids_over_instances(copy.deepcopy(dataset), cluster_dimensions, k, 'default', 20, n_inits=50, name=names[index])
        print(dataset)
        #DataViz.plot_clusters_3d(dataset_kmed, ['attr_x', 'attr_y', 'attr_z'], 'cluster', ['label'])
        if index ==0:
            plotting_pos(dataset, name=names[index]) # plot position
        else:
            plotting_rot(dataset, name=names[index]) # plot rotation
        DataViz.plot_silhouette(dataset, 'cluster'+names[index], 'silhouette'+names[index])
        #util.print_latex_statistics_clusters(dataset, 'cluster', cluster_dimensions, 'label')
        index += 1
    # save as csv
    dataset.to_csv(dataset_fname + '_cluster.csv')

if __name__ == "__main__":
    main()