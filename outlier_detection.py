import numpy as np
import pandas as pd
from pathlib import Path

from data_preprocessing.outlier_detection import DistributionBasedOutlierDetection
from util.visualize_dataset import VisualizeDataset

DATA_PATH = Path('./data/aggregated/')
DATASET_FNAMES = ('aggregated_1s', 'aggregated_250ms', 'aggregated_500ms')

def main():
    for dataset_fname in DATASET_FNAMES:
        dataset = pd.read_csv(Path(DATA_PATH / f"{dataset_fname}.csv"), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        outlier_columns = ['attr_x', 'attr_y', 'attr_z', 'attr_lat', 'attr_lng', 'attr_azimuth', 'attr_pitch', 'attr_roll']

        OutlierDistr = DistributionBasedOutlierDetection()

        # choose the suitable c value
        # DataViz = VisualizeDataset(__file__)
        # for col in outlier_columns:
        #     print(f"Applying Chauvenet outlier criteria for column {col}")
        #     data = dataset[dataset[col].notnull()]
        #     for c in range(2, 4):
        #         dataset_chauvenet = OutlierDistr.chauvenet(c, data, col)
        #         DataViz.plot_binary_outliers(dataset_chauvenet, col, col + '_outlier', f"{dataset_fname}_col-{col}_c-{c}")

        for col in outlier_columns:
            data = dataset[dataset[col].notnull()]
            result = OutlierDistr.chauvenet(3, data, col)
            dataset.loc[data.loc[result[f'{col}_outlier'] == True, col].index, col] = np.nan

        dataset.to_csv(DATA_PATH / f"{dataset_fname}_outliers.csv")

if __name__ == "__main__":
    main()