import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

DATA_PATH = Path('./data/aggregated')
DATASET_FNAMES = (
'aggregated_1s_outliers_imputations', 'aggregated_250ms_outliers_imputations', 'aggregated_500ms_outliers_imputations')


def main():
    for dataset_fname in DATASET_FNAMES:
        data = pd.read_csv(f'{DATA_PATH}/{dataset_fname}.csv')
        data.index = data.attr_time
        data = data.drop(['attr_time'], axis=1)
        features = data.iloc[:, :6].columns.values
        normalized_data = data.copy()
        normalized_data[features] = (data[features] - data[features].min()) / (
                data[features].max() - data[features].min())
        normalized_data.describe()

        pca = PCA(n_components=4)
        pca.fit(normalized_data[features])
        pcas = pca.transform(normalized_data[features])

        pcas_series = {f'pca_{i}': pd.Series(pcas.T[i]) for i in range(4)}
        pcas_df = pd.DataFrame(pcas_series)

        data = data.reset_index().join(pcas_df)
        data.to_csv(f"{DATA_PATH}/{dataset_fname}_pcas.csv")

if __name__ == "__main__":
    main()