import pandas as pd
from pathlib import Path

from util.visualize_dataset import VisualizeDataset
from feature_engineering.frequency_abstraction import FourierTransformation

DATA_PATH = Path('./data/engineered')
DATASET_FNAMES = ('aggregated_1s_pcas_slopes_clust', 'aggregated_500ms_pcas_slopes_clust', 'aggregated_250ms_pcas_slopes_clust')
MILLISECONDS_PER_INSTANCES = (1000, 500, 250)

def main():
    for dataset_fname, milliseconds_per_instance in zip(DATASET_FNAMES, MILLISECONDS_PER_INSTANCES):
        dataset = pd.read_csv(DATA_PATH / f"{dataset_fname}.csv.gz", compression="gzip", index_col=0)
        dataset.index = pd.to_datetime(dataset.index)

        FreqAbs = FourierTransformation()
        fs = float(1000) / milliseconds_per_instance
        ws = int(float(40*1000) / milliseconds_per_instance)
        periodic_predictor_cols = ("attr_x", "attr_y", "attr_z", "attr_azimuth", "attr_pitch", "attr_roll")
        dataset = FreqAbs.abstract_frequency(dataset, periodic_predictor_cols, ws, fs)

        window_overlap = 0.9
        skip_points = int((1 - window_overlap) * ws)
        dataset = dataset.iloc[::skip_points, :]

        dataset.to_csv(DATA_PATH / f"{dataset_fname}_freq.csv.gz", compression="gzip")

if __name__ == '__main__':
    main()
