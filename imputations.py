from data_preprocessing.imputation_methods import *
import pandas as pd
from pathlib import Path

DATA_PATH = Path('./data/preprocessed')
DATASET_FNAMES = ('aggregated_1s_outliers', 'aggregated_250ms_outliers', 'aggregated_500ms_outliers')

def main():
    imp = ImputationMissingValues()
    for dataset_fname in DATASET_FNAMES:
        dataset = pd.read_csv(Path(f"{DATA_PATH}/{dataset_fname}.csv.gz"), compression="gzip", index_col=0)
        dataset.index = pd.to_datetime(dataset.index)

        # interpolation
        columns_interpolate = ['attr_x', 'attr_y', 'attr_z', 'attr_azimuth', 'attr_pitch', 'attr_roll'] 
        for col in columns_interpolate: 
            df = imp.impute_interpolate(dataset, col)
        
        # Drop lat and long 
        df = df.drop(['attr_lat', 'attr_lng'], axis=1)
        
        # to csv file
        df.to_csv(f"{DATA_PATH}/{dataset_fname}_imp.csv.gz", compression="gzip")

if __name__ == "__main__":
    main()
    
