from data_preprocessing.imputation_methods import *
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('./data/aggregated')
DATASET_FNAMES = ('aggregated_1s_outliers_imputations', 'aggregated_250ms_outliers_imputations', 'aggregated_500ms_outliers_imputations')
intervals = [1., 0.25, 0.5]

def add_slope(df, column, interval):
    'adds the slope of the '
    slopes = []
    for i in range(len(df)):
        if i ==0:  # prevent nan
            slope = 0
        else:     #dx/dt
            slope = (df[column].iloc[(i)] - df[column].iloc[(i-1)]) / interval  
        slopes.append(slope)
        
    df[column +'_slope'] = slopes
    return df

def main():
    # including accelerations
    attributes =['attr_x', 'attr_y', 'attr_z',
        'attr_azimuth', 'attr_pitch', 'attr_roll', 'attr_x_slope', 'attr_y_slope', 'attr_z_slope',
        'attr_azimuth_slope', 'attr_pitch_slope', 'attr_roll_slope']  

    '''
    # without accelerations
    attributes =['attr_x', 'attr_y', 'attr_z',
        'attr_azimuth', 'attr_pitch', 'attr_roll']
    '''
    for i in range(len(DATASET_FNAMES)):
        dataset_fname = DATASET_FNAMES[i]
        df = pd.read_csv(Path(DATA_PATH / f"{dataset_fname}.csv"), index_col=0)

        for attribute in attributes:
            df = add_slope(df, attribute, intervals[i])

        df.to_csv(DATA_PATH / f"{dataset_fname}_slopes.csv")

if __name__ == "__main__":
    main()