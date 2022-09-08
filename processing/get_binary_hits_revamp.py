import pickle
import pandas as pd
import numpy as np
from dask_kubernetes import KubeCluster
from dask import delayed
import dask
import dask.dataframe as dd
import yaml
from dask.distributed import Client, LocalCluster

def get_status(row, gridy, min_selectivity, x_MAE, y_MAE, MAE_factor):
    selectivity_values = []
    x_E = row.x_min_energy
    y_E = row.y_min_energy
    if not np.isnan(x_E) and not np.isnan(y_E):
        x_min = np.argmin(np.abs(np.array(x_grid)-(x_E-MAE_factor*x_MAE)))
        x_max = np.argmin(np.abs(np.array(x_grid)-(x_E+MAE_factor*x_MAE)))
        y_min = np.argmin(np.abs(np.array(y_grid)-(y_E-MAE_factor*y_MAE)))
        y_max = np.argmin(np.abs(np.array(y_grid)-(y_E+MAE_factor*y_MAE)))
        x_grid_mini = x_grid[x_min:x_max+1]
        y_grid_mini = y_grid[y_min:y_max+1]
        for x_point in x_grid_mini:
            for y_point in y_grid_mini:
                if (((x_point - x_E)**2/(x_MAE*MAE_factor)**2) + ((y_point - y_E)**2/(y_MAE*MAE_factor)**2)) <= 1:
                    x_num = x_grid.index(x_point)
                    y_num = y_grid.index(y_point)
                    selectivity_values.append(gridy[y_num,x_num])
    if len(selectivity_values) >= 1:
        return max(selectivity_values)
    else: 
        return 0

with open('data/20220413_grid.pkl', 'rb') as f:
    grid = pickle.load(f)
    
grid_acetaldehyde = grid
x_grid = list(np.linspace(-2.45,0.25,500))
y_grid = list(np.linspace(-0.75, 1.5, 500))





if __name__ == "__main__":
    df = pd.read_pickle('outputs/post_processed_catlas_data_20220701_vals_w_o_hits.pkl')
    df["y_min_energy"] -= 0.1
    cluster = LocalCluster(n_workers=32, threads_per_worker=1) 
    client = Client(cluster)
    
    # Put the inputs into a dask bag object
    ddf = dd.from_pandas(df, npartitions=1000)

    # Map and apply the desired function
    ddf_mapped = ddf.apply(get_status, axis = 1, args = (grid_acetaldehyde, 0.1, 0.16, 0.14, 2))
    TF_series = ddf_mapped.compute()
    df['hit'] = TF_series
    df.to_pickle('post_processed_catlas_data_20220701_vals_minus_1_OH.pkl')
    print(df)
    
    

