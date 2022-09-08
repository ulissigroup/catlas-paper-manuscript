import pickle
import pandas as pd
import numpy as np
from dask_kubernetes import KubeCluster
from dask import delayed
import dask
import dask.dataframe as dd
from dask_kubernetes.objects import make_pod_from_dict
import yaml
from dask.distributed import Client, LocalCluster

def get_status(row, gridy, min_selectivity, x_MAE, y_MAE, MAE_factor):
    selectivity_values = []
    x_E = row.min_dE_gemnet_t_direct_h512
    y_E = row.y_min_energy
    if not np.isnan(x_E):
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


df = pd.read_pickle('all_unary_data_w_o_hit_class.pkl')


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=1) 
    client = Client(cluster)
    
    # Put the inputs into a dask bag object
    ddf = dd.from_pandas(df, npartitions=1000)

    # Map and apply the desired function
    ddf_mapped = ddf.apply(get_status, axis = 1, args = (grid_acetaldehyde, 0.02, 0.16, 0.21, 2.5))
    TF_series = ddf_mapped.compute()
    df['hit'] = TF_series
    df.to_pickle('post_processed_unary_catlas_data_20220506_vals_2p5.pkl')
    
    

