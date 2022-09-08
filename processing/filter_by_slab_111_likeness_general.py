import pickle
import pandas as pd
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import JmolNN
from dask.distributed import Client, LocalCluster
import dask.bag as db
from dask_kubernetes import KubeCluster

import dask
import yaml

def split_millers(millers):
    return millers[0], millers[1], millers[2]

def get_surface_cn(entry):
    surface_obj = entry["slab_surface_object"]
    slab_atoms = surface_obj.surface_atoms
    slab_structure = AseAtomsAdaptor.get_structure(slab_atoms)
    elements = np.unique(slab_structure.species)
    radius = max([element.van_der_waals_radius for element in elements])
    highest_atom = 0
    for atom in slab_structure:
        if atom.z > highest_atom:
            highest_atom = atom.z
    atoms_to_consider = {"atoms": [], "idxs": []}
    idx = 0
    for atom in slab_structure:
        if atom.z >= highest_atom - 0.5 * radius:
            atoms_to_consider['atoms'].append(atom)
            atoms_to_consider['idxs'].append(idx)
        idx += 1
    nn = JmolNN()
    surface_cn_list = []
    for it in range(len(atoms_to_consider['idxs'])):
        idx = atoms_to_consider['idxs'][it]
        #print(atoms_to_consider['atoms'][it])
        cn_info = nn.get_nn_info(slab_structure, idx)
        buddy_idxs = [site['site_index'] for site in cn_info]
        surface_buddies = [idx_now for idx_now in buddy_idxs if idx_now in atoms_to_consider["idxs"]]
        #print(surface_buddies)
        surface_cn_list.append(len(surface_buddies))
        
    entry["surface_cn_list"] = surface_cn_list
    entry["min_surface_cn"] = min(surface_cn_list)
    surface_cn_bools = [True if cn == 6 else False for cn in surface_cn_list]
    if all(surface_cn_bools):
        entry["111-like"] = True
    else:
        entry["111-like"] = False
    return entry

    
# Use dask to distribute the task of getting the surface coordination

if __name__ == "__main__":
    
    # Open pickle containing surface data so that we may screen by surface characteristics
    surfaces = pd.DataFrame(pd.read_pickle(""))
    surfaces["m1"], surfaces["m2"], surfaces["m3"] = zip(*surfaces.slab_millers.apply(split_millers))
    surfaces = surfaces.round({"slab_shift": 5})

    entries = surfaces.to_dict(orient = "records")
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    # Connect to the cluster
    client = Client(cluster)
    
    entries_bag = db.from_sequence(entries).repartition(npartitions = 50)
    entries_comp = entries_bag.map(get_surface_cn).compute()
    with open('', 'wb') as f:
        pickle.dump(entries_comp,f)