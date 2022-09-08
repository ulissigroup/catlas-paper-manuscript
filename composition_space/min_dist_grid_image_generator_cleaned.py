import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import itertools
import copy
import pickle

# This is from stack overflow
def compare_sublists(l, lol):
    for sublist in lol:
        temp = [i for i in sublist if i in l]
        if sorted(temp) == sorted(l):
            return True
    return False

def get_el(bulk_elements):
    return bulk_elements[0]

def get_distance_to_max(row, maxima):
    xmax, ymax = maxima
    x = row.x_min_energy
    y = row.y_min_energy
    return np.sqrt((x-xmax)**2 + (y-ymax)**2)
    
def filter_unwanted_els(els, unwanted = ["Tc"]):
    if len(els) == 2:
        bool1 = els[0] not in unwanted
        bool2 = els[1] not in unwanted
        overall_bool = all([bool1,bool2])
    else:
        overall_bool = els[0] not in unwanted 
    return overall_bool

def get_eah(mpid):
    return eah[mpid]["e_above_hull"]
    

with open("../e_above_hull/e_above_hull_info.pkl", "rb") as f:
    eah = pickle.load(f)


# Open dfs, merge them, and segment those that were classified as hits
df_binaries = pd.read_pickle("../processing/outputs/post_processed_catlas_data_20220701_vals_2.pkl")
df_binaries["e_above_hull"] = df_binaries.bulk_mpid.apply(get_eah)
df_binaries = df_binaries[df_binaries.e_above_hull <= 0.1]
df_unaries = pd.read_pickle('../processing/outputs/post_processed_catlas_data_unary_20220630.pkl')
df_unaries["e_above_hull"] = df_unaries.bulk_mpid.apply(get_eah)
df_unaries = df_unaries[df_unaries.e_above_hull <= 0.1]
# df_unaries.rename(columns = {"min_dE_gemnet_t_direct_h512": "x_min_energy"}, inplace = True)
df_binaries["wanted"] = df_binaries.bulk_elements.apply(filter_unwanted_els)
df_binaries = df_binaries[df_binaries.wanted]
df_unaries["wanted"] = df_unaries.bulk_elements.apply(filter_unwanted_els)
df_unaries = df_unaries[df_unaries.wanted]

df_binaries_111 = df_binaries[df_binaries["111-like"]]
df_binaries_valid = df_binaries_111[~df_binaries_111.x_min_energy.isnull()]
maxima = (-1.3705410821643291, -0.39153306613226446)
df_binaries_valid['distance_to_max'] = df_binaries_valid.apply(get_distance_to_max, axis = 1, args=(maxima,))
print(df_unaries)
df_unaries["distance"] = df_unaries.apply(get_distance_to_max, axis = 1, args = (maxima,))
df_unaries['element'] = df_unaries.bulk_elements.apply(get_el)
df_unaries_g = df_unaries.groupby(by = ["element"]).distance.agg(min)

# Grab the binary element combos to populate the grid
hit_el_combos = df_binaries_valid.bulk_elements.tolist()
distances = df_binaries_valid.distance_to_max.tolist()
el_combos = df_binaries.bulk_elements.tolist()

dist_lookup_dict = {}
for idx, combo in enumerate(hit_el_combos):
    combostr = combo[0] + combo[1]
    if combostr in dist_lookup_dict:
        if distances[idx] < dist_lookup_dict[combostr]:
            dist_lookup_dict[combostr] = distances[idx]
    else:
        dist_lookup_dict[combostr] = distances[idx]

# Deduce what elements to include (will be use to label the plot axes)
flat_combos = [item for sublist in el_combos for item in sublist]
elements = list(np.unique(flat_combos))
el_combos_all = list(itertools.combinations(elements,2))

# Deduce which combinations are not accounted for in MP
combos_not_in_MP = []
for combo in el_combos_all:
    if not compare_sublists(combo, el_combos):
        combos_not_in_MP.append(combo)
        
# Deduce which combinations do not have any 111-like facets
combos_no111 = []
all_111_like_surface_els = df_binaries_111.bulk_elements.tolist()
for combo in el_combos_all:
    if not compare_sublists(combo, all_111_like_surface_els):
        combos_no111.append(combo) 

sorted_elements = ['Pt', 'Ti', 'Re', 'Ni', 'Nb', 'Ru', 'Mo', 'Rh', 'V', 'Zn', 'Ir', 'Pd', 'Mn', 'Os', 'Cr', 'Co', 'Fe', 'Zr', 'Hf', 'W', 'Ta', 'Sc', 'Cu', 'Y', 'Cd', 'Hg', 'Au', 'Ag']

overall_grid = np.empty((len(sorted_elements), len(sorted_elements)))
overall_grid[:] = np.nan
for entry in el_combos_all:
    x1 = sorted_elements.index(entry[0])
    x2 = sorted_elements.index(entry[1])
    if entry[0]+entry[1] in dist_lookup_dict:
        if dist_lookup_dict[entry[0]+entry[1]] <= 1:
            overall_grid[x1,x2] = dist_lookup_dict[entry[0]+entry[1]]
            overall_grid[x2,x1] = dist_lookup_dict[entry[0]+entry[1]]

for idx, element in enumerate(elements):
    if element not in ["Hg", "Ta"]:
        if df_unaries_g[element] < 1:
            overall_grid[idx,idx] = df_unaries_g[element]
        
print(np.max(overall_grid))
# Set unfilled values = nan so they will appear white
# for idx in range(len(sorted_elements)):
#     for idx2 in range(len(sorted_elements)):
#         if overall_grid[idx,idx2] == -1:
#             overall_grid[idx,idx2] = np.nan
          
 # Set combos not appearing as stable in MP as -10 so they appear dark grey
for entry in combos_not_in_MP:
    x1 = sorted_elements.index(entry[0])
    x2 = sorted_elements.index(entry[1])
    overall_grid[x1,x2] = -10
    overall_grid[x2,x1] = -10
    
# Set combos w/o 111 surfaces as 20 so they appear light grey
for entry in combos_no111:
    x1 = sorted_elements.index(entry[0])
    x2 = sorted_elements.index(entry[1])
    if overall_grid[x1,x2] != -10:
        overall_grid[x1,x2] = 20
        overall_grid[x2,x1] = 20
    
# Make the counts figure
fig = plt.figure(figsize=(12, 12), dpi=80)
cmap = copy.copy(matplotlib.cm.get_cmap('plasma'))


im = plt.imshow(overall_grid, interpolation='none', aspect='equal', cmap = cmap,  vmin=0, vmax = 1)

cmap = im.get_cmap()
cmap.set_under('dimgrey')
cmap.set_over('lightgrey')

im.set_cmap(cmap)
ax = plt.gca();

ax.set_xticks(np.arange(0, 28, 1))
ax.set_yticks(np.arange(0, 28, 1))
ax.set_xticks(np.arange(-.5, 28, 1), minor=True)
ax.set_yticks(np.arange(-.5, 28, 1), minor=True)
ax.set_xticklabels(sorted_elements, size = 12)
ax.set_yticklabels(sorted_elements, size = 12)
ax.tick_params(axis='both', which = 'both', length = 0)

cbar = fig.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label('Distance to maximum selectivity [eV]',size=18)
             
ax.grid(which='minor', color='silver', linestyle='-', linewidth=1.5)
plt.savefig('distance_20220701.svg')
print(dist_lookup_dict)