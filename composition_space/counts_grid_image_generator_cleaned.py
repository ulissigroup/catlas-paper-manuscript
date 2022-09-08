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
    

# Open dfs, merge them, and segment those that were classified as hits
thresh = 0.1
df_binaries = pd.read_pickle("/home/jovyan/catlas-paper-figures/processing/outputs/post_processed_catlas_data_20220701_vals_2.pkl")

with open("/home/jovyan/catlas-paper-figures/2d_histograms/e_above_hull_info.pkl", "rb") as f:
    eah = pickle.load(f)
df_binaries["e_above_hull"] = df_binaries.bulk_mpid.apply(get_eah)
df_binaries = df_binaries[df_binaries.e_above_hull <= 0.1]

df_binaries["wanted"] = df_binaries.bulk_elements.apply(filter_unwanted_els)
df_binaries = df_binaries[df_binaries.wanted]

df_binaries_hits = df_binaries[df_binaries.hit >= thresh]
df_binaries_111 = df_binaries[df_binaries["111-like"]]
df_hits_and_111 = df_binaries_hits[df_binaries_hits["111-like"]]

df_unaries = pd.read_pickle('/home/jovyan/catlas-paper-figures/processing/outputs/post_processed_catlas_data_unary_20220630.pkl')
df_unaries["wanted"] = df_unaries.bulk_elements.apply(filter_unwanted_els)
df_unaries["e_above_hull"] = df_unaries.bulk_mpid.apply(get_eah)
df_unaries = df_unaries[df_unaries.e_above_hull <= 0.1]
df_unaries = df_unaries[df_unaries.wanted]

df_unaries = df_unaries[df_unaries["111-like"]]
hit_unary = df_unaries[df_unaries.hit >= thresh].bulk_elements.tolist()
unary_hit_counts = {}
for el in hit_unary:
    if el[0] in unary_hit_counts:
        unary_hit_counts[el[0]] += 1
    else:
        unary_hit_counts[el[0]] = 1 

# Grab the binary element combos to populate the grid
hit_el_combos = df_hits_and_111.bulk_elements.tolist()
el_combos = df_binaries.bulk_elements.tolist()


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

# Construct an initial grid (this will be used to find a sorted grid)
grid = np.zeros((len(elements), len(elements)))
for entry in hit_el_combos:
    if len(entry) == 2:
        x1 = elements.index(entry[0])
        x2 = elements.index(entry[1])
        grid[x1,x2] += 1
        grid[x2,x1] += 1
    elif len(entry) == 1:
        x = elements.index(entry[0])
        grid[x,x] += 1
print(np.max(grid))    
# Find the norm of each row to sort the grid
sorted_indices = np.argsort(np.linalg.norm(grid, axis=1))
sorted_elements = [elements[idx] for idx in sorted_indices]
sorted_elements = sorted_elements[::-1]
print(sorted_elements)

# sorted_elements = ['Pt', 'Ni', 'Ru', 'Co', 'Mo', 'Re', 'Ir', 'Pd', 'Rh', 'Zn', 'Fe', 'V', 'Nb', 'Mn', 'W', 'Ti', 'Ta', 'Cr', 'Os', 'Zr', 'Cu', 'Sc', 'Hf', 'Cd', 'Y', 'Au', 'Hg', 'Ag']

# Reconstruct the grid (this time sorted)
grid_sorted = np.zeros((len(elements), len(elements)))
for entry in hit_el_combos:
    if len(entry) == 2:
        x1 = sorted_elements.index(entry[0])
        x2 = sorted_elements.index(entry[1])
        grid_sorted[x1,x2] += 1
        grid_sorted[x2,x1] += 1

for element in unary_hit_counts:
    x = sorted_elements.index(element)
    grid_sorted[x,x] = unary_hit_counts[element]
        
overall_grid = grid_sorted

# Set unfilled values = nan so they will appear white
for idx in range(len(sorted_elements)):
    for idx2 in range(len(sorted_elements)):
        if overall_grid[idx,idx2] == 0:
            overall_grid[idx,idx2] = np.nan
          
 # Set combos not appearing as stable in MP as -10 so they appear dark grey
for entry in combos_not_in_MP:
    x1 = sorted_elements.index(entry[0])
    x2 = sorted_elements.index(entry[1])
    overall_grid[x1,x2] = -10
    overall_grid[x2,x1] = -10
    
# Set combos not appearing as stable in MP as -10 so they appear dark grey

for entry in combos_no111:
    x1 = sorted_elements.index(entry[0])
    x2 = sorted_elements.index(entry[1])
    if overall_grid[x1,x2] != -10:
        overall_grid[x1,x2] = 20
        overall_grid[x2,x1] = 20
    
# Make the counts figure
fig = plt.figure(figsize=(12, 12), dpi=80)
cmap = copy.copy(matplotlib.cm.get_cmap('viridis_r'))


im = plt.imshow(overall_grid, interpolation='none', aspect='equal', cmap = cmap,  vmin=1, vmax = 10)

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
cbar.set_label('Counts of surfaces qualified as candidates',size=18)
             
ax.grid(which='minor', color='silver', linestyle='-', linewidth=1.5)
plt.savefig('counts_grid_20220701.svg')
print(sorted_elements, overall_grid)
