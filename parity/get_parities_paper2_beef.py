import pandas as pd
from numpy import load
import numpy as np
import pickle
import re
from os.path import exists
import warnings
import os
import matplotlib.pyplot as plt
import datetime
from scipy.stats import linregress
from pymatgen.core.periodic_table import Element

def make_subplot_beef(subplot, df, args) -> dict:
    """Helper function for larger plot generation. Processes each subplot."""
    x1 = df["corrected_energy_y"].tolist()
    y1 = df["y_min_energy"].tolist()
    x2 = df["corrected_energy_x"].tolist()
    y2 = df["x_min_energy"].tolist()
    
    MAE1 = 0.14
    MAE2 = 0.16

    subplot.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    subplot.plot(
        [-1, 0],
        [
            -1 + 2*MAE1,
            0 + 2*MAE1,
        ],
        args["*OH_color"] + "--",
        linewidth=2,
    )
    subplot.plot(
        [-2, -1],
        [
            -2 + 2*MAE2,
            -1 + 2*MAE2,
        ],
        args["*CO_color"] + "--",
        linewidth=2,
    )
    subplot.plot(
        [-1, 0],
        [
            -1 - 2*MAE1,
            0 - 2*MAE1,
        ],
        args["*OH_color"] + "--",
        linewidth=2,
    )
    
    subplot.plot(
        [-2, -1],
        [
            -2 - 2*MAE2,
            -1 - 2*MAE2,
        ],
        args["*CO_color"] + "--",
        linewidth=2,
    )

    subplot.text(-1.8, -0.2, "*CO", fontsize = 13)
    subplot.text(-1.8, -0.6, "*OH", fontsize = 13)
    subplot.legend(
        [
            "y = x",
            "y = x +/- 2MAE(*CO)",
            "y = x +/- 2MAE(*OH)",
        ],
        loc="lower right",
        fontsize = 13,
    )
    
    subplot.scatter(x1, y1, s=50, facecolors="none", edgecolors=args["*OH_color"])
    subplot.scatter(x2, y2, s=50, facecolors="none", edgecolors=args["*CO_color"])
    subplot.axis("square")
    subplot.set_xlim([-2,0])
    subplot.set_ylim([-2,0])
    subplot.set_xticks(list(range(*[-2,0])))
    subplot.set_yticks(list(range(*[-2,0])))
    subplot.set_xlabel("DFT (BEEF-vdW) adsorption energy" + " [eV]", fontsize = 14)
    subplot.set_ylabel("ML predicted (RPBE) adsorption energy" + " [eV]", fontsize = 14)
    
def make_subplot_unary(subplot, df, args) -> dict:
    """Helper function for larger plot generation. Processes each subplot."""
    
    x = df['cathub_co'].tolist()
    y = df['catlas_co'].tolist()
   
    n = df.el.tolist()
    subplot.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    subplot.plot([-2, 0], [-2+2*args["*CO_MAE"], 2*args["*CO_MAE"]], args["*CO_color"]+"--", linewidth=2)
    subplot.plot([0, 2], [2*args["*OH_MAE"], 2+2*args["*OH_MAE"]], args["*OH_color"]+"--", linewidth=2)
    subplot.plot([-2, 0], [-2-2*args["*CO_MAE"], -2*args["*CO_MAE"]], args["*CO_color"]+"--", linewidth=2)
    subplot.plot([0, 2], [-2*args["*OH_MAE"], 2-2*args["*OH_MAE"]], args["*OH_color"]+"--", linewidth=2)
    labels_str = "*CO" + '_'+ "labels"
    labels = args[labels_str]
    labelx = labels['x']
    labely = labels['y']
    MAE = args["*CO_MAE"]

    for i, txt in enumerate(n):
        subplot.annotate(txt, (labelx[i], labely[i]), fontsize = 12)
        subplot.scatter(x[i], y[i], s=75, facecolors="none", edgecolors=args["*CO_color"], marker = args["CO_markers"][i])
    
    x2 = df['cathub_oh'].tolist()
    y2 = df['catlas_oh'].tolist()
    labels_str2 = "*OH" + '_'+ "labels"
    labels2 = args[labels_str2]
    labelx2 = labels2['x']
    labely2 = labels2['y']
    MAE = args["*OH_MAE"]

    for i, txt in enumerate(n):
        subplot.annotate(txt, (labelx2[i], labely2[i]), fontsize = 12)
        subplot.scatter(x2[i], y2[i], s=50, facecolors="none", edgecolors=args["*OH_color"], marker = args["OH_markers"][i])
           
    subplot.text(-1.8, 0.8, "*CO", fontsize = 13)
    subplot.text(-1.8, 0.4, "*OH", fontsize = 13)
    subplot.legend(
        [
            "y = x",
            "y = x +/- 2MAE(*CO)",
            "y = x +/- 2MAE(*OH)",
        ],
        loc="lower right",
        fontsize = 13,
    )
    subplot.axis("square")
    subplot.set_xlim([-2,1])
    subplot.set_ylim([-2,1])
    subplot.set_xticks(list(range(-2,1)), fontsize = 13)
    subplot.set_yticks(list(range(-2,1)), fontsize = 13)
    subplot.set_xlabel("DFT (BEEF-vdW) adsorption energy" + " [eV]", fontsize = 14)
    subplot.set_ylabel("ML predicted adsorption energy" + " [eV]", fontsize = 14)

def get_paper_parity_plot(
    smile1: str,
    smile2: str,
    df: pd.DataFrame,
    df_unary: pd.DataFrame,
    unary_args: dict,
    lims: list,
    plot_file_path: str,
):
    """Creates the pdf parity plot for a given smile and returns a dictionary summarizing plot results"""

    # Initialize splits and output dictionary
    f, [ax1, ax2] = plt.subplots(1, 2)#, sharey=True)

    # Process data for random calcs
    make_subplot_beef(
        ax1, df, unary_args
    )
    # Plot unary
    make_subplot_unary(
        ax2, df_unary, unary_args
    )

    f.set_figwidth(18)
    f.set_figheight(6)
    f.savefig(plot_file_path)
    plt.close(f)
    
def get_successful(status):
    bools = list(status.values())
    inv_bools = [not i for i in bools]
    return all(inv_bools)

# Load inputs and define global vars
if __name__ == "__main__":

    ### Load the data
    df_random_beefs = pd.read_pickle("data/DFT_data_merged.pkl")
    df_unary = pd.DataFrame([{"el": 'Rh', "cathub_co": -1.71, "cathub_oh" :0.25, 'catlas_co': -1.777716875,'catlas_oh': 0.410108},
                          {"el": 'Ag', "cathub_co": -0.1, "cathub_oh" :0.68, 'catlas_co':-0.1069719,'catlas_oh': 0.623195},
                          {"el": "Cu", "cathub_co": -0.41, "cathub_oh" :0.33, 'catlas_co': -0.631663,'catlas_oh': 0.211547},
                          {"el": "Pd", "cathub_co": -1.74, "cathub_oh" :0.64, 'catlas_co': -1.85869,'catlas_oh': 0.794797},
                          {"el": "Pt", "cathub_co": -1.46, "cathub_oh" :0.98, 'catlas_co': -1.5567218,'catlas_oh': 0.946563},
                          {"el": "Ir", "cathub_co": -1.78, "cathub_oh" :0.54, 'catlas_co': -1.8868656,'catlas_oh': 0.660405},
                         ])

    
    unary_args = {"*CO_MAE": 0.16, "*OH_MAE": 0.14, "*CO_labels": {"x": [-1.855, -0.16, -0.41, -1.72, -1.575, -1.95],
                  "y": [-1.67, 0.05, -0.53, -1.9, -1.41, -1.85]},
                  "*OH_labels":{"x": [0.27, 0.73, 0.35, 0.66, 0.9, 0.54], "y":[0.48, 0.66, 0.22, 0.88, 0.7, 0.79]},
                  "*OH_color": "r", "*CO_color": "b",
                  "OH_markers": ["^", "^", "^", "^", "^", "^"],
                  "CO_markers": ['s', 'X', 's', 's', 's', 'd']}

    ### Generate plot:
    get_paper_parity_plot('*CO', '*OH', df_random_beefs, df_unary, unary_args, [-4,2], 'beef_parities.svg')