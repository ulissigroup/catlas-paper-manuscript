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

def get_elements_in_groups(groups: list) -> list:
    """Grabs the element symbols of all elements in the specified groups"""
    valid_els = []

    if "transition metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_transition_metal]
        valid_els = [*valid_els, *new_valid_els]
    if "post-transition metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_post_transition_metal]
        valid_els = [*valid_els, *new_valid_els]
    if "metalloid" in groups:
        new_valid_els = [str(el) for el in Element if el.is_metalloid]
        valid_els = [*valid_els, *new_valid_els]
    if "rare earth metal" in groups:
        new_valid_els = [str(el) for el in Element if el.is_rare_earth_metal]
        valid_els = [*valid_els, *new_valid_els]
    if "alkali" in groups:
        new_valid_els = [str(el) for el in Element if el.is_alkali]
        valid_els = [*valid_els, *new_valid_els]
    if "alkaline" in groups or "alkali earth" in groups:
        new_valid_els = [str(el) for el in Element if el.is_alkaline]
        valid_els = [*valid_els, *new_valid_els]
    if "chalcogen" in groups:
        new_valid_els = [str(el) for el in Element if el.is_calcogen]
        valid_els = [*valid_els, *new_valid_els]
    if "halogen" in groups:
        new_valid_els = [str(el) for el in Element if el.is_halogen]
        valid_els = [*valid_els, *new_valid_els]

    implemented_groups = [
        "transition metal",
        "post-transition metal",
        "metalloid",
        "rare earth metal",
        "alkali",
        "alkaline",
        "alkali earth",
        "chalcogen",
        "halogen",
    ]

    for group in groups:
        if group not in implemented_groups:
            warnings.warn(
                "Group not implemented: "
                + group
                + "\n Implemented groups are: "
                + str(implemented_groups)
            )
    
    return list(np.unique(valid_els))

def apply_filters(bulk_filters: dict, df: pd.DataFrame) -> pd.DataFrame:
    """filters the dataframe to only include material types specified in the yaml"""

    def get_acceptable_elements_boolean(
        stoichiometry: dict, acceptable_els: list
    ) -> bool:
        elements = set(stoichiometry.keys())
        return elements.issubset(acceptable_els)

    def get_required_elements_boolean(stoichiometry: dict, required_els: list) -> bool:
        elements = list(stoichiometry.keys())
        return all([required_el in elements for required_el in required_els])

    def get_number_elements_boolean(stoichiometry: dict, number_els: list) -> bool:
        element_num = len(list(stoichiometry.keys()))
        return element_num in number_els

    def get_active_host_boolean(stoichiometry: dict, active_host_els: dict) -> bool:
        active = active_host_els["active"]
        host = active_host_els["host"]
        elements = set(stoichiometry.keys())
        return all(
            [
                all([el in [*active, *host] for el in elements]),
                any([el in host for el in elements]),
                any([el in active for el in elements]),
            ]
        )

    for name, val in bulk_filters.items():
        if (
            str(val) != "None"
        ):  # depending on how yaml is created, val may either be "None" or NoneType
            if name == "filter_by_acceptable_elements":
                df["filter_acceptable_els"] = df.stoichiometry.apply(
                    get_acceptable_elements_boolean, args=(val,)
                )
                df = df[df.filter_acceptable_els]
                df = df.drop(columns=["filter_acceptable_els"])
            elif name == "filter_by_required_elements":
                df["filter_required_els"] = df.stoichiometry.apply(
                    get_required_elements_boolean, args=(val,)
                )
                df = df[df.filter_required_els]
                df = df.drop(columns=["filter_required_els"])

            elif name == "filter_by_num_elements":
                df["filter_number_els"] = df.stoichiometry.apply(
                    get_number_elements_boolean, args=(val,)
                )
                df = df[df.filter_number_els]
                df = df.drop(columns=["filter_number_els"])

            elif name == "filter_by_element_groups":
                valid_els = get_elements_in_groups(val)
                df["filter_acceptable_els"] = df.stoichiometry.apply(
                    get_acceptable_elements_boolean, args=(valid_els,)
                )
                df = df[df.filter_acceptable_els]
                df = df.drop(columns=["filter_acceptable_els"])

            elif name == "filter_by_elements_active_host":
                df["filter_active_host_els"] = df.stoichiometry.apply(
                    get_active_host_boolean, args=(val,)
                )
                df = df[df.filter_active_host_els]
                df = df.drop(columns=["filter_active_host_els"])

            elif name == "filter_ignore_mpids":
                continue
            elif name == "filter_by_mpids":
                warnings.warn(name + " has not been implemented for parity generation")
            elif name == "filter_by_object_size":
                continue
            else:
                warnings.warn(name + " has not been implemented")
    return df

def make_subplot(subplot, df, adsorbate, energy_key1, energy_key2, lims) -> dict:
    """Helper function for larger plot generation. Processes each subplot."""
    x = df[energy_key1].tolist()
    y = df[energy_key2].tolist()
    
    
    MAE = sum(abs(np.array(x) - np.array(y))) / len(x)

    subplot.plot([-4, 2], [-4, 2], "k-", linewidth=3)
    subplot.plot(
        [-4, 2],
        [
            -4 + 2*MAE,
            2 + 2*MAE,
        ],
        "k--",
        linewidth=2,
    )
    subplot.plot(
        [-4, 2],
        [
            -4 - 2*MAE,
            2 - 2*MAE,
        ],
        "k--",
        linewidth=2,
    )
    if adsorbate == "*OH":
        ec = 'r'
    else:
        ec = 'b'
    subplot.scatter(x, y, s=25, facecolors="none", edgecolors=ec)
    subplot.text(-3.95, 1.75, adsorbate, fontsize = 13)
    subplot.text(-3.95, 1.05, f"N = {len(x)}", fontsize = 13)
    subplot.text(-3.95, 1.4, f"MAE({adsorbate}) = {MAE:1.2f} eV", fontsize = 13)
    subplot.legend(
        [
            "y = x",
            f"y = x +/- 2MAE({adsorbate})",
        ],
        loc="lower right",
        fontsize = 13,
    )
    subplot.axis("square")
    subplot.set_xlim(lims)
    subplot.set_ylim(lims)
    subplot.set_xticks(list(range(*lims)),fontsize = 13)
    subplot.set_yticks(list(range(*lims)), fontsize = 13)
    subplot.set_xlabel("DFT (RPBE) adsorption energy" + " [eV]", fontsize = 14)
    subplot.set_ylabel("ML predicted adsorption energy" + " [eV]", fontsize = 14)

def get_paper_parity_plot(
    smile1: str,
    smile2: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    lims: list,
    plot_file_path: str,
    energy_key1="DFT_energy",
    energy_key2="ML_energy",
):
    """Creates the pdf parity plot for a given smile and returns a dictionary summarizing plot results"""

    # Filter the data to only include the desired smile
    df_smile_specific1 = df1[df1.adsorbate == smile1]
    df_smile_specific2 = df2[df2.adsorbate == smile2]


    # Initialize splits and output dictionary
    f, [ax1, ax2, ax3] = plt.subplots(1, 3)#, sharey=True)

    # Process data for first smile
    make_subplot(
        ax1, df_smile_specific1, smile1, energy_key1, energy_key2, lims
    )

    # Process data for second smile
    make_subplot(
        ax2, df_smile_specific2, smile2, energy_key1, energy_key2, lims
    )
    
    make_subplot(
        ax3, df3, smile2, energy_key1, energy_key2, lims
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
    dfCO = pd.read_pickle("data/gemnet_relaxations_for_parity.pkl")
    dfOH = pd.read_pickle("data/gemnet-is2re-finetuned-11-01.pkl")
    dfOH_rel = pd.read_pickle("data/OH_gemnet-dT_val_relaxations.pkl")
    

    ### Apply filters
    df_filtered_OH = apply_filters({"filter_by_element_groups":['transition metal'], 'filter_by_num_elements': [2]}, dfOH)
    df_CO = apply_filters({"filter_by_element_groups":['transition metal'], 'filter_by_num_elements': [2]}, dfCO)

    ### Generate plot:
    get_paper_parity_plot('*CO', '*OH', df_CO, df_filtered_OH, dfOH_rel, [-4,2], 'CO_OH_parity3_new.svg')