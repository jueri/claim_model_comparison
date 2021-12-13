# -*- coding: utf-8 -*-
"""This module contains functions related to the dataset on which 
the models are trained and tested.

Example:
    The dataset can simply be loaded with a singe function:
        $ from dataset import load_dataset
        $
        $ load_dataset()

"""

import os
import pandas as pd

from config import DATA_PATH

def load_dataset() -> pd.DataFrame:
    """Function to load the dataset.

    Returns:
        pd.DatFrame: Dataframe with all data.
    """
    data = pd.read_csv(os.path.join(DATA_PATH))  # load Data

    claims = data[data["Claim"] == True]
    no_claims = data[data["Claim"] == False].sample(n=len(claims), random_state=42)
    data_sample = pd.concat([claims, no_claims])
    return data_sample

