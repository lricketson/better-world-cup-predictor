import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from util import parse_match_to_dataframe
from constants import STATE_TO_IDX, HOME_ATTACK_IDX, AWAY_ATTACK_IDX


def extract_features_at_minute(
    df_past: pd.DataFrame, elapsed_seconds: float, epsilon: float = 1e-6
) -> List[float]:
    """"""
