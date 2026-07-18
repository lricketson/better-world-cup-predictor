import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from util import parse_match_to_dataframe
from constants import STATE_TO_IDX, HOME_ATTACK_IDX, AWAY_ATTACK_IDX

HOME_STATES_IDX = [0, 1, 2, 3, 4]
AWAY_STATES_IDX = [5, 6, 7, 8, 9]


def extract_features_at_minute(
    df_past: pd.DataFrame, elapsed_seconds: float, epsilon: float = 1e-6
) -> List[float]:
    """
    Calculates the raw 5D feature vector (V0, V1, V2, V3, V4) from kickoff up to time t.
    """
    if df_past.empty:
        return [0.5, 0.0, 0.0, 0.0, 0.0]

    # build historical n_past and T_past ledgers
    n_past = np.zeros((12, 12), dtype=np.float32)
    T_past = np.zeros(12, dtype=np.float32)

    for _, row in df_past.iterrows():
        start_idx = STATE_TO_IDX.get(row.get("starting_state"))
        finish_idx = STATE_TO_IDX.get(row.get("finishing_state"))
        t_spent = row.get("time_spent_seconds", 0.0)

        if start_idx is not None:
            T_past[start_idx] += float(t_spent)
            if finish_idx is not None:
                n_past[start_idx, finish_idx] += 1.0

    # V0: Field Tilt (Home share of attacking third touches)
    home_att = n_past[HOME_ATTACK_IDX, :].sum()
    away_att = n_past[AWAY_ATTACK_IDX, :].sum()

    total_att = home_att + away_att
    v0_tilt = float(home_att / total_att) if total_att > 0 else 0.50

    # V1 and V2: progression ratio (Home forward zone transitions / Total Home transitions)
    def calculate_progression_ratio(team_indices: List[int]) -> float:
        progressions = 0.0
        total_transitions = 0.0
        for i in range(len(team_indices)):
            s_idx = team_indices[i]
            total_transitions += n_past[s_idx, :].sum()
            for j in range(i + 1, len(team_indices)):
                f_idx = team_indices[j]
                progressions += n_past[s_idx, f_idx]
        return float(progressions / total_transitions) if total_transitions > 0 else 0.0

    v1_h_prog = calculate_progression_ratio(HOME_STATES_IDX)
    v2_a_prog = calculate_progression_ratio(AWAY_STATES_IDX)

    # V3: Markovian tempo (total touches per minute)
    # sum of departure rates across all the zones 0-9
    active_departures = n_past[:10, :].sum(axis=1)
    active_T = T_past[:10] + epsilon
    transition_rates = active_departures / active_T
    v3_tempo = float(np.sum(transition_rates))

    # V4: goal differential
    # n_past is the 12x12 matrix of transition counts, so [:, 10] gets the counts of an event that started
    # somewhere and ended in zone 10 (home goal). and then we sum up all those counts
    home_goals = n_past[:, 10].sum()
    away_goals = n_past[:, 11].sum()
    v4_score_diff = float(home_goals - away_goals)

    return [v0_tilt, v1_h_prog, v2_a_prog, v3_tempo, v4_score_diff]


def build_future_ledgers(df_future: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregates future transition counts (n) and holding times (T) from minute t to full time.
    Returns: n_matrix shape (12, 12) and T_vector shape (12,)
    """
