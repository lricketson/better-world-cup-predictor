from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import time

STATES = [
    "Z:0_P:H",
    "Z:1_P:H",
    "Z:2_P:H",
    "Z:3_P:H",
    "Z:4_P:H",
    "Z:0_P:A",
    "Z:1_P:A",
    "Z:2_P:A",
    "Z:3_P:A",
    "Z:4_P:A",
    "Goal_H",
    "Goal_A",
]

STATE_TO_IDX = {state: i for i, state in enumerate(STATES)}


class LiveEventScraper:
    """
    Ingests live match event streams, maintains real-time CTMC state ledgers in RAM, and exports clean
    snapshots for the Tri-Modal Bayesian Decay engine.
    """

    def __init__(self, home_team_id: int, away_team_id: int):
        self.home_id = home_team_id
        self.away_id = away_team_id

        # real-time state trackers
        self.current_clock: float = 0.0  # elapsed match time in seconds
        self.last_event_time: float = 0.0  # timestamp of the previous event
        self.scoreboard: list[int] = [0, 0]  # [home_goals, away_goals]

        # initialise ball at home kickoff  (Z:2_P:H, which is Index 2)
        self.current_state_idx: int = STATE_TO_IDX["Z:2_P:H"]
        self.current_state_str: str = "Z:2_P:H"

        # continuous-time maths ledgers (in RAM)
        # n_live is a 12x12 matrix tracking exact transition counts from state i to j
        self.n_live = np.zeros((12, 12), dtype=np.float32)

        # T_live is a 12-element vector tracking total cumulative seconds spent in state i
        self.T_live = np.zeros(12, dtype=np.float32)
