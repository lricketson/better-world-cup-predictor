from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import time
import torch
from helpers import (
    apply_stoppage_cap,
    get_spatial_zone,
    resolve_goal_state,
    resolve_possession,
)

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

        self.scoreboard = torch.zeros(2, dtype=torch.long, pin_memory=True)

        # initialise ball at home kickoff  (Z:2_P:H, which is Index 2)
        self.current_state_idx: int = STATE_TO_IDX["Z:2_P:H"]
        self.current_state_str: str = "Z:2_P:H"

        # continuous-time maths ledgers (in RAM)
        # n_live is a 12x12 matrix tracking exact transition counts from state i to j
        self.n_live = torch.zeros((12, 12), dtype=torch.float32, pin_memory=True)

        # T_live is a 12-element vector tracking total cumulative seconds spent in state i
        self.T_live = torch.zeros(12, dtype=torch.float32, pin_memory=True)

    def process_event(self, event_packet: Dict[str, any]):
        # skip non-touch events like cards
        if not event_packet.get("isTouch", False):
            return
        event_time = event_packet["expandedMinute"] * 60 + event_packet["second"]
        raw_delta = event_time - self.last_event_time
        delta_t = apply_stoppage_cap(raw_delta)

        is_goal = event_packet.get("isGoal", False) is True
        is_own = event_packet.get("isOwnGoal", False) is True
        is_home = event_packet["teamId"] == self.home_id

        goal_state = resolve_goal_state(is_goal, is_own, is_home)

        if goal_state:
            finishing_goal_state = goal_state
            # update pinned scoreboard tensor directly
            if goal_state == "Goal_H":
                self.scoreboard[0] += 1
            else:
                self.scoreboard[1] += 1

        else:
            # not a goal
            end_x = event_packet.get("endX", event_packet["x"])
            zone = get_spatial_zone(end_x)
            outcome_val = event_packet.get("outcomeType", {}).get("value", 1)
            possession = resolve_possession(
                event_packet["teamId"], self.home_id, outcome_val
            )

            finishing_state_str = f"Z:{zone}_P:{possession}"

        next_state_idx = STATE_TO_IDX[finishing_state_str]

        self.n_live[self.current_state_idx, next_state_idx] += 1.0
        self.T_live[self.current_state_idx] += delta_t

        self.current_clock = event_time
        self.last_event_time = event_time
        self.current_state_idx = next_state_idx
        self.current_state_str = finishing_state_str
