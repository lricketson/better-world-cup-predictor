import torch
from typing import Dict, Any, List

HOME_STATES_IDX = [0, 1, 2, 3, 4]
AWAY_STATES_IDX = [5, 6, 7, 8, 9]
HOME_ATTACK_IDX = [3, 4]  # Z:3_P:H and Z:4_P:H
AWAY_ATTACK_IDX = [8, 9]  # Z:3_P:A and Z:4_P:A


class TacticalVectoriser:
    """
    Transforms raw live CTMC ledgers from LiveEventScraper into normalised 4D vectors containing
    aggregated match statistics (Field Tilt, Progression Ratio, Match Tempo, Goal Diff) ready
    for native PyTorch Euclidean distance calculations (torch.cdist).
    """

    def __init__(
        self, historical_means: List[float] = None, historical_stds: List[float] = None
    ):
        """
        Initialises standardisation parameters. In the final version, these will be calculated from the
        historical match database.
        """

        self.use_pinned = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_pinned else "cpu")

        # base statistics (field tilt, progression ratio, match tempo, goal diff)
        default_means = [0.50, 1.00, 35.0, 0.0]
        default_stds = [0.15, 0.40, 12.0, 1.2]

        self.mu = torch.tensor(
            historical_means or default_means,
            dtype=torch.float32,
            device=self.device,
            pin_memory=self.use_pinned if self.device == "cpu" else False,
        )

        self.sigma = torch.tensor(
            historical_stds or default_stds,
            dtype=torch.float32,
            device=self.device,
            pin_memory=self.use_pinned if self.device == "cpu" else False,
        )

    def vectorise(self, payload: Dict[str, Any], epsilon: float = 1e-6) -> torch.Tensor:
        """
        Ingests the dictionary exported by LiveEventScraper.export_engine_payload()
        and outputs a normalised 1D PyTorch tensor of shape (4,).
        """

        n_live = payload["n_live"].to(self.device, non_blocking=True)
        T_live = payload["T_live"].to(self.device, non_blocking=True)
        scoreboard = payload["scoreboard"].to(self.device, non_blocking=True)

        # --- V0: Field Tilt ---

        home_attack_touches = n_live[HOME_ATTACK_IDX, :].sum()
        away_attack_touches = n_live[AWAY_ATTACK_IDX, :].sum()
        total_attack_touches = home_attack_touches + away_attack_touches

        if total_attack_touches > 0:
            v0_tilt = home_attack_touches / (total_attack_touches + epsilon)
        else:
            # for kickoff or early minutes with zero deep touches, default to 50-50 split
            v0_tilt = torch.tensor(0.5, device=self.device)

        # --- V1: Progression Ratio ---
        p_prog_h = self._count_progressions(n_live, HOME_STATES_IDX)
        p_prog_a = self._count_progressions(n_live, AWAY_STATES_IDX)

        v1_prog_ratio = (p_prog_h + 1.0) / (p_prog_a + 1.0)

        # --- V2: Match Tempo ---

        # isolate non-terminal starting states (indices 0-9)
        active_n = n_live[:10, :]
        active_T = T_live[:10].unsqueeze(1) + epsilon

        transition_rates = active_n / active_T
        v2_tempo = transition_rates.sum()

        # --- V3: Goal Diff ---
        v3_score_diff = (scoreboard[0] - scoreboard[1]).float()

        raw_vector = torch.stack([v0_tilt, v1_prog_ratio, v2_tempo, v3_score_diff])
        normalised_vector = (raw_vector - self.mu) / (self.sigma + epsilon)

        return normalised_vector

    def _count_progressions(
        self, n_matrix: torch.Tensor, team_indices: List[int]
    ) -> torch.Tensor:
        """
        A helper function to sum up the number of transitions where finishing zone index > starting
        zone index, and possession is kept.
        """
        progressions = torch.tensor(0, device=self.device)
        for i in range(len(team_indices)):
            start_idx = team_indices[i]
            for j in range(i + 1, len(team_indices)):
                finish_idx = team_indices[j]
                progressions += n_matrix[start_idx, finish_idx]
        return progressions
