from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from constants import BEST_ALPHA, BEST_BETA
from util import (
    standardise_possessions,
    align_team_perspective,
    calculate_global_q,
    calculate_specific_q,
    create_full_team_df,
)


class FeatureStrategy(ABC):
    """
    The contract that all matrix feature modifiers must follow.
    """

    @abstractmethod
    def apply(self, matrix: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        pass


class EloModifier(FeatureStrategy):
    """
    Applies exponential scaling to on-ball actions based on Elo differentials.
    """

    def apply(self, matrix: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        df = matrix.copy()

        elo_home = ctx["elo_home"]
        elo_away = ctx["elo_away"]
        beta = ctx.get("beta", BEST_BETA)

        df["start_zone"] = df["starting_state"].str[2]
        df["start_poss"] = df["starting_state"].str[-1]
        df["finish_poss"] = df["finishing_state"].str[-1]

        is_goal = df["finishing_state"].str.startswith("Goal")
        df["finish_zone"] = df["finishing_state"].str[2]

        # if home team has the ball, the elo diff is elo_home - elo_away, and vice versa
        active_diff = np.where(
            df["start_poss"] == "H", elo_home - elo_away, elo_away - elo_home
        )

        is_progression = (
            (~is_goal)
            & (df["finish_zone"] > df["start_zone"])
            & (df["start_poss"] == df["finish_poss"])
        )
        is_scoring = (is_goal) & (df["start_poss"] == df["finish_poss"])

        is_positive_action = is_progression | is_scoring

        modifier = np.where(is_positive_action, active_diff, -active_diff)
        df["updated_lambda_ij"] = df["updated_lambda_ij"] * np.exp(beta * modifier)

        return df


class MatrixPipeline:
    """
    Orchestrates the Bayesian prior and passes it through an assembly line of modular strategies.
    """

    def __init__(self, strategies: list[FeatureStrategy]):
        self.strategies = strategies

    def build_grid(self, global_q: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        # pull core IDs from context dict
        home_team, home_id = ctx["home_team"], ctx["home_id"]
        away_team, away_id = ctx["away_team"], ctx["away_id"]
        alpha = ctx.get("alpha", BEST_ALPHA)

        # fetch raw historical data and standardise
        full_home_df = standardise_possessions(create_full_team_df(home_team))
        full_away_df = standardise_possessions(create_full_team_df(away_team))

        # align home/away perspectives
        aligned_home_df = align_team_perspective(full_home_df, home_id, sim_role="H")
        aligned_away_df = align_team_perspective(full_away_df, away_id, sim_role="A")

        # aggregate transition counts
        home_counts, _ = calculate_global_q(aligned_home_df)
        away_counts, _ = calculate_global_q(aligned_away_df)

        # Bayesian conjugate updating
        home_q_matrix, _ = calculate_specific_q(global_q, ctx["alpha"], home_counts)
        away_q_matrix, _ = calculate_specific_q(global_q, ctx["alpha"], away_counts)

        # get attacking rows
        home_attacking_rows = home_q_matrix[
            home_q_matrix["starting_state"].str.endswith("H")
        ]
        away_attacking_rows = away_q_matrix[
            away_q_matrix["starting_state"].str.endswith("A")
        ]

        combined_matrix = pd.concat([home_attacking_rows, away_attacking_rows])

        # Assembly line:
        for strategy in self.strategies:
            combined_matrix = strategy.apply(combined_matrix, ctx)

        final_q_grid = combined_matrix.pivot(
            index="starting_state",
            columns="finishing_state",
            values="updated_lambda_ij",
        ).fillna(0)

        return final_q_grid
