import pandas as pd
import json
import os
import glob
import numpy as np
from helpers import (
    parse_match_to_dataframe,
    safe_parse,
    align_team_perspective,
    standardise_possessions,
)


def create_master_df(folder_path="./data/world_cup_2026"):
    print(f"[*] Scanning '{folder_path}' for match data...")

    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    if not json_files:
        print("[-] No JSON files found. Check your directory path.")
        return pd.DataFrame()

    print(f"[*] Found {len(json_files)} match files. Beginning parsing...")

    df_list = []
    failed_files = []

    for i, file_path in enumerate(json_files, 1):
        try:
            match_df = parse_match_to_dataframe(file_path)
            if not match_df.empty:
                df_list.append(match_df)
            print(
                f"  [+] ({i}/{len(json_files)}) Parsed: {os.path.basename(file_path)}"
            )
        except Exception as e:
            # if a file is formatted weirdly or corrupted, log it and continue
            print(
                f"  [-] ({i}/{len(json_files)}) Failed to parse {os.path.basename(file_path)}: {str(e)}"
            )
            failed_files.append(file_path)
    if not df_list:
        print("[-] Critical Error: All files failed to parse.")
        return pd.DataFrame()

    print("[*] Concatenating all parsed matches into Master DataFrame...")

    master_df = pd.concat(df_list)

    print(f"[+] Master DataFrame built successfully! Total Events: {len(master_df)}")

    if failed_files:
        print(f"[!] Warning: {len(failed_files)} files were skipped due to errors.")

    return master_df


def calculate_global_q(master_df: pd.DataFrame):
    """
    Takes a concatenated DataFrame of all matches and returns a global generator matrix Q.
    """

    # calculate the numerator n_ij
    # find events with same starting and finishing states and count the occurrences
    transition_counts = (
        master_df.groupby(["starting_state", "finishing_state"])
        .size()
        .reset_index(name="n_ij")
    )

    # find total time spent in state i
    time_spent = (
        master_df.groupby("starting_state")["time_spent_seconds"]
        .sum()
        .reset_index(name="T_i")
    )

    q_matrix = pd.merge(transition_counts, time_spent, on="starting_state")

    q_matrix["lambda_ij"] = q_matrix["n_ij"] / q_matrix["T_i"]

    q_grid = q_matrix.pivot(
        index="starting_state", columns="finishing_state", values="lambda_ij"
    ).fillna(0)

    return q_matrix, q_grid


def create_full_team_df(team_name, folder_path="./data/world_cup_2026"):
    # switch from spaces to underscores for file names
    search_name = team_name.replace(" ", "_")

    # get all files
    all_files = glob.glob(os.path.join(folder_path, "*.json"))

    team_files = [f for f in all_files if search_name in os.path.basename(f)]

    if not team_files:
        print(f"[-] No matches found for {team_name} in {folder_path}.")
        return pd.DataFrame()
    print(f"[*] Found {len(team_files)} matches for {team_name}. Parsing...")

    # turn files to dfs
    dfs_list = [safe_parse(file) for file in team_files]
    valid_dfs = [df for df in dfs_list if not df.empty]

    if not valid_dfs:
        print(f"[-] All files for {team_name} failed to parse.")
        return pd.DataFrame()

    # concatenate
    merged_df = pd.concat(valid_dfs)
    print(
        f"[+] Successfully built dataframe for {team_name} ({len(merged_df)} events)."
    )
    return merged_df


def calculate_specific_q(
    global_q: pd.DataFrame, alpha: float, team_data_df: pd.DataFrame
):
    team_data_clean = team_data_df.rename(columns={"n_ij": "team_n", "T_i": "team_T"})

    merged = pd.merge(
        left=global_q,
        right=team_data_clean[
            [
                "starting_state",
                "finishing_state",
                "team_n",
                "team_T",
            ]
        ],
        on=["starting_state", "finishing_state"],
        how="left",
    )

    merged["team_n"] = merged["team_n"].fillna(0)
    merged["team_T"] = merged["team_T"].fillna(0)

    merged["updated_lambda_ij"] = (merged["n_ij"] * alpha + merged["team_n"]) / (
        merged["T_i"] * alpha + merged["team_T"]
    )
    updated_q_matrix = merged[
        ["starting_state", "finishing_state", "updated_lambda_ij"]
    ].copy()
    updated_q_grid = updated_q_matrix.pivot(
        index="starting_state", columns="finishing_state", values="updated_lambda_ij"
    ).fillna(0)

    return updated_q_matrix, updated_q_grid


def apply_elo_hazards(
    team_q_matrix: pd.DataFrame,
    elo_home: float,
    elo_away: float,
    beta: float = 0.0005,
):
    df = team_q_matrix.copy()

    # extract starting attributes
    df["start_zone"] = df["starting_state"].str[2]
    df["start_poss"] = df["starting_state"].str[-1]

    # extract finishing attributes
    df["finish_poss"] = df["finishing_state"].str[-1]

    # create a Boolean series of 'is this event a goal?'
    is_goal = df["finishing_state"].str.startswith("Goal")

    # df["finish_zone"] = df.loc[~is_goal, "finishing_state"].str[2]

    df["finish_zone"] = df["finishing_state"].str[2]

    active_diff = np.where(
        df["start_poss"] == "H", elo_home - elo_away, elo_away - elo_home
    )

    # define tactically positive actions
    is_progression = (
        (~is_goal)  # not a goal
        & (df["start_poss"] == df["finish_poss"])  # possession was kept
        & (df["finish_zone"] > df["start_zone"])  # ball was advanced forward
    )

    is_scoring = (is_goal) & (
        df["start_poss"] == df["finish_poss"]
    )  # it was a goal, and the team that started with possession is the one that scored

    is_positive_action = is_progression | is_scoring

    modifier = np.where(
        is_positive_action,
        active_diff,
        -active_diff,
    )

    df["match_lambda_ij"] = df["updated_lambda_ij"] * np.exp(beta * modifier)
    final_q_matrix = df[["starting_state", "finishing_state", "match_lambda_ij"]].copy()

    final_q_grid = final_q_matrix.pivot(
        index="starting_state", columns="finishing_state", values="match_lambda_ij"
    ).fillna(0)
    return final_q_matrix, final_q_grid


def create_final_matrix(
    home_team: str,
    home_id: int,
    away_team: str,
    away_id: int,
    elo_home: float,
    elo_away: float,
    global_q: pd.DataFrame,
    alpha: float,
):

    # 1. Fetch raw historical data (Raw Events)
    full_home_df = standardise_possessions(create_full_team_df(home_team))
    full_away_df = standardise_possessions(create_full_team_df(away_team))

    # 2. Align the perspectives (Raw Events)
    aligned_home_df = align_team_perspective(full_home_df, home_id, sim_role="H")
    aligned_away_df = align_team_perspective(full_away_df, away_id, sim_role="A")

    # 3. [THE FIX]: Aggregate the raw events into Transition Counts (n_ij) and Times (T_i)
    # We can reuse our global function to do this calculation for specific teams!
    home_counts, _ = calculate_global_q(aligned_home_df)
    away_counts, _ = calculate_global_q(aligned_away_df)

    # 4. Pass the AGGREGATED counts into the Bayesian update
    home_q_matrix, _ = calculate_specific_q(global_q, alpha, home_counts)
    away_q_matrix, _ = calculate_specific_q(global_q, alpha, away_counts)

    # 5. The Slicing
    home_attacking_rows = home_q_matrix[
        home_q_matrix["starting_state"].str.endswith("H")
    ]
    away_attacking_rows = away_q_matrix[
        away_q_matrix["starting_state"].str.endswith("A")
    ]

    combined_match_matrix = pd.concat([home_attacking_rows, away_attacking_rows])
    print("combined_match_matrix.head()", combined_match_matrix.head())

    # 6. Apply Elo Hazards
    final_q_matrix, final_q_grid = apply_elo_hazards(
        combined_match_matrix, elo_home, elo_away
    )
    print("final_q_matrix.head()", final_q_matrix.head())

    return final_q_grid


if __name__ == "__main__":
    # 1. Build the massive event ledger
    master_df = create_master_df("data/world_cup_2026")

    # 2. Only proceed to matrix calculation if data was successfully loaded
    if not master_df.empty:
        # 3. Calculate the dense Global Prior matrix
        rates_df, q_grid = calculate_global_q(master_df)

        print("\n[*] --- GLOBAL TRANSITION RATE MATRIX (Q) ---")
        print(q_grid.head())

        # save the matrix to CSV so i don't have to recalculate it every time
        q_grid.to_csv("global_transition_matrix.csv")
