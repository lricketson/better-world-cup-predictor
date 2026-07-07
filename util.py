import pandas as pd
import json
import os
import glob


def parse_match_to_dataframe(filepath):

    # open file and extract json
    with open(filepath, mode="r", encoding="utf-8") as f:
        match_data = json.load(f)

    # get ids of home and away teams
    home_id = match_data["home"]["teamId"]
    away_id = match_data["away"]["teamId"]

    # isolate the events array and load only that into Pandas
    events_list = match_data["events"]
    df = pd.DataFrame(events_list)

    # only take the rows where the ball was actually touched
    # this prevents taking events like red cards (which we'll come back to later, but this is a ledger of ball movement)
    if "isTouch" in df.columns:
        df = df[df["isTouch"] == True].copy()

    df["match_seconds"] = df["expandedMinute"] * 60 + df["second"]
    df = df.set_index("match_seconds")
    df = df.sort_index()

    def calculate_state(row, is_start_coordinate=True):
        possession = "Home" if row["teamId"] == home_id else "Away"

        # if we're looking at event end coordinates:
        if not is_start_coordinate:
            # Opta stores event outcomeTypes as a dict: {'value': 1, 'displayName': 'Successful'}
            # Value == 0 means unsuccessful. So if the event was unsuccessful, it means they lost
            # possession, so we give possession to the other team.
            outcome_val = row.get("outcomeType", {}).get("value")  #
            if outcome_val == 0:
                possession = "Away" if possession == "Home" else "Home"

        # determine which coordinate to use for the event
        x_val = row["x"] if is_start_coordinate else row.get("endX", row["x"])

        # cap possible edge cases
        x_val = max(0, min(x_val, 100))

        zone = ""
        # to capture the penalty boxes
        if x_val < 17:
            zone = "0"
        elif x_val < 39:
            zone = "1"
        elif x_val < 61:
            zone = "2"
        elif x_val < 83:
            zone = "3"
        else:
            zone = "4"

        state = f"Z:{zone}_P:{possession}"
        return state

    # get the events' starting states
    df["starting_state"] = df.apply(
        lambda row: calculate_state(row, is_start_coordinate=True), axis=1
    )
    # and the finishing states
    df["finishing_state"] = df.apply(
        lambda row: calculate_state(row, is_start_coordinate=False), axis=1
    )

    # fill NaN values
    if "isGoal" in df.columns:
        is_goal = df["isGoal"] == True
    else:
        is_goal = pd.Series(False, index=df.index)

    # this produces a series of every event telling us whether it was an event done by the home team or not
    is_home = df["teamId"] == home_id

    # if the event was a goal and the event was done by the home team, it's a home goal
    df.loc[is_goal & is_home, "finishing_state"] = "Goal_H"
    df.loc[is_goal & ~is_home, "finishing_state"] = "Goal_A"

    # find time spent in the starting state (the difference between this event and the next)
    df["time_spent_seconds"] = df.index.to_series().diff().shift(-1)

    # fill the very last event of the match (which has no next event) with the standard 2 seconds
    df["time_spent_seconds"] = df["time_spent_seconds"].fillna(2.0)

    # cap stoppages at 15 seconds to preserve tactical reality
    df.loc[df["time_spent_seconds"] > 15.0, "time_spent_seconds"] = 3.0

    cols_to_keep = [
        "eventId",
        "teamId",
        "type",
        "outcomeType",
        "starting_state",
        "finishing_state",
        "time_spent_seconds",
    ]

    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols].copy()

    # change to snake_case from JS naming convention
    df = df.rename(
        columns={
            "eventId": "event_id",
            "teamId": "team_id",
            "outcomeType": "outcome_type",
        }
    )

    return df


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


def safe_parse(filepath):
    """Wrapper to catch corrupted files without crashing the whole script."""
    try:
        return parse_match_to_dataframe(filepath)
    except Exception as e:
        print(f"  [-] Failed to parse {os.path.basename(filepath)}: {str(e)}")
        return pd.DataFrame()  # Return empty df on failure


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


def scrape_elo_ratings():
    url = "https://www.eloratings.net/World.tsv"
    elo_df = pd.read_csv(url, sep="\t", header=None)
    elo_df.columns = [
        "rank",
        "country_code",
        "rating",
        "matches_played",
        "wins",
        "draws",
        "losses",
        "goals_for",
        "goals_against",
        "points_change_1yr",
    ]
    return elo_df


def incorporate_elo_diff(elo_diff, beta):
    pass


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
