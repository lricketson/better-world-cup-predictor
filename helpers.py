import pandas as pd
import numpy as np
import json
import os


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

    if "isOwnGoal" in df.columns:
        is_own_goal = df["isOwnGoal"] == True
    else:
        is_own_goal = pd.Series(False, index=df.index)

    # if the event was a goal and the event was done by the home team, it's a home goal
    df.loc[is_goal & is_home & ~is_own_goal, "finishing_state"] = "Goal_H"
    df.loc[is_goal & ~is_home & ~is_own_goal, "finishing_state"] = "Goal_A"

    # own goals logic
    df.loc[is_goal & is_home & is_own_goal, "finishing_state"] = "Goal_A"
    df.loc[is_goal & ~is_home & is_own_goal, "finishing_state"] = "Goal_H"

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


def safe_parse(filepath):
    """Wrapper to catch corrupted files without crashing the whole script."""
    try:
        return parse_match_to_dataframe(filepath)
    except Exception as e:
        print(f"  [-] Failed to parse {os.path.basename(filepath)}: {str(e)}")
        return pd.DataFrame()  # Return empty df on failure


def safe_scrape_elo():
    url = "https://www.eloratings.net/World.tsv"
    df = pd.read_csv(url, sep="\t", header=None)

    # THE FIX: Map to 2 (Code) and 3 (Rating)
    df.rename(columns={0: "rank", 2: "country_code", 3: "rating"}, inplace=True)

    return df


def country_to_country_code(country: str) -> str:
    """
    Returns the standard 3-letter FIFA country code for the 48 teams
    participating in the 2026 FIFA World Cup.
    """
    mapping = {
        # Group A
        "mexico": "MEX",
        "south africa": "RSA",
        "south korea": "KOR",
        "republic of korea": "KOR",
        "korea republic": "KOR",
        "czech republic": "CZE",
        "czechia": "CZE",
        # Group B
        "canada": "CAN",
        "bosnia and herzegovina": "BIH",
        "bosnia": "BIH",
        "qatar": "QAT",
        "switzerland": "SUI",
        # Group C
        "brazil": "BR",
        "morocco": "MAR",
        "haiti": "HAI",
        "scotland": "SCO",
        # Group D
        "united states": "USA",
        "usa": "USA",
        "united states of america": "USA",
        "paraguay": "PAR",
        "australia": "AUS",
        "turkey": "TUR",
        "türkiye": "TUR",
        # Group E
        "germany": "GER",
        "curaçao": "CUW",
        "curacao": "CUW",
        "ivory coast": "CIV",
        "côte d'ivoire": "CIV",
        "cote d'ivoire": "CIV",
        "ecuador": "ECU",
        # Group F
        "netherlands": "NED",
        "japan": "JPN",
        "sweden": "SWE",
        "tunisia": "TUN",
        # Group G
        "belgium": "BEL",
        "egypt": "EGY",
        "iran": "IRN",
        "ir iran": "IRN",
        "new zealand": "NZL",
        # Group H
        "spain": "ESP",
        "cape verde": "CPV",
        "cabo verde": "CPV",
        "saudi arabia": "KSA",
        "uruguay": "URU",
        # Group I
        "france": "FRA",
        "senegal": "SEN",
        "iraq": "IRQ",
        "norway": "NO",
        # Group J
        "argentina": "ARG",
        "algeria": "ALG",
        "austria": "AUT",
        "jordan": "JOR",
        # Group K
        "portugal": "POR",
        "dr congo": "COD",
        "congo dr": "COD",
        "democratic republic of congo": "COD",
        "democratic republic of the congo": "COD",
        "uzbekistan": "UZB",
        "colombia": "COL",
        # Group L
        "england": "ENG",
        "croatia": "CRO",
        "ghana": "GHA",
        "panama": "PAN",
    }

    # Clean up the input string and look up the dictionary
    formatted_country = country.strip().lower()
    return mapping.get(formatted_country, "Code not found")


def align_team_perspective(team_df: pd.DataFrame, team_id: int, sim_role: str):
    """
    Forces all of a team's historical actions to align with a certain chosen role (home or away) for
    the upcoming simulation.
    """

    # isolate just the actions of the team in question
    df = team_df[team_df["team_id"] == team_id].copy()

    # define the opponent's role
    opp_role = "A" if sim_role == "H" else "H"

    start_poss = df["starting_state"].str[-1]
    finish_poss = df["finishing_state"].str[-1]

    is_turnover = start_poss != finish_poss
    is_goal = df["finishing_state"].str.startswith("Goal")

    zone_start = df["starting_state"].str.slice(0, 3)
    df["starting_state"] = zone_start + f"_P:{sim_role}"

    zone_finish = df["finishing_state"].str.slice(0, 3)
    new_finish = np.where(
        is_turnover, zone_finish + f"_P:{opp_role}", zone_finish + f"_P:{sim_role}"
    )

    new_finish = np.where(is_goal & ~is_turnover, f"Goal_{sim_role}", new_finish)
    new_finish = np.where(is_goal & is_turnover, f"Goal_{opp_role}", new_finish)

    df["finishing_state"] = new_finish

    return df


def standardise_possessions(df):
    """Forces all 'Home' and 'Away' strings to 'H' and 'A' across the dataframe."""
    df_clean = df.copy()
    df_clean["starting_state"] = (
        df_clean["starting_state"].str.replace("Home", "H").str.replace("Away", "A")
    )
    df_clean["finishing_state"] = (
        df_clean["finishing_state"].str.replace("Home", "H").str.replace("Away", "A")
    )
    return df_clean


def probability_to_odds(probability: float, vig_margin: float = 0.0):
    """
    Converts a probability to its corresponding odds metric, optionally baking in
    a bookmaker's vig.
    """
    implied_probability = probability * (1 + vig_margin)
    odds = 1 / implied_probability
    return round(odds, 3)


def match_outcome_probs_to_odds(prob_home: float, prob_away: float, prob_draw: float):
    odds_home = probability_to_odds(prob_home)
    odds_away = probability_to_odds(prob_away)
    odds_draw = probability_to_odds(prob_draw)
    return odds_home, odds_away, odds_draw
