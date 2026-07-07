import pandas as pd
import json


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
        is_goal = df["isGoal"].fillna(False).astype(bool)
    else:
        is_goal = pd.Series(False, index=df.index)

    # this produces a series of every event telling us whether it was an event done by the home team or not
    is_home = df["team_id"] == home_id

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
