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
        lambda row: calculate_state(row, is_start_coordinate=True)
    )
    # and the finishing states
    df["finishing_state"] = df.apply(
        lambda row: calculate_state(row, is_start_coordinate=False)
    )

    cols_to_keep = [
        "eventId",
        "teamId",
        "type",
        "outcomeType",
        "starting_state",
        "finishing_state",
    ]

    existing_cols = [c for c in cols_to_keep if c in df.columns]

    return df[existing_cols]
