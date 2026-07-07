import pandas as pd

processed_matches_list = []
for match in matches:
    processed_match = pd.read_json(match)
    processed_matches_list.append(processed_match)
