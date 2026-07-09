import numpy as np
import pandas as pd
from tqdm import tqdm


def prep_simulation_parameters(q_grid: pd.DataFrame):
    """Converts the raw rate matrix into fast-lookup simulation parameters."""
    states = q_grid.columns.tolist()
    exit_rates = {}
    transition_probs = {}

    for state in states:
        # if it's a Goal state, we handle separately
        if "Goal" in state:
            continue

        row = q_grid.loc[state]
        total_rate = row.sum()

        # prevent division by 0
        if total_rate == 0:
            exit_rates[state] = 0.0001
            transition_probs[state] = np.ones(len(states)) / len(states)
        else:
            exit_rates[state] = total_rate
            transition_probs[state] = (row / total_rate).values

    return states, exit_rates, transition_probs


def simulate_match(states, exit_rates, transition_probs, match_seconds=5400):
    """Simulates exactly one 90-minute match."""

    current_state = "Z:2_P:H"  # home kickoff to start
    time_elapsed = 0.0
    home_goals = 0
    away_goals = 0

    # the Gillespie loop
    while time_elapsed < match_seconds:
        # manual override for Goal states
        if current_state == "Goal_H":
            home_goals += 1
            current_state = "Z:2_P:A"  # away kickoff
            time_elapsed += 30.0  # 30 sec of dead time for celebration
            continue
        elif current_state == "Goal_A":
            away_goals += 1
            current_state = "Z:2_P:H"
            time_elapsed += 30.0
            continue
        # how long does the ball stay in each state?
        # model using the exponential distribution
        rate_i = exit_rates[current_state]
        dt = np.random.exponential(
            scale=1.0 / rate_i
        )  # if exit rate is bigger, then dt is going to be smaller
        time_elapsed += dt

        if time_elapsed >= match_seconds:
            break

        # where does the ball go?
        # pick the next state by looking at the transition probabilities of the current row
        probs = transition_probs[current_state]
        next_state = np.random.choice(states, p=probs)
        current_state = next_state
    return home_goals, away_goals


def run_monte_carlo(q_grid: pd.DataFrame, num_simulations=10000):
    """Runs N alternate realities and calculates probabilities based on empirical results."""

    print(f"\n[*] Prepping matrix for {num_simulations} Monte Carlo simulations...")
    states, exit_rates, transition_probs = prep_simulation_parameters(q_grid)

    results = {"Home": 0, "Draw": 0, "Away": 0}
    home_goals_total = []
    away_goals_total = []

    print("[*] Running Engine...")
    for _ in tqdm(range(num_simulations), desc="Simulating Matches", unit="match"):
        h_goals, a_goals = simulate_match(states, exit_rates, transition_probs)

        home_goals_total.append(h_goals)
        away_goals_total.append(a_goals)

        if h_goals > a_goals:
            results["Home"] += 1
        elif a_goals > h_goals:
            results["Away"] += 1
        else:
            results["Draw"] += 1

    # calculate probabilities
    prob_h = results["Home"] / num_simulations
    prob_a = results["Away"] / num_simulations
    prob_draw = results["Draw"] / num_simulations

    # xG for each team in this matchup
    sim_xg_home = np.mean(home_goals_total)
    sim_xg_away = np.mean(away_goals_total)

    print("\n==================================")
    print("      FINAL MATCH FORECAST        ")
    print("==================================")
    print(f"Home Win:  {prob_h:.2f}%")
    print(f"Draw:      {prob_draw:.2f}%")
    print(f"Away Win:  {prob_a:.2f}%")
    print("----------------------------------")
    print(f"Expected Score: Home {sim_xg_home:.2f} - {sim_xg_away:.2f} Away")
    print("==================================\n")

    return prob_h, prob_a, prob_draw
