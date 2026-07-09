import torch
import pandas as pd
import numpy as np
import time


def run_pytorch_monte_carlo(
    q_grid: pd.DataFrame, num_simulations=10000, match_seconds=5400.0
):
    """
    A function for running vectorised Monte Carlo simulations.
    """
    print(f"\n[*] Prepping matrix for {num_simulations} parallel simulations...")

    # detect hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device.type.upper()}")

    # map strings to integers since GPUs only understand maths
    states = q_grid.columns.tolist()
    num_states = len(states)

    state_to_idx = {s: i for i, s in enumerate(states)}
    goal_h_idx = state_to_idx.get("Goal_H", -1)  # the index number representing goal_h
    goal_a_idx = state_to_idx.get("Goal_A", -1)
    kickoff_h_idx = state_to_idx.get("Z:2_P:H", 0)
    kickoff_a_idx = state_to_idx.get("Z:2_P:A", 0)

    # build tensors on the GPU
    # a 1D tensor of size 12 since there are 12 states. it stores the total exit rate out of each state.
    # it governs when the ball moves (since it's fed into the exponential distribution later).
    exit_rates = torch.zeros(num_states, device=device, dtype=torch.float32)
    # a 2D tensor (a 12x12 matrix equivalent). It stores the probabilities of the ball transitioning from
    # state i to state j, so it governs where the ball moves.
    transition_probs = torch.zeros(
        (num_states, num_states), device=device, dtype=torch.float32
    )

    for i, state in enumerate(states):
        if "Goal" in state:
            exit_rates[state] = 1.0
            transition_probs[i, i] = 1.0
            continue
        row = q_grid.loc[state].values
        total_rate = row.sum()

        if total_rate == 0:
            exit_rates[i] = 0.0001
            transition_probs[i] = 1.0 / num_states
        else:
            exit_rates[i] = total_rate
            # turns raw transition rates into true probability distributions (that sum to 1)
            transition_probs[i] = torch.tensor(
                row / total_rate, device=device, dtype=torch.float32
            )

    # initialise 10,000 matches simultaneously
    # current_states tracks the state of the ball in each of the 10k matches at once
    current_states = torch.full(
        (num_simulations,), kickoff_h_idx, device=device, dtype=torch.long
    )
    # tracks the match clocks independently for each of the 10k matches
    times = torch.zeros(num_simulations, device=device, dtype=torch.float32)
    # tensors acting as scoreboards for each of the 10k matches
    home_goals = torch.zeros(num_simulations, device=device, dtype=torch.long)
    away_goals = torch.zeros(num_simulations, device=device, dtype=torch.long)

    print("[*] Firing PyTorch Engine...")

    start_time = time.time()

    # a 10k-wide array of Boolean values, where True means the match is still going and False means
    # it's finished
    active_mask = times < match_seconds

    # THE VECTORISED LOOP
    while active_mask.any():
        # active_indices is the list of indices of matches that are still in play
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        # the list of current states of the matches that are still in play
        active_states = current_states[active_indices]

        # A. handle goals
        # checks every single match to see if any of their current states are goals
        is_goal_h = active_states == goal_h_idx
        is_goal_a = active_states == goal_a_idx

        if is_goal_h.any():
            # isolates the exact timelines where a goal was scored
            # idx_h is a boolean array with 1s where a goal was scored and 0s elsewhere
            idx_h = active_indices[is_goal_h]
            # increments the scoreboards just for those matches where a (home) goal was scored
            home_goals[idx_h] += 1
            # resets current state back to away team's kickoff
            current_states[idx_h] = kickoff_a_idx
            # adds 30 sec dead time for celebration
            times[idx_h] += 30.0

        if is_goal_a.any():
            idx_a = active_indices[is_goal_a]
            away_goals[idx_a] += 1
            current_states[idx_a] = kickoff_h_idx
            times[idx_a] += 30.0

        # refresh states in case kickoffs happened
        active_states = current_states[active_indices]

        # --- B. Time jumps (vectorised exponential distribution) ---
        rates = exit_rates[active_states]
        u = torch.rand(len(active_indices), device=device)
        dt = -torch.log(u) / rates
        times[active_indices] += dt

        # --- C. Next states ---
        # only move the ball if the time jump didn't push the time past 90 min
        valid_transition_mask = times[active_indices] < match_seconds
        valid_indices = active_indices[valid_transition_mask]
        valid_states = current_states[valid_indices]

        if len(valid_indices) > 0:
            probs = transition_probs[valid_states]
            next_states = torch.multinomial(probs, 1).squeeze(1)
            current_states[valid_indices] = next_states

        # update the master tracking mask
        active_mask = times < match_seconds

    elapsed = time.time() - start_time
    h_goals = home_goals.cpu().numpy()
    a_goals = away_goals.cpu().numpy()

    home_wins = np.sum(h_goals > a_goals)
    away_wins = np.sum(a_goals > h_goals)
    draws = np.sum(h_goals == a_goals)

    prob_h = home_wins / num_simulations
    prob_a = away_wins / num_simulations
    prob_d = draws / num_simulations

    sim_xg_home = np.mean(h_goals)
    sim_xg_away = np.mean(a_goals)

    print("\n==================================")
    print("      FINAL MATCH FORECAST        ")
    print("==================================")
    print(f"Hardware Time: {elapsed:.3f} seconds")
    print(f"Home Win:  {(prob_h * 100):.2f}%")
    print(f"Draw:      {(prob_d * 100):.2f}%")
    print(f"Away Win:  {(prob_a * 100):.2f}%")
    print("----------------------------------")
    print(f"Expected Score: Home {sim_xg_home:.2f} - {sim_xg_away:.2f} Away")
    print("==================================\n")

    return prob_h, prob_d, prob_a
