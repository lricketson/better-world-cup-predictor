import pandas as pd
from blueprint import MatrixPipeline, EloModifier
from constants import TEAM_ID_MAP, BEST_ALPHA, BEST_BETA, WC_FINAL_LIVE_URL
from helpers import country_to_elo, safe_scrape_elo
from bayesian_decay import BayesianDecayEngine
from live_scraper import LiveEventScraper
from live_feed_handler import LiveFeedHandler
from knn_indexer import TacticalKNNIndexer
from vectoriser import TacticalVectoriser
from live_simulator import run_live_pytorch_monte_carlo
import asyncio
import glob
import torch
import csv
import os
import time

home_team = "Spain"
away_team = "Argentina"

elo_df = safe_scrape_elo()

elo_home = country_to_elo(elo_df, home_team)
elo_away = country_to_elo(elo_df, away_team)

prior_ctx = {
    "home_team": home_team,
    "home_id": TEAM_ID_MAP[home_team],
    "away_team": away_team,
    "away_id": TEAM_ID_MAP[away_team],
    "elo_home": elo_home,
    "elo_away": elo_away,
    "alpha": BEST_ALPHA,
    "beta": BEST_BETA,
}


def execute_live_pipeline(
    scraper: LiveEventScraper,
    vectoriser: TacticalVectoriser,
    indexer: TacticalKNNIndexer,
    decay_engine: BayesianDecayEngine,
):
    """
    Fires on live event updates: extracts state -> k-NN -> Bayesian blend -> Monte Carlo -> CSV Persist.
    """
    payload = scraper.export_engine_payload()
    current_clock = payload["clock_seconds"]
    current_minute = int(current_clock // 60)
    current_state_idx = payload["active_ball_state_idx"]
    home_goals = payload["scoreboard"][0]
    away_goals = payload["scoreboard"][1]

    # Don't run simulations during pre-game dead time
    if current_clock <= 0:
        return

    normalised_vec = vectoriser.vectorise(payload)

    lambda_knn, distances, indices = indexer.get_pseudo_prior(
        normalised_vec, current_clock
    )

    n_live = payload["n_live"]
    T_live = payload["T_live"]

    lambda_live = scraper.get_live_transition_rates()

    q_live_blended = decay_engine.blend(
        lambda_live=lambda_live,
        T_live=T_live,
        lambda_knn=lambda_knn,
        clock_seconds=current_clock,
    )

    prob_h, prob_d, prob_a = run_live_pytorch_monte_carlo(
        q_matrix=q_live_blended,
        current_clock=current_clock,
        current_state_idx=current_state_idx,
        live_home_goals=home_goals,
        live_away_goals=away_goals,
        num_simulations=10000,
        verbose=False,
    )

    # 6. Print Live Dashboard to Terminal
    print("\n" + "=" * 50)
    print(f" [LIVE ENGINE] Minute: {current_minute}' | Clock: {current_clock:.0f}s")
    print(f" Scoreboard: {home_team} {home_goals} - {away_goals} {away_team}")
    print("--------------------------------------------------")
    print(f" {home_team} Win:  {(prob_h * 100):.2f}%")
    print(f" Draw (90m):     {(prob_d * 100):.2f}%")
    print(f" {away_team} Win:  {(prob_a * 100):.2f}%")
    print("=" * 50)

    # =========================================================================
    # 7. PERSIST CONTINUOUS-TIME TRAJECTORY TO CSV
    # =========================================================================
    csv_path = "./data/live_win_probabilities.csv"
    file_exists = os.path.exists(csv_path)

    # Ensure the ./data/ directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    wall_clock_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header row only if the file is being created for the first time
            if not file_exists:
                writer.writerow(
                    [
                        "wall_clock_time",
                        "match_clock_seconds",
                        "match_minute",
                        "home_team",
                        "home_goals",
                        "away_team",
                        "away_goals",
                        "prob_home_win",
                        "prob_draw",
                        "prob_away_win",
                    ]
                )

            # Append current live simulation state
            writer.writerow(
                [
                    wall_clock_timestamp,
                    f"{current_clock:.1f}",
                    current_minute,
                    home_team,
                    home_goals,
                    away_team,
                    away_goals,
                    f"{prob_h:.6f}",
                    f"{prob_d:.6f}",
                    f"{prob_a:.6f}",
                ]
            )
    except Exception as e:
        print(f"[-] Warning: Failed to write to CSV log: {e}")


async def main_async():
    print("[*] Initializing Quantitative Engine for World Cup Final...")

    global_q_matrix = pd.read_csv("./data/global_q_matrix.csv")
    baseline_pipeline = MatrixPipeline([EloModifier()])
    final_q_grid = baseline_pipeline.build_grid(global_q_matrix, prior_ctx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_tensor = torch.tensor(final_q_grid.values, dtype=torch.float32, device=device)
    bayesian_decay_eng = BayesianDecayEngine(historical_baseline=prior_tensor)

    slice_files = glob.glob("./compiled_db/*.pt")

    if not slice_files:
        print("[-] WARNING: No compiled database slices found in ./compiled_db/!")
        # fall back to default 15 neighbours
        indexer = TacticalKNNIndexer(k_neighbours=15)
    else:
        sample_data = torch.load(slice_files[0], map_location=device)
        M_matches = sample_data["vectors_normalised"].shape[0]
        dynamic_k = max(5, int(M_matches**0.5))
        indexer = TacticalKNNIndexer(k_neighbours=dynamic_k)
        print(
            f"[+] Database size M={M_matches}. Dynamically set k_neighbours={dynamic_k}"
        )
        for filepath in slice_files:
            slice_data = torch.load(filepath, map_location=device)
            minute_key = slice_data["minute"]
            indexer.register_historical_slice(
                minute_timestamp=minute_key,
                vectors_matrix=slice_data["vectors_normalised"],  # the M 5D vectors
                n_future_matrix=slice_data["n_future"],  # M matrices of shape (12, 12)
                T_future_matrix=slice_data["T_future"],  # M vectors of shape (12,)
            )
        print(
            f"[+] Loaded {len(slice_files)} offline database slices into K-NN registry."
        )

    # instantiate vectoriser and LiveScraper
    # pass mu and sigma from the first db slice to ensure standardised z-scores
    sample_slice = (
        torch.load(slice_files[0], map_location=device) if slice_files else {}
    )

    vectoriser = TacticalVectoriser(
        historical_means=sample_slice.get("mu", None),
        historical_stds=sample_slice.get("sigma", None),
    )

    scraper = LiveEventScraper(
        home_team_id=prior_ctx["home_id"], away_team_id=prior_ctx["away_id"]
    )

    handler = LiveFeedHandler(
        scraper=scraper,
        websocket_url=WC_FINAL_LIVE_URL,
        fallback_json_path="./live_match_feed.json",
    )

    print("[+] Engine Online! Waiting for live kickoff events...")

    while True:
        # Check local fallback file for new match events every second
        await handler.poll_fallback_file(poll_interval=1.0)

        if scraper.current_clock > 0:
            execute_live_pipeline(
                scraper=scraper,
                vectoriser=vectoriser,
                indexer=indexer,
                decay_engine=bayesian_decay_eng,
            )
        await asyncio.sleep(1.0)


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[*] Quantitative engine shut down cleanly by operator.")


if __name__ == "__main__":
    main()
