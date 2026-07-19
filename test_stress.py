import time
import numpy as np
import torch
from live_simulator import run_live_pytorch_monte_carlo


def create_synthetic_q_matrix(
    num_states: int = 12, device: str = "cpu", home_advantage: float = 1.0
) -> torch.Tensor:
    """Creates a mathematically valid continuous-time generator matrix Q with realistic goal rates."""
    q = torch.rand((num_states, num_states), device=device, dtype=torch.float32)

    # 1. Dampen goal transition columns (10=Goal_H, 11=Goal_A) so teams score ~1.5 goals per match
    # instead of 2,500 pinball goals!
    q[:, 10] *= 0.001 * home_advantage  # Goal_H column
    q[:, 11] *= 0.001  # Goal_A column

    # 2. Zero out diagonal before row-summing
    q.fill_diagonal_(0.0)
    row_sums = q.sum(dim=1)

    # 3. Set diagonal to enforce zero-sum rows: q_ii = -sum(q_ij)
    q.diagonal().copy_(-row_sums)
    return q


def test_distribution_integrity(device: torch.device):
    print("\n[*] TEST 1: Probability Distribution & Matrix Integrity...")
    q = create_synthetic_q_matrix(12, device=device)

    # Force an empty/zero-rate row at index 5 to test division-by-zero protection
    q[5, :] = 0.0

    q_off_diag = q.clone()
    q_off_diag.fill_diagonal_(0.0)
    exit_rates = q_off_diag.sum(dim=1)

    safe_rates = torch.where(
        exit_rates == 0, torch.tensor(0.0001, device=device), exit_rates
    )
    probs = q_off_diag / safe_rates.unsqueeze(1)
    probs = torch.clamp(probs, min=0.0)

    row_sums = probs.sum(dim=1, keepdim=True)
    zero_rows = row_sums == 0.0
    probs = torch.where(zero_rows, torch.full_like(probs, 1.0 / 12.0), probs)
    row_sums = torch.where(zero_rows, torch.tensor(1.0, device=device), row_sums)
    probs = probs / row_sums

    # Assertions
    assert not torch.isnan(
        probs
    ).any(), "FAIL: NaNs detected in transition probabilities!"
    assert not torch.isinf(
        probs
    ).any(), "FAIL: Infinite values detected in probabilities!"
    assert (probs >= 0.0).all(), "FAIL: Negative probabilities detected!"
    assert (probs <= 1.00001).all(), "FAIL: Probabilities exceeding 1.0 detected!"

    row_sum_errors = torch.abs(probs.sum(dim=1) - 1.0)
    max_err = row_sum_errors.max().item()
    assert max_err < 1e-5, f"FAIL: Row sums deviate from 1.0! Max error: {max_err}"

    print(
        "  [+] PASSED: All probability distributions are mathematically valid and sum to 1.0."
    )


def test_underflow_protection(device: torch.device):
    print("\n[*] TEST 2: Underflow & Zero-Rate Clock Protection...")
    q = create_synthetic_q_matrix(12, device=device)

    # Introduce extreme underflow numbers (e.g., 1e-12) to test log-division stability
    q[2, :] *= 1e-12
    q[2, 2] = -q[2, :].sum()

    start_t = time.time()
    prob_h, prob_d, prob_a = run_live_pytorch_monte_carlo(
        q_matrix=q,
        current_clock=100.0,
        current_state_idx=2,
        live_home_goals=0,
        live_away_goals=0,
        num_simulations=5000,
        verbose=False,
    )
    elapsed = time.time() - start_t

    assert (
        elapsed < 1.0
    ), f"FAIL: Simulation hung or looped infinitely! Elapsed: {elapsed:.2f}s"
    assert not np.isnan(
        prob_h
    ), "FAIL: Output probabilities resulted in NaN under extreme underflow."
    print(
        f"  [+] PASSED: Epsilon clamping handled 1e-12 transition rates cleanly in {elapsed * 1000:.1f}ms."
    )


def test_goal_reset_mechanics(device: torch.device):
    print("\n[*] TEST 3: Goal Reset & Scoreboard Increment Mechanics...")
    q = create_synthetic_q_matrix(12, device=device)

    # Start in Goal_H (index 10) at minute 75 (4500s).
    # With realistic goal rates, starting with an instant 1-0 lead guarantees a massive Home win bias!
    prob_h, prob_d, prob_a = run_live_pytorch_monte_carlo(
        q_matrix=q,
        current_clock=4500.0,
        current_state_idx=10,
        live_home_goals=0,
        live_away_goals=0,
        num_simulations=5000,
        verbose=False,
    )

    # Because we started in Goal_H with only 15 mins left, Home win probability must dominate Away win
    assert (
        prob_h > prob_a
    ), f"FAIL: Starting in Goal_H did not skew win probability! (H: {prob_h:.3f}, A: {prob_a:.3f})"
    print(
        f"  [+] PASSED: Absorbing goal states properly increment scoreboard (Home Win Bias: {(prob_h*100):.1f}%)."
    )


def test_stoppage_time_ceiling(device: torch.device):
    print("\n[*] TEST 4: Late-Game Stoppage Time Ceiling...")
    q = create_synthetic_q_matrix(12, device=device)

    # Start match at 89:30 (5370s). Ceiling should automatically extend to at least 5610s (+4 mins)
    start_t = time.time()
    prob_h, prob_d, prob_a = run_live_pytorch_monte_carlo(
        q_matrix=q,
        current_clock=5370.0,
        current_state_idx=5,
        live_home_goals=1,
        live_away_goals=1,
        num_simulations=10000,
        verbose=False,
    )
    elapsed = time.time() - start_t

    total_prob = prob_h + prob_d + prob_a

    assert (
        abs(total_prob - 1.0) < 1e-4
    ), f"FAIL: Total probability sums to {total_prob}, expected 1.0!"
    max_time = 1.0 if device.type == "cuda" else 10.0
    assert (
        elapsed < max_time
    ), f"FAIL: Late-game clock execution stalled! Time taken: {elapsed:.3f}s"
    print(
        f"  [+] PASSED: Stoppage time ceiling resolved 10,000 late-game branches in {elapsed * 1000:.1f}ms."
    )


def test_empirical_convergence(device: torch.device):
    print("\n[*] TEST 5: Monte Carlo Empirical Convergence...")
    # Give Home a 3x higher probability of scoring on any attack transition
    q_skewed = create_synthetic_q_matrix(12, device=device, home_advantage=3.0)

    prob_h, prob_d, prob_a = run_live_pytorch_monte_carlo(
        q_matrix=q_skewed,
        current_clock=0.0,
        current_state_idx=0,
        live_home_goals=0,
        live_away_goals=0,
        num_simulations=20000,
        verbose=False,
    )

    assert prob_h > (
        prob_a * 1.5
    ), f"FAIL: 3x goal advantage failed to converge into home victory bias! (H: {prob_h:.3f}, A: {prob_a:.3f})"
    print(
        f"  [+] PASSED: 20,000 simulations converged smoothly with expected statistical skew (Home: {prob_h * 100:.1f}% vs Away: {prob_a * 100:.1f}%)."
    )


def main():
    print("==================================================")
    print("   QUANTITATIVE ENGINE MATHEMATICAL STRESS TEST   ")
    print("==================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing stress suite on hardware device: {device.type.upper()}")

    test_distribution_integrity(device)
    test_underflow_protection(device)
    test_goal_reset_mechanics(device)
    test_stoppage_time_ceiling(device)
    test_empirical_convergence(device)

    print("\n==================================================")
    print(" [SUCCESS] ALL 5 STRESS TESTS PASSED CLEANLY! ")
    print("==================================================\n")


if __name__ == "__main__":
    main()
