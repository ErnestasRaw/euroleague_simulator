import os
import pandas as pd
from constants import INPUT_DIR

def validate_data():
    # Load and standardize results data
    results_path = os.path.join(INPUT_DIR, "results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found.")
    df_results = pd.read_csv(results_path)
    required_result_cols = {"HomeTeam", "AwayTeam", "HomePTS", "AwayPTS"}
    if not required_result_cols.issubset(df_results.columns):
        raise ValueError("Missing required columns in results.csv")
    df_results = df_results.rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"})
    df_results["Home"] = df_results["Home"].astype(str).str.strip()
    df_results["Away"] = df_results["Away"].astype(str).str.strip()

    # Load remaining games data (if available)
    remaining_path = os.path.join(INPUT_DIR, "remaining_games.csv")
    if os.path.exists(remaining_path):
        df_remaining = pd.read_csv(remaining_path)
        for col in ["Home", "Away"]:
            if col in df_remaining.columns:
                df_remaining[col] = df_remaining[col].astype(str).str.strip()
    else:
        df_remaining = pd.DataFrame(columns=["Home", "Away"])

    # Combine datasets for a complete season view
    df_total = pd.concat([df_results[["Home", "Away"]], df_remaining[["Home", "Away"]]], ignore_index=True)
    
    # Determine teams and expected games count
    teams = set(df_total["Home"]).union(set(df_total["Away"]))
    n = len(teams)
    expected_games = n * (n - 1)
    total_games = len(df_total)
    if total_games != expected_games:
        raise ValueError(f"Invalid game count: expected {expected_games}, got {total_games}")

    # Check home and away counts: each team should have (n-1) games in each role
    home_counts = df_total.groupby("Home").size().to_dict()
    away_counts = df_total.groupby("Away").size().to_dict()
    for team in teams:
        expected_count = n - 1
        if home_counts.get(team, 0) != expected_count:
            raise ValueError(f"{team} has {home_counts.get(team, 0)} home games, expected {expected_count}")
        if away_counts.get(team, 0) != expected_count:
            raise ValueError(f"{team} has {away_counts.get(team, 0)} away games, expected {expected_count}")

    # Verify every ordered pair appears exactly once
    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                continue
            count = len(df_total[(df_total["Home"] == team_a) & (df_total["Away"] == team_b)])
            if count != 1:
                raise ValueError(f"Expected 1 game with {team_a} vs {team_b} (home for {team_a}), got {count}")

    print("Validation passed: Regular season structure is correct with full results and remaining games.")

if __name__ == "__main__":
    validate_data()
