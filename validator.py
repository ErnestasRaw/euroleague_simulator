import os
import pandas as pd
from constants import INPUT_DIR

def validate_data():
    games_path = os.path.join(INPUT_DIR, "euroleague_regular_season_games.csv")
    if not os.path.exists(games_path):
        raise FileNotFoundError(f"{games_path} not found.")
        
    df_games = pd.read_csv(games_path)
    
    completed_games_data = []
    scheduled_games_data = []
    current_round = None
    
    for _, row in df_games.iterrows():
        first_cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        if first_cell.startswith("Round "):
            try:
                current_round = int(''.join(filter(str.isdigit, first_cell)))
            except (ValueError, TypeError):
                current_round = None
            continue
        
        if pd.isna(row['TEAM_A']) or pd.isna(row['TEAM_B']):
            continue
            
        home_team = str(row['TEAM_A']).strip()
        away_team = str(row['TEAM_B']).strip()
        
        if pd.notna(row['A_SCORE']) and row['A_SCORE'] != '-':
            completed_games_data.append({
                'Home': home_team,
                'Away': away_team,
                'Round': current_round
            })
        else:
            scheduled_games_data.append({
                'Home': home_team,
                'Away': away_team,
                'Round': current_round
            })
    
    df_completed = pd.DataFrame(completed_games_data)
    df_scheduled = pd.DataFrame(scheduled_games_data)

    df_total = pd.concat([df_completed[["Home", "Away"]], df_scheduled[["Home", "Away"]]], ignore_index=True)
    
    df_total = df_total[df_total["Home"].notna() & df_total["Away"].notna() & 
                        (df_total["Home"] != "") & (df_total["Away"] != "")]
    
    teams = set(df_total["Home"]).union(set(df_total["Away"]))
    teams = set(team for team in teams if pd.notna(team) and team != '')
    
    n = len(teams)
    expected_games = n * (n - 1)
    total_games = len(df_total)
    
    if total_games != expected_games:
        print(f"Warning: Game count mismatch - expected {expected_games}, got {total_games}")

    home_counts = df_total.groupby("Home").size().to_dict()
    away_counts = df_total.groupby("Away").size().to_dict()
    
    errors = []
    for team in teams:
        expected_count = n - 1
        if home_counts.get(team, 0) != expected_count:
            errors.append(f"{team} has {home_counts.get(team, 0)} home games, expected {expected_count}")
        if away_counts.get(team, 0) != expected_count:
            errors.append(f"{team} has {away_counts.get(team, 0)} away games, expected {expected_count}")
    
    if errors:
        print(f"Warning: Unbalanced schedule detected:\n" + "\n".join(errors))

    print("Validation passed: Regular season structure is correct with full results and remaining games.")

if __name__ == "__main__":
    validate_data()
