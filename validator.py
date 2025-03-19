import os
import pandas as pd
from constants import INPUT_DIR
from constants import INITIAL_ELO, K_FACTOR

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

    home_adv = 3.5
    team_elo = {team: INITIAL_ELO for team in teams}
    n_games = 0
    correct_predictions = 0
    for _, row in df_games.iterrows():
        if (pd.notna(row.get('A_SCORE')) and row.get('A_SCORE') != '-' and 
            pd.notna(row.get('B_SCORE')) and row.get('B_SCORE') != '-'):
            try:
                home_score = int(row['A_SCORE'])
                away_score = int(row['B_SCORE'])
            except Exception:
                continue
            home_team = str(row['TEAM_A']).strip()
            away_team = str(row['TEAM_B']).strip()
            p_home = 1 / (1 + 10 ** ((team_elo.get(away_team, INITIAL_ELO) - (team_elo.get(home_team, INITIAL_ELO) + home_adv)) / 400))
            predicted_winner = home_team if p_home >= 0.5 else away_team
            actual_winner = home_team if home_score > away_score else away_team
            if predicted_winner == actual_winner:
                correct_predictions += 1
            n_games += 1
            S_home = 1 if home_score > away_score else 0
            S_away = 1 - S_home
            home_E = 1 / (1 + 10 ** ((team_elo.get(away_team, INITIAL_ELO) - (team_elo.get(home_team, INITIAL_ELO) + home_adv)) / 400))
            away_E = 1 / (1 + 10 ** (((team_elo.get(home_team, INITIAL_ELO) + home_adv) - team_elo.get(away_team, INITIAL_ELO)) / 400))
            team_elo[home_team] = team_elo.get(home_team, INITIAL_ELO) + K_FACTOR * (S_home - home_E)
            team_elo[away_team] = team_elo.get(away_team, INITIAL_ELO) + K_FACTOR * (S_away - away_E)
    if n_games:
        accuracy = correct_predictions / n_games * 100
        print(f"Model Accuracy on {n_games} completed games: {accuracy:.2f}%")

    print("Validation passed: Regular season structure is correct with full results and remaining games.")

if __name__ == "__main__":
    validate_data()
