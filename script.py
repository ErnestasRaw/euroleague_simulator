import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
from constants import INPUT_DIR, OUTPUT_DIR, N_SIMULATIONS, INITIAL_ELO, HOME_ADVANTAGE, K_FACTOR, SELECTED_TEAM

def load_and_prepare_data():
    # Load CSV and prepare initial Elo and wins
    games_raw = pd.read_csv(os.path.join(INPUT_DIR, "results.csv"))
    games = games_raw[pd.to_numeric(games_raw["HomePTS"], errors='coerce').notnull()][["HomeTeam", "AwayTeam", "HomePTS", "AwayPTS"]]
    teams = list(set(games['HomeTeam']).union(set(games['AwayTeam'])))
    elo = {team: INITIAL_ELO for team in teams}
    current_wins = {team: 0 for team in teams}
    
    # Update Elo and wins from past games
    for _, row in games.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        try:
            home_pts = float(row['HomePTS'])
            away_pts = float(row['AwayPTS'])
        except:
            continue
        if home_pts > away_pts:
            S_home, S_away = 1, 0
            current_wins[home] += 1
        else:
            S_home, S_away = 0, 1
            current_wins[away] += 1
        home_E = 1 / (1 + 10 ** ((elo[away] - (elo[home] + HOME_ADVANTAGE)) / 400))
        away_E = 1 / (1 + 10 ** (((elo[home] + HOME_ADVANTAGE) - elo[away]) / 400))
        elo[home] += K_FACTOR * (S_home - home_E)
        elo[away] += K_FACTOR * (S_away - away_E)
    return teams, elo, current_wins

def load_remaining_games(elo, teams):
    remaining_games = pd.read_csv(os.path.join(INPUT_DIR, "remaining_games.csv")).to_dict(orient="records")
    for game in remaining_games:
        game['Home'] = game['Home'].strip()
        game['Away'] = game['Away'].strip()
        for team in [game['Home'], game['Away']]:
            if team not in elo:
                elo[team] = INITIAL_ELO
                teams.append(team)
    return remaining_games

def run_simulation(teams, elo, current_wins, remaining_games):
    final_positions = {team: [0]*len(teams) for team in teams}
    wins_sum = {team: 0 for team in teams}

    for _ in range(N_SIMULATIONS):
        standings = copy.deepcopy(current_wins)
        for game in remaining_games:
            home, away = game['Home'], game['Away']
            home_elo = elo[home] + HOME_ADVANTAGE
            away_elo = elo[away]
            p_home_win = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            if np.random.rand() < p_home_win:
                standings[home] += 1
            else:
                standings[away] += 1
        sorted_teams = sorted(teams, key=lambda t: (standings[t], elo[t]), reverse=True)
        for pos, team in enumerate(sorted_teams, start=1):
            final_positions[team][pos-1] += 1
            wins_sum[team] += standings[team]
    return final_positions, wins_sum
 
def export_results(teams, elo, final_positions, wins_sum):
    final_data = []
    for team in teams:
        avg_wins = wins_sum[team] / N_SIMULATIONS
        avg_pos = sum((i+1)*cnt for i, cnt in enumerate(final_positions[team])) / N_SIMULATIONS
        pos_perc = [cnt/N_SIMULATIONS for cnt in final_positions[team]]
        row = {"Team": team, "AvgWins": avg_wins, "AvgPosition": avg_pos}
        for i, perc in enumerate(pos_perc, start=1):
            row[f"Pos{i}"] = perc
        final_data.append(row)
    
    final_df = pd.DataFrame(final_data)
    final_df.to_csv(os.path.join(OUTPUT_DIR, "final_table_prediction.csv"), index=False)
    
    bracket_df = final_df.sort_values("AvgPosition")
    bracket_df.to_csv(os.path.join(OUTPUT_DIR, "final_bracket_prediction.csv"), index=False)
    
    game_predictions = []
    remaining_games = pd.read_csv(os.path.join(INPUT_DIR, "remaining_games.csv")).to_dict(orient="records")
    for game in remaining_games:
        home = game['Home'].strip()
        away = game['Away'].strip()
        home_elo = elo[home] + HOME_ADVANTAGE
        away_elo = elo[away]
        p_home_win = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        game_predictions.append({
            "Home": home,
            "Away": away,
            "p_HomeWin": p_home_win,
            "p_AwayWin": 1 - p_home_win,
            "Round": game["Round"]
        })
    games_df = pd.DataFrame(game_predictions)
    games_df.to_csv(os.path.join(OUTPUT_DIR, "every_game_prediction.csv"), index=False)
    
    sel = final_df[final_df["Team"] == SELECTED_TEAM].iloc[0]
    selected_detail = {"Team": SELECTED_TEAM,
                       "AvgWins": sel["AvgWins"],
                       "AvgPosition": sel["AvgPosition"]}
    for i in range(len(teams)):
        selected_detail[f"Pos{i+1}"] = sel[f"Pos{i+1}"]
    pd.DataFrame([selected_detail]).to_csv(os.path.join(OUTPUT_DIR, "selected_team_detail.csv"), index=False)
    
    return final_df, sel

def export_tables():
    def export_df_image(csv_filename, image_filename):
        df = pd.read_csv(os.path.join(OUTPUT_DIR, csv_filename))
        dfi.export(df, os.path.join(OUTPUT_DIR, image_filename))
    
    export_df_image("final_table_prediction.csv", "final_table_prediction_table.png")
    export_df_image("final_bracket_prediction.csv", "final_bracket_prediction_table.png")
    export_df_image("every_game_prediction.csv", "every_game_prediction_table.png")
    export_df_image("selected_team_detail.csv", "selected_team_detail_table.png")

def generate_graphs(final_df, sel, teams):
    positions = [f"Pos{i}" for i in range(1, len(teams)+1)]
    finishing_probs = [sel[f"Pos{i}"] * 100 for i in range(1, len(teams)+1)]
    plt.figure(figsize=(10,6))
    plt.bar(positions, finishing_probs, label=SELECTED_TEAM)
    plt.title("Finishing Position Probabilities")
    plt.xlabel("Finishing Position")
    plt.ylabel("Probability (%)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "selected_team_finishing_probabilities.png"))
    plt.close()

    sorted_df = final_df.sort_values("AvgWins", ascending=False)
    plt.figure(figsize=(12,8))
    bars = plt.bar(sorted_df["Team"], sorted_df["AvgWins"], label="Avg Wins")
    for idx, bar in enumerate(bars, start=1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{idx}", ha="center", va="bottom", fontsize=10)
    plt.title("Average Wins per Team")
    plt.xlabel("Team")
    plt.ylabel("Average Wins")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_wins.png"))
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    teams, elo, current_wins = load_and_prepare_data()
    remaining_games = load_remaining_games(elo, teams)
    final_positions, wins_sum = run_simulation(teams, elo, current_wins, remaining_games)
    final_df, sel = export_results(teams, elo, final_positions, wins_sum)
    export_tables()
    generate_graphs(final_df, sel, teams)
    print("Simulation complete. Outputs and graphs generated in the 'output' folder.")

if __name__ == "__main__":
    main()