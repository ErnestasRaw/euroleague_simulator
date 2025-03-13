import os
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt  # Added for generating graphs

# Load CSV data, preprocess, and calculate Elo
#
# TODO write some scraping code to get the data.
# Relevant links: [
# https://www.flashscore.com/basketball/europe/euroleague/results/,
# https://www.euroleaguebasketball.net/en/euroleague/game-center/,
# ]
games_raw = pd.read_csv("results.csv")
games = games_raw[pd.to_numeric(games_raw["HomePTS"], errors='coerce').notnull()][["HomeTeam", "AwayTeam", "HomePTS", "AwayPTS"]]

teams = list(set(games['HomeTeam']).union(set(games['AwayTeam'])))
elo = {team: 1500 for team in teams}
current_wins = {team: 0 for team in teams}

# Update Elo and current wins for past games
for _, row in games.iterrows():
    home, away = row['HomeTeam'], row['AwayTeam']
    try:
        home_pts = float(row['HomePTS'])
        away_pts = float(row['AwayPTS'])
    except:
        continue  # skip if points cannot be parsed
    # Determine game outcome
    if home_pts > away_pts:
        S_home, S_away = 1, 0
        current_wins[home] += 1
    else:
        S_home, S_away = 0, 1
        current_wins[away] += 1
    # Calculate expected scores (include home advantage for home)
    home_E = 1 / (1 + 10 ** ((elo[away] - (elo[home] + 100)) / 400))
    away_E = 1 / (1 + 10 ** (((elo[home] + 100) - elo[away]) / 400))
    # Update Elo ratings with K = 32
    elo[home] += 32 * (S_home - home_E)
    elo[away] += 32 * (S_away - away_E)

# Load remaining schedule from CSV file
remaining_games = pd.read_csv("remaining_games.csv").to_dict(orient="records")

# Clean team names and ensure all teams from remaining_games are present in ratings
for game in remaining_games:
    game['Home'] = game['Home'].strip()
    game['Away'] = game['Away'].strip()
    for team in [game['Home'], game['Away']]:
        if team not in elo:
            elo[team] = 1500
            current_wins[team] = 0
            teams.append(team)

# Variable: selected team
selected_team = "Zalgiris Kaunas"

# Prepare simulation aggregators for final positions and wins
n_simulations = 10000
final_positions = {team: [0]*len(teams) for team in teams}
wins_sum = {team: 0 for team in teams}

# Monte Carlo simulation
for _ in range(n_simulations):
    standings = copy.deepcopy(current_wins)
    # Simulate remaining games
    for game in remaining_games:
        home, away = game['Home'], game['Away']
        home_elo = elo[home] + 100  # home advantage
        away_elo = elo[away]
        p_home_win = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        if np.random.rand() < p_home_win:
            standings[home] += 1
        else:
            standings[away] += 1
    sorted_teams = sorted(teams, key=lambda t: (standings[t], elo[t]), reverse=True)
    # Aggregate finishing positions and wins for each team
    for pos, team in enumerate(sorted_teams, start=1):
        final_positions[team][pos-1] += 1
        wins_sum[team] += standings[team]

# Create output folder if not exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Build final table prediction data: average wins, average finishing position and per position percentages.
final_data = []
for team in teams:
    avg_wins = wins_sum[team] / n_simulations
    avg_pos = sum((i+1)*cnt for i, cnt in enumerate(final_positions[team])) / n_simulations
    pos_perc = [cnt/n_simulations for cnt in final_positions[team]]
    row = {"Team": team, "AvgWins": avg_wins, "AvgPosition": avg_pos}
    # Add dynamic columns for finishing chance per rank
    for i, perc in enumerate(pos_perc, start=1):
        row[f"Pos{i}"] = perc
    final_data.append(row)
final_df = pd.DataFrame(final_data)
final_df.to_csv(os.path.join(output_dir, "final_table_prediction.csv"), index=False)

# Build final bracket prediction: teams sorted by average finishing position.
bracket_df = final_df.sort_values("AvgPosition")
bracket_df.to_csv(os.path.join(output_dir, "final_bracket_prediction.csv"), index=False)

# Build every game prediction: compute win probabilities based on Elo (these are static per game)
game_predictions = []
for game in remaining_games:
    home = game['Home']
    away = game['Away']
    home_elo = elo[home] + 100  # home advantage
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
games_df.to_csv(os.path.join(output_dir, "every_game_prediction.csv"), index=False)

# Build the selected team's detailed output
sel = final_df[final_df["Team"] == selected_team].iloc[0]
selected_detail = {
    "Team": selected_team,
    "AvgWins": sel["AvgWins"],
    "AvgPosition": sel["AvgPosition"]
}
# Add each finishing position as a new column.
for i in range(len(teams)):
    selected_detail[f"Pos{i+1}"] = sel[f"Pos{i+1}"]
pd.DataFrame([selected_detail]).to_csv(os.path.join(output_dir, "selected_team_detail.csv"), index=False)

# Generate graphs/tables as images with legends
positions = [f"Pos{i}" for i in range(1, len(teams)+1)]
finishing_probs = [sel[f"Pos{i}"] * 100 for i in range(1, len(teams)+1)]
plt.figure(figsize=(10,6))
plt.bar(positions, finishing_probs, label=selected_team)
plt.title("Finishing Position Probabilities")
plt.xlabel("Finishing Position")
plt.ylabel("Probability (%)")
plt.legend()
plt.savefig(os.path.join(output_dir, "selected_team_finishing_probabilities.png"))
plt.close()

sorted_df = final_df.sort_values("AvgWins", ascending=False)
plt.figure(figsize=(12,8))
plt.bar(sorted_df["Team"], sorted_df["AvgWins"], label="Avg Wins")
plt.title("Average Wins per Team")
plt.xlabel("Team")
plt.ylabel("Average Wins")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_wins.png"))
plt.close()

print("Simulation complete. Outputs and graphs generated in the 'output' folder.")