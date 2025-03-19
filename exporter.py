import os
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
import constants
from constants import N_SIMULATIONS

logging.getLogger('matplotlib').setLevel(logging.WARNING)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def ordinal(n):
    n = int(n)
    if 11 <= n % 100 <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def get_rename_map(team_count):
    rename_map = {}
    for i in range(1, team_count+1):
        rename_map[f"Pos{i}"] = ordinal(i)
    return rename_map

def format_percentage(cell):
    try:
        value = float(cell) if not isinstance(cell, str) else float(cell.strip("%"))
    except Exception:
        return "-"
    return "-" if round(value*100,2)==0 else f"{value*100:.2f}%"

def create_styled_df(df, title, target_cols, apply_gradient=True):
    styled_df = df.style.set_caption(title).set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
        {'selector': 'caption', 'props': [('font-weight', 'bold'), ('font-size', '1.1em')]},
        {'selector': 'thead tr th', 'props': [('background-color', '#f2f2f2'), ('font-weight', 'bold')]}
    ])
    if apply_gradient and target_cols:
        def style_row(row):
            probs = []
            for col in target_cols:
                cell = row[col] if col in row else 0
                try:
                    val = 0 if (isinstance(cell, str) and cell.strip() == "-") else \
                          (float(cell.strip("%"))/100 if isinstance(cell, str) and "%" in cell else float(cell))
                except Exception:
                    val = 0
                probs.append(val)
            row_max = max(probs) if probs and max(probs) else 0
            styles = []
            for col in row.index:
                if col in target_cols and row_max:
                    try:
                        cell = row[col]
                        prob = 0 if (isinstance(cell, str) and cell.strip() == "-") else \
                               (float(cell.strip("%"))/100 if isinstance(cell, str) and "%" in cell else float(cell))
                    except Exception:
                        prob = 0
                    styles.append("background-color: rgba(255,215,0,0.5);" if abs(prob - row_max) < 1e-6 
                                  else f"background-color: rgba(0,128,0,{prob/row_max});")
                else:
                    styles.append("")
            return styles
        styled_df = styled_df.apply(style_row, axis=1)
    return styled_df

def export_table_image(df, out_path, title, target_cols, dpi=80, max_rows=None):
    if out_path.lower().endswith(".png"):
        try:
            round_info = os.path.basename(os.path.dirname(os.path.dirname(constants.OUTPUT_DIR)))
            title += f" ({round_info})"
        except Exception:
            pass
    styled_df = create_styled_df(df, title, target_cols)
    try:
        import dataframe_image as dfi
        if max_rows and len(df) > max_rows:
            logging.info(f"Table has {len(df)} rows, limiting to {max_rows} rows for image export")
            styled_df = create_styled_df(df.head(max_rows), f"{title} (First {max_rows} rows)", target_cols)
        dfi.export(styled_df, out_path, dpi=dpi)
    except Exception as e:
        raise RuntimeError("Error generating table image: " + str(e))

def export_results(teams, elo, final_positions, wins_sum, home_adv):
    try:
        teams = [team for team in teams if pd.notna(team) and team != '']
        
        ensure_dir(constants.OUTPUT_DIR)
        final_data = []
        for team in teams:
            avg_wins = wins_sum[team] / N_SIMULATIONS
            avg_pos = sum((i+1) * cnt for i, cnt in enumerate(final_positions[team])) / N_SIMULATIONS
            pos_perc = [cnt / N_SIMULATIONS for cnt in final_positions[team]]
            
            top6_prob = sum(pos_perc[:6])
            top10_prob = sum(pos_perc[:10])
            
            row = {
                "Team": team, 
                "AvgPosition": avg_pos, 
                "AvgWins": avg_wins,
                "Top6": top6_prob,
                "Top10": top10_prob
            }
            
            for i, perc in enumerate(pos_perc, start=1):
                row[f"Pos{i}"] = perc
            final_data.append(row)
        final_df = pd.DataFrame(final_data)
        team_count = len(teams)
        rename_map = get_rename_map(team_count)
        base_cols = ["Team", "AvgPosition", "AvgWins", "Top6", "Top10"]
        other_cols = [col for col in final_df.columns if col not in base_cols]
        sorted_df = final_df[base_cols + other_cols].sort_values("AvgPosition")
        final_prediction_df = sorted_df.rename(columns=rename_map)
        
        ordinal_cols = [rename_map[f"Pos{i}"] for i in range(1, team_count+1)]
        for col in ordinal_cols:
            if col in final_prediction_df.columns:
                final_prediction_df[col] = final_prediction_df[col].apply(format_percentage)
                
        if "Top6" in final_prediction_df.columns:
            final_prediction_df["Top6"] = final_prediction_df["Top6"].apply(format_percentage)
        if "Top10" in final_prediction_df.columns:
            final_prediction_df["Top10"] = final_prediction_df["Top10"].apply(format_percentage)
                
        final_prediction_df.to_csv(os.path.join(constants.OUTPUT_DIR, "final_prediction_table.csv"), index=False)
        
        all_games_path = os.path.join(constants.INPUT_DIR, "euroleague_regular_season_games.csv")
        all_games_df = pd.read_csv(all_games_path)
        
        all_games = []
        future_games = []  
        current_round = None
        
        for _, row in all_games_df.iterrows():
            first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            if first_col.startswith("Round "):
                try:
                    current_round = int(''.join(filter(str.isdigit, first_col)))
                except (ValueError, TypeError):
                    current_round = None
                continue
                
            if pd.notna(row.get("TEAM_A")) and pd.notna(row.get("TEAM_B")):
                game_data = {
                    'Home': row.get("TEAM_A", "Unknown").strip(),
                    'Away': row.get("TEAM_B", "Unknown").strip(),
                    'HomeScore': row.get("A_SCORE", "-"),
                    'AwayScore': row.get("B_SCORE", "-"),
                    'Round': current_round,
                    'Date': row.iloc[0] if pd.notna(row.iloc[0]) else ""  
                }
                all_games.append(game_data)
                
                if str(game_data.get('HomeScore')) == '-':
                    future_games.append(game_data)
        
        pd.DataFrame(all_games).to_csv(os.path.join(constants.OUTPUT_DIR, "all_games.csv"), index=False)
        
        game_predictions = []
        for game in future_games: 
            home = str(game.get('Home', 'Unknown')).strip()
            if home.lower() == 'nan':
                home = 'Unknown'
            away = str(game.get('Away', 'Unknown')).strip()
            if away.lower() == 'nan':
                away = 'Unknown'
            if home not in elo:
                home_elo = 0 + home_adv
            else:
                home_elo = elo[home] + home_adv
            if away not in elo:
                away_elo = 0
            else:
                away_elo = elo[away]
            p_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            game_predictions.append({
                "Home": home,
                "Away": away,
                "Home Win Probability": p_home,
                "Away Win Probability": 1 - p_home,
                "Round": game.get("Round", "Unknown"),
                "Date": game.get("Date", "")
            })
        pd.DataFrame(game_predictions).to_csv(os.path.join(constants.OUTPUT_DIR, "every_game_prediction.csv"), index=False)
        
        return final_df
    except Exception as e:
        import traceback
        logging.error("Error in export_results: " + traceback.format_exc())
        raise

def export_image_from_csv(csv_filename, image_filename, title, target_cols, max_rows=None):
    ensure_dir(constants.OUTPUT_DIR)
    csv_path = os.path.join(constants.OUTPUT_DIR, csv_filename)
    df = pd.read_csv(csv_path)
    df.index = range(1, len(df)+1)
    export_table_image(df, os.path.join(constants.OUTPUT_DIR, image_filename), title, target_cols, max_rows=max_rows)

def export_predicted_final_standings():
    """Generate a prediction of the final standings if all remaining games
    go according to our predictions (team with higher win probability wins)."""
    
    try:
        csv_path = os.path.join(constants.OUTPUT_DIR, "all_games.csv")
        all_games_df = pd.read_csv(csv_path)
        
        predictions_path = os.path.join(constants.OUTPUT_DIR, "every_game_prediction.csv")
        predictions_df = pd.read_csv(predictions_path)
        
        teams = set()
        current_wins = {}
        
        for _, row in all_games_df.iterrows():
            home = row.get('Home', 'Unknown')
            away = row.get('Away', 'Unknown')
            
            if isinstance(home, str) and home.strip():
                teams.add(home)
                if home not in current_wins:
                    current_wins[home] = 0
                    
            if isinstance(away, str) and away.strip():
                teams.add(away)
                if away not in current_wins:
                    current_wins[away] = 0
            
            if row.get('HomeScore') != '-' and row.get('AwayScore') != '-':
                try:
                    home_score = int(row.get('HomeScore'))
                    away_score = int(row.get('AwayScore'))
                    
                    if home_score > away_score:
                        current_wins[home] += 1
                    else:
                        current_wins[away] += 1
                except (ValueError, TypeError):
                    pass
        
        final_standings = current_wins.copy()
        
        for _, row in predictions_df.iterrows():
            home = row['Home']
            away = row['Away']
            p_home = row['Home Win Probability']
            
            if p_home >= 0.5:
                final_standings[home] += 1
            else:
                final_standings[away] += 1
        
        standings_data = []
        for team, wins in final_standings.items():
            if team != 'Unknown' and not pd.isna(team) and team != '':
                standings_data.append({
                    'Team': team,
                    'Wins': wins
                })
        
        standings_df = pd.DataFrame(standings_data).sort_values(
            by='Wins', ascending=False).reset_index(drop=True)
        
        standings_df.insert(0, 'Position', [ordinal(i+1) for i in range(len(standings_df))])
        
        standings_df.to_csv(os.path.join(constants.OUTPUT_DIR, "predicted_final_standings.csv"), index=False)
        
        img_path = os.path.join(constants.OUTPUT_DIR, "predicted_final_standings.png")
        export_table_image(standings_df, img_path, 
                          "Predicted Final Standings (if all predictions are correct)", 
                          []) 
        
        return standings_df
    
    except Exception as e:
        logging.error(f"Error in export_predicted_final_standings: {e}", exc_info=True)
        raise

def export_final_prediction_table():
    teams = pd.read_csv(os.path.join(constants.OUTPUT_DIR, "final_prediction_table.csv"))["Team"].tolist()
    team_count = len(teams)
    target_cols = [ordinal(i) for i in range(1, team_count+1)]
    export_image_from_csv("final_prediction_table.csv", "final_prediction_table.png", "Overall Final Prediction Table", target_cols)

def export_every_game_prediction_table():
    target = ["Home Win Probability", "Away Win Probability"]
    csv_path = os.path.join(constants.OUTPUT_DIR, "every_game_prediction.csv")
    
    try:
        df = pd.read_csv(csv_path)
        
        df.to_csv(os.path.join(constants.OUTPUT_DIR, "every_game_prediction_future.csv"), index=False)
        
        max_rows = 80
        export_image_from_csv("every_game_prediction_future.csv", 
                            "every_game_prediction_table.png", 
                            "Future Game Predictions", 
                            target, 
                            max_rows=max_rows)
    except Exception as e:
        logging.error(f"Error in export_every_game_prediction_table: {e}")
        raise

def export_team_results(final_df, teams):
    teams = [team for team in teams if pd.notna(team) and team != '']
    
    teams_folder = os.path.join(constants.OUTPUT_DIR, "teams")
    ensure_dir(teams_folder)
    team_count = len(teams)
    rename_map = get_rename_map(team_count)
    ordinal_cols = [rename_map[f"Pos{i}"] for i in range(1, team_count+1)]
    
    for _, row in final_df.iterrows():
        if pd.isna(row["Team"]) or row["Team"] == '':
            continue
            
        team = str(row["Team"]) if not isinstance(row["Team"], str) else row["Team"]
        team_folder = os.path.join(teams_folder, team.replace(" ", "_"))
        ensure_dir(team_folder)
        team_detail_df = pd.DataFrame([row])
        team_detail_csv = os.path.join(team_folder, "team_detail.csv")
        team_detail_df.to_csv(team_detail_csv, index=False)
        
        team_detail = team_detail_df.rename(columns=rename_map)
        for col in ordinal_cols:
            if col in team_detail.columns:
                team_detail[col] = team_detail[col].apply(format_percentage)
                
        if "Top6" in team_detail.columns:
            team_detail["Top6"] = team_detail["Top6"].apply(format_percentage)
        if "Top10" in team_detail.columns:
            team_detail["Top10"] = team_detail["Top10"].apply(format_percentage)
                
        img_path = os.path.join(team_folder, "team_detail_table.png")
        export_table_image(team_detail, img_path, f"Team Detail for {team}", ordinal_cols)

        finishing_probs = [row[f"Pos{i}"] * 100 for i in range(1, team_count+1)]
        plt.figure(figsize=(10,6))
        bars = plt.bar(ordinal_cols, finishing_probs, label=team)
        plt.title(f"{team} Finishing Probabilities")
        plt.xlabel("Finishing Position")
        plt.ylabel("Probability (%)")
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}%", 
                     ha="center", va="bottom", fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(team_folder, "finishing_probabilities.png"))
        plt.close()

def generate_graphs(final_df):
    sorted_df = final_df.sort_values("AvgWins", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12,8))
    bars = plt.bar(range(len(sorted_df)), sorted_df["AvgWins"], label="Expected Wins")
    x_labels = [f"{ordinal(i+1)}: {sorted_df.loc[i, 'Team']}" for i in range(len(sorted_df))]
    plt.xticks(range(len(sorted_df)), x_labels, rotation=45, ha="right")
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}", 
                 ha="center", va="bottom", fontsize=10)
    plt.title("Average Wins per Team")
    plt.xlabel("Ranking: Team")
    plt.ylabel("Expected Wins")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(constants.OUTPUT_DIR, "average_wins.png"))
    plt.close()
