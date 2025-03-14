import os
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
import constants
from constants import N_SIMULATIONS

logging.getLogger('matplotlib').setLevel(logging.WARNING)

def load_tag_mappings():
    base_dir = os.path.dirname(constants.__file__)
    mapping_path = os.path.join(base_dir, "tag_mappings.json")
    with open(mapping_path, "r") as f:
        return json.load(f)

def ordinal(n):
    n = int(n)
    if 11 <= n % 100 <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def export_results(teams, elo, final_positions, wins_sum, home_adv):
    try:
        if not os.path.exists(constants.OUTPUT_DIR):
            os.makedirs(constants.OUTPUT_DIR, exist_ok=True)
        final_data = []
        for team in teams:
            avg_wins = round(wins_sum[team] / N_SIMULATIONS, 2)
            avg_pos = round(sum((i+1)*cnt for i, cnt in enumerate(final_positions[team])) / N_SIMULATIONS, 2)
            pos_perc = [cnt / N_SIMULATIONS for cnt in final_positions[team]]
            row = {"Team": team, "AvgWins": avg_wins, "AvgPosition": avg_pos}
            for i, perc in enumerate(pos_perc, start=1):
                row[f"Pos{i}"] = perc
            final_data.append(row)
        final_df = pd.DataFrame(final_data)

        tag_mappings = load_tag_mappings()
        rename_map = tag_mappings.get("default_mapping", {})
        team_size = len(teams)
        for i in range(1, team_size+1):
            rename_map[f"Pos{i}"] = ordinal(i)
        # Sort data by expected final position
        sorted_df = final_df.sort_values("AvgPosition")
        # Swap the order of "AvgWins" and "AvgPosition" BEFORE renaming
        cols = list(sorted_df.columns)
        cols_order = ["Team", "AvgPosition", "AvgWins"] + [col for col in cols if col not in ["Team", "AvgPosition", "AvgWins"]]
        sorted_df = sorted_df[cols_order]
        final_prediction_df = sorted_df.rename(columns=rename_map)

        ordinal_columns = [rename_map[f"Pos{i}"] for i in range(1, team_size+1)]
        for col in ordinal_columns:
            if col in final_prediction_df.columns:
                final_prediction_df[col] = final_prediction_df[col].apply(
                    lambda x: "-" if round(x*100, 2)==0 else f"{x*100:.2f}%")
        
        final_prediction_df.to_csv(os.path.join(constants.OUTPUT_DIR, "final_prediction_table.csv"), index=False)
        remaining_games_path = os.path.join(constants.INPUT_DIR, "remaining_games.csv")
        if not os.path.exists(remaining_games_path):
            raise FileNotFoundError(f"{remaining_games_path} not found.")
        remaining_games = pd.read_csv(remaining_games_path).to_dict(orient="records")

        game_predictions = []
        for game in remaining_games:
            home = game['Home'].strip()
            away = game['Away'].strip()
            home_elo = elo[home] + home_adv
            away_elo = elo[away]
            p_home_win = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            game_predictions.append({
                "Home": home,
                "Away": away,
                "Home Win Probability": p_home_win, 
                "Away Win Probability": 1 - p_home_win,
                "Round": game["Round"]
            })
        game_pred_df = pd.DataFrame(game_predictions)
        for col in ["Home Win Probability", "Away Win Probability"]:
            game_pred_df[col] = game_pred_df[col].apply(
                    lambda x: "-" if round(x*100, 2)==0 else f"{x*100:.2f}%")
        game_pred_df.to_csv(os.path.join(constants.OUTPUT_DIR, "every_game_prediction.csv"), index=False)
        
        return final_df
    except Exception as e:
        import traceback
        logging.error("Error in export_results: " + traceback.format_exc())
        raise

def export_final_prediction_table():
    if not os.path.exists(constants.OUTPUT_DIR):
         os.makedirs(constants.OUTPUT_DIR, exist_ok=True)
    def export_df_image(csv_filename, image_filename, title):
         df = pd.read_csv(os.path.join(constants.OUTPUT_DIR, csv_filename))
         df.index = range(1, len(df)+1)
         create_table_image(df, os.path.join(constants.OUTPUT_DIR, image_filename), title)
    export_df_image("final_prediction_table.csv", "final_prediction_table.png", "Overall Final Prediction Table")

def export_every_game_prediction_table():
    if not os.path.exists(constants.OUTPUT_DIR):
         os.makedirs(constants.OUTPUT_DIR, exist_ok=True)
    def export_df_image(csv_filename, image_filename, title):
         df = pd.read_csv(os.path.join(constants.OUTPUT_DIR, csv_filename))
         df.index = range(1, len(df)+1)
         create_table_image(df, os.path.join(constants.OUTPUT_DIR, image_filename), title)
    export_df_image("every_game_prediction.csv", "every_game_prediction_table.png", "Every Game Prediction Table")

def create_table_image(df, path, title="Table", apply_gradient=True):
    """
    Create a table image using dataframe_image with reduced resolution
    and optional gradient highlighting.
    """
    try:
        import dataframe_image as dfi
        styled_df = df.style.set_caption(title)
        if apply_gradient:
            if "every game prediction" in title.lower():
                target_cols = ["Home Win Probability", "Away Win Probability"]
            else:
                target_cols = [ordinal(i) for i in range(1, 19)]
            cols_to_style = [col for col in df.columns if col in target_cols]
            if cols_to_style:
                def style_row(row):
                    try:
                        probs = []
                        for col in cols_to_style:
                            cell = row[col]
                            value = 0 if (isinstance(cell, str) and cell.strip() == "-") else (float(cell.strip("%"))/100 if isinstance(cell, str) and "%" in cell else float(cell))
                            probs.append(value)
                        row_max = max(probs) if probs else 0
                    except Exception:
                        row_max = 0
                    styles = []
                    for col in row.index:
                        if col in cols_to_style and row_max:
                            try:
                                cell = row[col]
                                prob = 0 if (isinstance(cell, str) and cell.strip() == "-") else (float(cell.strip("%"))/100 if isinstance(cell, str) and "%" in cell else float(cell))
                            except Exception:
                                prob = 0
                            styles.append("background-color: rgba(255,215,0,0.5);" if abs(prob - row_max) < 1e-6 else f"background-color: rgba(0,128,0,{prob/row_max});")
                        else:
                            styles.append("")
                    return styles
                styled_df = styled_df.apply(style_row, axis=1)
        dfi.export(styled_df, path, dpi=80)
    except ImportError as e:
        raise ImportError("dataframe_image is required. Install via 'pip install dataframe_image'.")
    except Exception as e:
        raise RuntimeError("Error generating table image: " + str(e))

def export_team_results(final_df, teams):
    teams_folder = os.path.join(constants.OUTPUT_DIR, "teams")
    os.makedirs(teams_folder, exist_ok=True)
    for _, row in final_df.iterrows():
        team = row["Team"]
        team_folder = os.path.join(teams_folder, team.replace(" ", "_"))
        os.makedirs(team_folder, exist_ok=True)
        team_detail_df = pd.DataFrame([row])
        # Removed index setting for single row as requested
        csv_path = os.path.join(team_folder, "team_detail.csv")
        team_detail_df.to_csv(csv_path, index=False)
        tag_mappings = load_tag_mappings()
        rename_map = tag_mappings.get("default_mapping", {})
        team_size = len(teams)
        for i in range(1, team_size+1):
            rename_map[f"Pos{i}"] = ordinal(i)
        team_detail = team_detail_df.rename(columns=rename_map)
        team_detail_png = team_detail.copy()
        cols = [rename_map[f"Pos{i}"] for i in range(1, team_size+1)]
        for col in cols:
            if col in team_detail_png.columns:
                team_detail_png[col] = team_detail_png[col].apply(
                    lambda x: "-" if round(float(x)*100,2)==0 else f"{float(x)*100:.2f}%"
                )
        img_png_path = os.path.join(team_folder, "team_detail_table.png")
        create_table_image(team_detail_png, img_png_path, f"Team Detail for {team}")
        positions = [rename_map[f"Pos{i}"] for i in range(1, team_size+1)]
        finishing_probs = [row[f"Pos{i}"] * 100 for i in range(1, team_size+1)]
        plt.figure(figsize=(10,6))
        bars = plt.bar(positions, finishing_probs, label=team)
        plt.title(f"{team} Finishing Probabilities")
        plt.xlabel("Finishing Position")
        plt.ylabel("Probability (%)")
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=10)
        plt.legend()
        plt.tight_layout()
        graph_path = os.path.join(team_folder, "finishing_probabilities.png")
        plt.savefig(graph_path)
        plt.close()

def generate_graphs(final_df, teams):
    sorted_df = final_df.sort_values("AvgWins", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12,8))
    bars = plt.bar(range(len(sorted_df)), sorted_df["AvgWins"], label="Expected Wins")
    x_labels = [f"{ordinal(i+1)}: {sorted_df.loc[i, 'Team']}" for i in range(len(sorted_df))]
    plt.xticks(range(len(sorted_df)), x_labels, rotation=45, ha="right")
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
    plt.title("Average Wins per Team")
    plt.xlabel("Ranking: Team")
    plt.ylabel("Expected Wins")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(constants.OUTPUT_DIR, "average_wins.png"))
    plt.close()
