import os
import copy
import logging
import pandas as pd
import numpy as np
from constants import INPUT_DIR, OUTPUT_DIR, N_SIMULATIONS, INITIAL_ELO, K_FACTOR
from exporter import export_results, generate_graphs
import hashlib
from datetime import datetime
import uuid
import json
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn
import matplotlib
matplotlib.use('Agg')

class StepFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'step'):
            record.step = 0
        return True

def load_and_prepare_data():
    try:
        file_path = os.path.join(INPUT_DIR, "euroleague_regular_season_games.csv")
        games_raw = pd.read_csv(file_path)
        
        completed_games_data = []
        scheduled_games_data = []
        
        current_round = None
        for _, row in games_raw.iterrows():
            # Check if this is a round header row
            first_cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            if first_cell.startswith("Round "):
                try:
                    current_round = int(''.join(filter(str.isdigit, first_cell)))
                except (ValueError, TypeError):
                    current_round = None
                continue
            
            # Skip rows without team names
            if pd.isna(row['TEAM_A']) or pd.isna(row['TEAM_B']):
                continue
                
            home_team = str(row['TEAM_A']).strip()
            away_team = str(row['TEAM_B']).strip()
            
            # Check if game is completed or scheduled
            if pd.notna(row['A_SCORE']) and row['A_SCORE'] != '-':
                # Completed game
                try:
                    home_score = int(row['A_SCORE'])
                    away_score = int(row['B_SCORE'])
                    completed_games_data.append({
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'HomePTS': home_score,
                        'AwayPTS': away_score,
                        'Round': current_round
                    })
                except (ValueError, TypeError):
                    logging.warning(f"Invalid score format: {row['A_SCORE']}-{row['B_SCORE']} for {home_team} vs {away_team}")
            else:
                # Scheduled game
                scheduled_games_data.append({
                    'Home': home_team,
                    'Away': away_team,
                    'Round': current_round
                })
        
        completed_games = pd.DataFrame(completed_games_data)
        scheduled_games = pd.DataFrame(scheduled_games_data)
        
        default_adv = 3.5  # 
        home_margin_info = {}
        for _, row in completed_games.iterrows():
            team = row["HomeTeam"]
            margin = row["HomePTS"] - row["AwayPTS"]
            home_margin_info.setdefault(team, []).append(margin)
        
        teams_completed = set(completed_games["HomeTeam"].tolist() + completed_games["AwayTeam"].tolist()) if not completed_games.empty else set()
        teams_scheduled = set(scheduled_games["Home"].tolist() + scheduled_games["Away"].tolist()) if not scheduled_games.empty else set()
        teams = list(filter(None, teams_completed.union(teams_scheduled)))
        
        home_advantages = {}
        for team in teams:
            if team in home_margin_info and len(home_margin_info[team]) > 0:
                home_advantages[team] = np.mean(home_margin_info[team])
            else:
                home_advantages[team] = default_adv
        
        elo = {team: INITIAL_ELO for team in teams}
        current_wins = {team: 0 for team in teams}
        
        for _, row in completed_games.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            home_pts, away_pts = row["HomePTS"], row["AwayPTS"]
            
            if home_pts > away_pts:
                S_home, S_away = 1, 0
                current_wins[home] += 1
            else:
                S_home, S_away = 0, 1
                current_wins[away] += 1
                
            team_home_adv = home_advantages.get(home, default_adv)
            home_E = 1 / (1 + 10 ** ((elo[away] - (elo[home] + team_home_adv)) / 400))
            away_E = 1 / (1 + 10 ** (((elo[home] + team_home_adv) - elo[away]) / 400))
            
            elo[home] += K_FACTOR * (S_home - home_E)
            elo[away] += K_FACTOR * (S_away - away_E)
            
        remaining_games = scheduled_games.to_dict(orient="records")
        
        return teams, elo, current_wins, home_advantages, remaining_games
    except Exception as e:
        logging.error(f"Error in load_and_prepare_data: {e}", exc_info=True)
        raise

def run_simulation(teams, elo, current_wins, remaining_games, home_adv):
    try:
        final_positions = {team: [0]*len(teams) for team in teams}
        wins_sum = {team: 0 for team in teams}
        
        for _ in range(N_SIMULATIONS):
            standings = copy.deepcopy(current_wins)
            
            for game in remaining_games:
                home, away = game['Home'], game['Away']
                if not home or not away or home not in elo or away not in elo:
                    continue
                    
                team_home_adv = home_adv.get(home, 3.5)
                home_elo = elo[home] + team_home_adv
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
    except Exception:
        logging.error("Error in run_simulation", exc_info=True)
        raise

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_input_hashes():
    input_file = os.path.join(INPUT_DIR, "euroleague_regular_season_games.csv")
    if os.path.exists(input_file):
        return {os.path.basename(input_file): compute_file_hash(input_file)}
    return {"euroleague_regular_season_games.csv": "file_not_found"}

def backup_input_data_zip(output_folder):
    from zipfile import ZipFile
    zip_path = os.path.join(output_folder, "input_data.zip")
    input_file = os.path.join(INPUT_DIR, "euroleague_regular_season_games.csv")
    
    if os.path.exists(input_file):
        with ZipFile(zip_path, 'w') as zipf:
            zipf.write(input_file, arcname=os.path.basename(input_file))

def extract_round_numbers(df):
    round_data = []
    
    for idx, row in df.iterrows():
        first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        if first_col.startswith("Round "):
            try:
                round_num = int(''.join(filter(str.isdigit, first_col)))
                round_data.append(round_num)
            except (ValueError, TypeError):
                pass
    
    return round_data

def find_rounds_with_unplayed_games(df):
    rounds_with_unplayed_games = []
    current_round = None
    
    for _, row in df.iterrows():
        first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        if first_col.startswith("Round "):
            try:
                round_num = int(''.join(filter(str.isdigit, first_col)))
                current_round = round_num
            except (ValueError, TypeError):
                current_round = None
        elif current_round is not None and pd.notna(row.get("TEAM_A")) and pd.notna(row.get("TEAM_B")):
            if pd.notna(row.get("A_SCORE")) and row.get("A_SCORE") == '-':
                rounds_with_unplayed_games.append(current_round)
    
    return sorted(set(rounds_with_unplayed_games)) if rounds_with_unplayed_games else []

def check_round_status(df, round_num):
    round_games = []
    in_target_round = False
    
    for _, row in df.iterrows():
        first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
        
        if first_col == f"Round {round_num}":
            in_target_round = True
            continue
        elif in_target_round and first_col.startswith("Round "):
            break
            
        if in_target_round and pd.notna(row.get("TEAM_A")) and pd.notna(row.get("TEAM_B")):
            round_games.append(row)
    
    total_games = len(round_games)
    if total_games == 0:
        return "unknown"
        
    unplayed_games = sum(1 for row in round_games if row["A_SCORE"] == '-')
    
    if unplayed_games == total_games:
        return "full"  # All games in this round are unplayed
    elif unplayed_games == 0:
        return "complete"  # All games in this round are played
    else:
        return "mid_round"  # Some games in this round are played, some unplayed

def prepare_output_directory():
    input_hashes = get_input_hashes()
    games_file_path = os.path.join(INPUT_DIR, "euroleague_regular_season_games.csv")
    
    round_folder = "unknown_round"
    stage = "mid_round"
    min_round = max_round = 0
    remaining_count = 0
    
    if os.path.exists(games_file_path):
        try:
            df = pd.read_csv(games_file_path)
            
            unplayed_rounds = find_rounds_with_unplayed_games(df)
            
            if unplayed_rounds:
                min_round = min(unplayed_rounds)
                max_round = max(unplayed_rounds)
                round_folder = f"R{min_round}-{max_round}"
                
                stage = check_round_status(df, min_round)
                remaining_count = len(df[df["A_SCORE"] == "-"])
            else:
                all_rounds = extract_round_numbers(df)
                
                if all_rounds:
                    min_round = max(all_rounds)
                    max_round = min_round
                    round_folder = f"R{min_round}"
                    stage = "complete"
                    remaining_count = 0
        except Exception as e:
            logging.error(f"Error processing game data: {e}")
    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique = uuid.uuid4().hex[:6]
    unique_folder = f"{now}_{unique}"
    new_folder = os.path.join(OUTPUT_DIR, round_folder, stage, unique_folder)
    
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(os.path.join(new_folder, "teams"), exist_ok=True)
    
    backup_input_data_zip(new_folder)
    
    input_values = {}
    for file_path in [games_file_path]:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                input_values[os.path.basename(file_path)] = {
                    "size": len(content),
                    "hash": compute_file_hash(file_path),
                    "sample": content[:100] if len(content) > 100 else content
                }
            except Exception:
                pass
    
    key_data = {
         "timestamp": now,
         "input_hashes": input_hashes,
         "round_info": {"min_round": min_round, "max_round": max_round},
         "remaining_games_count": remaining_count,
         "input_values": input_values
    }
    key_file = os.path.join(new_folder, "data_key.json")
    with open(key_file, "w") as f:
         json.dump(key_data, f)
    
    return new_folder

def update_default_folder(sim_folder):
    default_folder = os.path.join(OUTPUT_DIR, "default")
    sim_key_file = os.path.join(sim_folder, "data_key.json")
    
    try:
        with open(sim_key_file, "r") as f:
            sim_key = json.load(f)
    except Exception:
        return
        
    sim_count = sim_key.get("remaining_games_count", 0)
    sim_time_str = sim_key.get("timestamp", "1970-01-01_00-00-00")
    from datetime import datetime as dt
    sim_time = dt.strptime(sim_time_str, "%Y-%m-%d_%H-%M-%S")
    
    update_default = False
    if os.path.exists(default_folder):
         default_key_file = os.path.join(default_folder, "data_key.json")
         try:
             with open(default_key_file, "r") as f:
                 default_key = json.load(f)
             default_count = default_key.get("remaining_games_count", 0)
             default_time_str = default_key.get("timestamp", "1970-01-01_%H-%M-%S")
             default_time = dt.strptime(default_time_str, "%Y-%m-%d_%H-%M-%S")
             
             if sim_count < default_count or (sim_count == default_count and sim_time > default_time):
                 update_default = True
         except Exception:
             update_default = True
    else:
         update_default = True

    if update_default:
         import shutil
         if os.path.exists(default_folder):
             shutil.rmtree(default_folder)
         shutil.copytree(sim_folder, default_folder)

def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='STEP %(step)d/7: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, 'simulation.log'))
        ]
    )
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.addFilter(StepFilter())
    return logger

def main():
    new_output_dir = prepare_output_directory()
    import constants
    constants.OUTPUT_DIR = new_output_dir
    os.makedirs(new_output_dir, exist_ok=True)
    
    console = Console()
    completed_steps = []
    context = {}
    
    steps = [
        {
            "desc": "Data validation",
            "func": lambda: __import__("validator").validate_data()
        },
        {
            "desc": "Load and prepare data",
            "func": lambda: context.update(
                dict(zip(
                    ["teams", "elo", "current_wins", "home_adv", "remaining_games"],
                    load_and_prepare_data()
                ))
            )
        },
        {
            "desc": "Run simulation",
            "func": lambda: context.update(
                dict(zip(
                    ["final_positions", "wins_sum"],
                    run_simulation(context["teams"], context["elo"], context["current_wins"],
                                   context["remaining_games"], context["home_adv"])
                ))
            )
        },
        {
            "desc": "Export results",
            "func": lambda: context.update({
                "final_df": export_results(context["teams"], context["elo"],
                                            context["final_positions"], context["wins_sum"],
                                            np.mean(list(context["home_adv"].values())))
            })
        },
        {
            "desc": "Export final prediction table image",
            "func": lambda: __import__("exporter").export_final_prediction_table()
        },
        {
            "desc": "Export every game prediction table image",
            "func": lambda: __import__("exporter").export_every_game_prediction_table()
        },
        {
            "desc": "Export predicted final standings", 
            "func": lambda: __import__("exporter").export_predicted_final_standings()
        },
        {
            "desc": "Export team results",
            "func": lambda: __import__("exporter").export_team_results(context["final_df"], context["teams"])
        },
        {
            "desc": "Generate overall graphs",
            "func": lambda: generate_graphs(context["final_df"])
        }
    ]
    
    total_steps = len(steps)
    with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
            ) as progress:
        task = progress.add_task("Simulation Progress", total=total_steps)
        
        for step in steps:
            try:
                step["func"]()
                completed_steps.append(f"{step['desc']} was successful")
            except Exception as e:
                completed_steps.append(f"{step['desc']} have failed")
                console.print(f"[red]{step['desc']} failed: {e}[/red]")
                return
            progress.advance(task)
            console.print("Completed Steps:", completed_steps)
    
    update_default_folder(new_output_dir)
    console.print("[green]Simulation complete.[/green]")

if __name__ == "__main__":
    main()