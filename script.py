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
        games_raw = pd.read_csv(os.path.join(INPUT_DIR, "results.csv"))
        games = games_raw[pd.to_numeric(games_raw["HomePTS"], errors='coerce').notnull()][["HomeTeam", "AwayTeam", "HomePTS", "AwayPTS"]]
        teams = list(set(games['HomeTeam']).union(set(games['AwayTeam'])))
        elo = {team: INITIAL_ELO for team in teams}
        current_wins = {team: 0 for team in teams}
        games["HomePTS_numeric"] = pd.to_numeric(games["HomePTS"], errors="coerce")
        games["AwayPTS_numeric"] = pd.to_numeric(games["AwayPTS"], errors="coerce")
        dynamic_home_advantage = games["HomePTS_numeric"].sub(games["AwayPTS_numeric"]).mean()
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
            home_E = 1 / (1 + 10 ** ((elo[away] - (elo[home] + dynamic_home_advantage)) / 400))
            away_E = 1 / (1 + 10 ** (((elo[home] + dynamic_home_advantage) - elo[away]) / 400))
            elo[home] += K_FACTOR * (S_home - home_E)
            elo[away] += K_FACTOR * (S_away - away_E)
        return teams, elo, current_wins, dynamic_home_advantage
    except Exception:
        logging.error("Error in load_and_prepare_data", exc_info=True)
        raise

def load_remaining_games(elo, teams):
    try:
        remaining_games = pd.read_csv(os.path.join(INPUT_DIR, "remaining_games.csv")).to_dict(orient="records")
        for game in remaining_games:
            game['Home'] = game['Home'].strip()
            game['Away'] = game['Away'].strip()
            for team in [game['Home'], game['Away']]:
                if team not in elo:
                    elo[team] = INITIAL_ELO
                    teams.append(team)
        return remaining_games
    except Exception:
        logging.error("Error in load_remaining_games", exc_info=True)
        raise

def run_simulation(teams, elo, current_wins, remaining_games, home_adv):
    try:
        final_positions = {team: [0]*len(teams) for team in teams}
        wins_sum = {team: 0 for team in teams}
        for _ in range(N_SIMULATIONS):
            standings = copy.deepcopy(current_wins)
            for game in remaining_games:
                home, away = game['Home'], game['Away']
                home_elo = elo[home] + home_adv
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
    input_dir = INPUT_DIR
    files = {"results.csv": os.path.join(input_dir, "results.csv"),
             "remaining_games.csv": os.path.join(input_dir, "remaining_games.csv")}
    hashes = {}
    for name, path in files.items():
        hashes[name] = compute_file_hash(path)
    return hashes

def backup_input_data_zip(sim_folder):
    from zipfile import ZipFile
    data_zip_path = os.path.join(sim_folder, "input_data.zip")
    csv_files = [os.path.join(INPUT_DIR, "results.csv"), os.path.join(INPUT_DIR, "remaining_games.csv")]
    with ZipFile(data_zip_path, 'w') as zipf:
         for file in csv_files:
             if os.path.exists(file):
                  zipf.write(file, arcname=os.path.basename(file))

def prepare_output_directory():
    input_hashes = get_input_hashes()
    FULL_GAME_COUNT = 4
    remaining_games_path = os.path.join(INPUT_DIR, "remaining_games.csv")
    if os.path.exists(remaining_games_path):
        try:
            df = pd.read_csv(remaining_games_path)
            remaining_count = df.shape[0]
            if "Round" in df.columns:
                min_round = int(df["Round"].min())
                max_round = int(df["Round"].max())
                count_min_round = df[df["Round"] == min_round].shape[0]
                round_folder = f"R{min_round}-{max_round}"
                stage = "full" if count_min_round >= FULL_GAME_COUNT else "mid_round"
            else:
                round_folder, stage, min_round, max_round, remaining_count = "unknown_round", "mid_round", 0, 0, 0
        except Exception:
            round_folder, stage, min_round, max_round, remaining_count = "unknown_round", "mid_round", 0, 0, 0
    else:
        round_folder, stage, min_round, max_round, remaining_count = "unknown_round", "mid_round", 0, 0, 0
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique = uuid.uuid4().hex[:6]
    unique_folder = f"{now}_{unique}"
    new_folder = os.path.join(OUTPUT_DIR, round_folder, stage, unique_folder)
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(os.path.join(new_folder, "teams"), exist_ok=True)
    backup_input_data_zip(new_folder)
    import glob
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    input_values = {}
    for csv_file in csv_files:
        try:
            with open(csv_file, "r") as f:
                content = f.read()
            input_values[os.path.basename(csv_file)] = {
                "size": len(content),
                "hash": compute_file_hash(csv_file),
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
             default_time_str = default_key.get("timestamp", "1970-01-01_00-00-00")
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
                    ["teams", "elo", "current_wins", "home_adv"],
                    load_and_prepare_data()
                ))
            )
        },
        {
            "desc": "Load remaining games",
            "func": lambda: context.update({
                "remaining_games": load_remaining_games(context["elo"], context["teams"])
            })
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
                                            context["home_adv"])
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
            "desc": "Export team results",
            "func": lambda: __import__("exporter").export_team_results(context["final_df"], context["teams"])
        },
        {
            "desc": "Generate overall graphs",
            "func": lambda: generate_graphs(context["final_df"], context["teams"])
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