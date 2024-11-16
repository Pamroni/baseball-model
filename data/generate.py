import os
import concurrent.futures
import time
import requests
import statsapi
from tqdm import tqdm

from .batting import get_batter_stats
from .pitching import get_pitcher_stats
from .baseball_data_model import BaseballGameData, TeamData, csv_to_teamdata
from .statsapi_utils import get_date, team_info

COOLDOWN_TIME = 1
REQUESTS_ERROR_RETRY = 10

"""
Other ideas:
gameData["weather"] - use environmental data
"""


def generate_data(game_id, year) -> BaseballGameData:
    # Get game details
    game_response = statsapi.get("game", {"gamePk": game_id})

    query_date, display_date = get_date(game_response)  # string of datetime
    away_team, away_team_id, home_team, home_team_id = team_info(game_response)

    # Extract lineup information
    away_lineup = game_response["liveData"]["boxscore"]["teams"]["away"]["battingOrder"]
    away_starting_pitcher_id = game_response["liveData"]["boxscore"]["teams"]["away"][
        "pitchers"
    ][
        0
    ]  # Astros
    away_runs = game_response["liveData"]["linescore"]["teams"]["away"]["runs"]

    home_lineup = game_response["liveData"]["boxscore"]["teams"]["home"]["battingOrder"]
    home_starting_pitcher_id = game_response["liveData"]["boxscore"]["teams"]["home"][
        "pitchers"
    ][
        0
    ]  # Orioles
    home_runs = game_response["liveData"]["linescore"]["teams"]["home"]["runs"]

    # Feature set: [[OPPONENT_STARTING_PITCHER],[ROSTER]]
    away_opponent_pitcher_stats = get_pitcher_stats(
        home_starting_pitcher_id, query_date, year
    )
    away_batting_stats = []
    for batter in away_lineup:
        away_batting_stats.extend(get_batter_stats(batter, query_date, year))

    home_opponent_pitcher_stats = get_pitcher_stats(
        away_starting_pitcher_id, query_date, year
    )
    home_batting_stats = []
    for batter in home_lineup:
        home_batting_stats.extend(get_batter_stats(batter, query_date, year))

    home_features = [] + home_batting_stats + home_opponent_pitcher_stats
    away_features = [] + away_batting_stats + away_opponent_pitcher_stats

    return BaseballGameData(
        game_id=game_id,
        date=display_date,
        home_team_id=home_team_id,
        home_team=home_team,
        home_score=home_runs,
        home_features=home_features,
        away_team_id=away_team_id,
        away_team=away_team,
        away_score=away_runs,
        away_features=away_features,
    )


def season_data(year) -> list[BaseballGameData]:
    # Get all games for a season
    game_data: list[BaseballGameData] = []
    game_ids = []
    start = time.time()
    months = [
        ("04", "30"),
        ("05", "31"),
        ("06", "30"),
        ("07", "31"),
        ("08", "31"),
        ("09", "30"),
    ]
    for month, end_day in months:
        games = statsapi.schedule(
            start_date=f"{month}/01/{year}", end_date=f"{month}/{end_day}/{year}"
        )
        for game in games:
            game_ids.append(game["game_id"])
        del games
    print(f"Found {len(game_ids)} games in {time.time()-start} seconds")
    time.sleep(COOLDOWN_TIME)
    start = time.time()
    for game_id in tqdm(game_ids):
        retries = 0
        while retries < REQUESTS_ERROR_RETRY:
            try:
                game_data.append(generate_data(game_id, year))
                break
            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"Retrying {game_id} due to requests error: {e}")
                time.sleep(COOLDOWN_TIME * 4)
            except Exception as e:
                retries += 1
                print(f"Retrying {game_id} due to unknown error: {e}")
                time.sleep(COOLDOWN_TIME * 4)
        time.sleep(COOLDOWN_TIME)
    print(
        f"Generated {len(game_data)} games worth of data in {time.time()-start} seconds"
    )
    return game_data


def write_to_csv(game_data: list[BaseballGameData], filename):
    with open(filename, "a") as f:
        for game in game_data:
            f.write(",".join(game.away_team_data.get_csv_data()) + "\n")
            f.write(",".join(game.home_team_data.get_csv_data()) + "\n")


def process_year(year):
    filename = f"csv_data/{year}_data.csv"
    print(f"Generating data for {year}")
    game_data = season_data(year)
    print(f"Generated {len(game_data)} games worth of data, writing to {filename}")
    write_to_csv(game_data, filename)


"""
2021 missed values:
Retrying 632457 due to unknown error: list index out of range
Retrying 633468 due to unknown error: could not convert string to float: '-'

2022 missed values:
Retrying 707079 due to unknown error: list index out of range
Retrying 706920 due to unknown error: list index out of range
Retrying 706953 due to unknown error: list index out of range

2023 missed values:
none 

2024 missed values:
Retrying 746577 due to unknown error: list index out of range

"""
if __name__ == "__main__":
    years = ["2022", "2023", "2024"]
    for year in years:
        process_year(year)
        time.sleep(COOLDOWN_TIME * 10)
