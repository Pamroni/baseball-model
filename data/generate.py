import concurrent.futures
import time
import requests
import statsapi
from tqdm import tqdm

from .batting import get_batter_stats
from .pitching import get_pitcher_stats
from .baseball_data_model import (
    BaseballGameData,
    save_feature_names,
)
from .statsapi_utils import get_date, team_info, get_runs

COOLDOWN_TIME = 1
REQUESTS_ERROR_RETRY = 10

"""
Other ideas:
gameData["weather"] - use environmental data

Pitching matchups vs opp st pitcher
hydrate = 'stats(group=[hitting],type=[vsPlayer],opposingPlayerId={},season=2019,sportId=1)'.format(opponentId)


"""


def generate_data(game_id, year, innings=None, write_names=False) -> BaseballGameData:
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

    home_lineup = game_response["liveData"]["boxscore"]["teams"]["home"]["battingOrder"]
    home_starting_pitcher_id = game_response["liveData"]["boxscore"]["teams"]["home"][
        "pitchers"
    ][
        0
    ]  # Orioles

    # Get runs
    away_runs, home_runs = get_runs(game_response, innings)

    # Feature set: [[OPPONENT_STARTING_PITCHER],[ROSTER]]
    away_opponent_pitcher_stats, away_opponent_pitcher_feature_names = (
        get_pitcher_stats(home_starting_pitcher_id, query_date, year)
    )
    away_batting_stats = []
    away_feature_names = []
    for lineup_number, batter in enumerate(away_lineup):
        away_batter_stats, away_batter_feature_names = get_batter_stats(
            player_id=batter,
            game_date=query_date,
            year=year,
            pitcher_id=home_starting_pitcher_id,
        )
        for name in away_batter_feature_names:
            away_feature_names.append(f"{lineup_number+1}_{name}")

        away_batting_stats.extend(away_batter_stats)

    home_opponent_pitcher_stats, home_opponent_pitcher_features_names = (
        get_pitcher_stats(away_starting_pitcher_id, query_date, year)
    )
    home_batting_stats = []
    home_feature_names = []
    for lineup_number, batter in enumerate(home_lineup):
        home_batter_stats, home_batter_feature_names = get_batter_stats(
            player_id=batter,
            game_date=query_date,
            year=year,
            pitcher_id=away_starting_pitcher_id,
        )
        for name in home_batter_feature_names:
            home_feature_names.append(f"{lineup_number+1}_{name}")
        home_batting_stats.extend(home_batter_stats)

    # Assert that the features are the same
    assert home_feature_names == away_feature_names
    assert away_opponent_pitcher_feature_names == home_opponent_pitcher_features_names

    feature_names = [] + home_feature_names + home_opponent_pitcher_features_names
    if write_names:
        save_feature_names(feature_names)

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


def season_data(year, innings=None) -> list[BaseballGameData]:
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
                game_data.append(generate_data(game_id, year, innings))
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
    innings_runs = 5
    if innings_runs:
        filename = f"csv_data/{year}_data_{innings_runs}_innings.csv"
    else:
        filename = f"csv_data/{year}_data.csv"
    print(f"Generating data for {year}")
    game_data = season_data(year, innings_runs)
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
    years = ["2019", "2021", "2022", "2023", "2024"]
    threaded = True
    if threaded:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_year, years)
    else:
        for year in years:
            process_year(year)
