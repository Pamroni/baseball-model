import pandas as pd

from ..skeleton_dataset import Dataset
from ..mlb.utils import (
    get_game_response,
    get_home_lineup,
    get_away_lineup,
    team_info,
    get_runs,
    get_response_date,
)
from .utils import get_team_id, get_player_stats
from .batting import (
    get_batting_season_stats,
    get_last_x_batter_stats,
    get_two_year_stats,
)
from .pitching import (
    get_last_x_starting_pitcher_stats,
    get_pitching_season_stats,
    remove_bullpen_yesterday,
    get_pitcher_team_features,
)

SUPPORTED_YEARS = [
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
]


class FangraphsDatasetReduced(Dataset):
    def __init__(self):
        super().__init__()
        self.csv_file_prefix = "./csv_data/fangraphs_advanced_reduced"
        self.batter_days = 5
        self.starting_pitcher_days = 14

    def get_feature_size(self):
        features = self.generate_features(661527)
        return len(features)

    def get_csv_file_prefix(self) -> str:
        return self.csv_file_prefix

    def load_training_data(self, year: str, no_zeros = False):
        file_path = f"{self.csv_file_prefix}_{year}.csv"
        try:
            df = pd.read_csv(file_path, header=None)
            initial_rows = len(df)
            df = df.dropna()
            final_rows = len(df)
            if initial_rows != final_rows:
                print(
                    f"Dropped {initial_rows - final_rows} rows with NaN values from {file_path}"
                )
            features = df.iloc[:, 2:].values.tolist()
            labels = df.iloc[:, 1].values.tolist()

            if no_zeros:
                # add 1e-9 to any zero value
                features = [
                    [feature + 1e-9 if feature == 0 else feature for feature in row]
                    for row in features
                ]
            return features, labels
        except FileNotFoundError as e:
            print(f"File {file_path} not found.")
            raise e

    def generate_csv_data(self, game_id):
        # Placeholder for actual implementation
        label, features = self.generate_training_data(game_id)
        return label, features

    def generate_training_data(self, game_id):
        game_response = get_game_response(game_id)
        features = self.generate_features(game_id)
        label = self.get_run_differential(game_response)

        return label, features

    def generate_features(self, game_id):
        # Placeholder for actual implementation
        features = []
        game_response = get_game_response(game_id)
        game_date = get_response_date(game_response)
        features = []
        away_team, _, home_team, _ = team_info(game_response)
        home_lineup, home_starting_pitcher = get_home_lineup(game_response)
        away_lineup, away_starting_pitcher = get_away_lineup(game_response)
        home_team_id = get_team_id(home_team)
        away_team_id = get_team_id(away_team)

        home_batter_features = self.get_batter_features(
            game_date, home_team_id, home_lineup
        )
        home_starting_pitcher_features = self.get_starting_pitcher_features(
            game_date, home_team_id, home_starting_pitcher
        )
        home_bullpen_features = self.get_bullpen_features(game_date, home_team_id)

        away_batter_features = self.get_batter_features(
            game_date, away_team_id, away_lineup
        )
        away_starting_pitcher_features = self.get_starting_pitcher_features(
            game_date, away_team_id, away_starting_pitcher
        )
        away_bullpen_features = self.get_bullpen_features(game_date, away_team_id)

        features.extend(home_batter_features)
        features.extend(home_starting_pitcher_features)
        features.extend(home_bullpen_features)
        features.extend(away_batter_features)
        features.extend(away_starting_pitcher_features)
        features.extend(away_bullpen_features)

        return features

    def get_run_differential(self, game_response):
        away_runs, home_runs = get_runs(game_response)
        return int(home_runs) - int(away_runs)

    def get_batter_features(self, game_date, team_id, team_lineup):
        features = []
        team_batter_season = get_batting_season_stats(game_date, team_id)
        team_batter_last_x = get_last_x_batter_stats(
            game_date, team_id, self.batter_days
        )

        for player_id in team_lineup:
            player_season = get_player_stats(player_id, team_batter_season)
            player_last_x = get_player_stats(player_id, team_batter_last_x)
            if player_season is not None and player_last_x is not None:
                features.extend(player_season)
                features.extend(player_last_x)
            else:
                raise ValueError(f"Player {player_id} not found in stats data")
        return features

    def get_starting_pitcher_features(self, game_date, team_id, starting_pitcher_id):
        features = []
        starters_biweekly = get_last_x_starting_pitcher_stats(
            game_date, team_id, self.starting_pitcher_days, "starter"
        )
        starters_season = get_pitching_season_stats(game_date, team_id, "starter")
        player_biweekly = get_player_stats(starting_pitcher_id, starters_biweekly)
        player_season = get_player_stats(starting_pitcher_id, starters_season)
        if player_biweekly is None or player_season is None:
            raise ValueError(f"Player {starting_pitcher_id} not found in stats data")

        features.extend(player_biweekly)
        features.extend(player_season)
        return features

    def get_bullpen_features(self, game_date, team_id):
        features = []
        overall_bullpen_month = get_last_x_starting_pitcher_stats(
            game_date, team_id, self.starting_pitcher_days, "reliever"
        )
        bullpen_last_week = get_last_x_starting_pitcher_stats(
            game_date, team_id, 7, "reliever"
        )
        bullpen_yesterday = get_last_x_starting_pitcher_stats(
            game_date, team_id, 1, "reliever"
        )
        bullpen_fresh_week = remove_bullpen_yesterday(
            bullpen_last_week, bullpen_yesterday
        )

        bullpen_month_features = get_pitcher_team_features(overall_bullpen_month)
        bullpen_fresh_week_features = get_pitcher_team_features(bullpen_fresh_week)
        if bullpen_month_features is None or bullpen_fresh_week_features is None:
            raise ValueError(f"Player {team_id} not found in stats data")

        features.extend(bullpen_month_features)
        features.extend(bullpen_fresh_week_features)
        return features
