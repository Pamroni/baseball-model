import csv
import os 
class BaseballGameData:
    def __init__(
        self,
        game_id,
        date,
        home_team_id,
        home_team,
        home_score,
        home_features,
        away_team_id,
        away_team,
        away_score,
        away_features,
    ):
        self.game_id = game_id
        self.date = date
        self.home_team_id = home_team_id
        self.home_team = home_team
        self.home_score = home_score
        self.home_features = home_features
        self.away_team_id = away_team_id
        self.away_team = away_team
        self.away_score = away_score
        self.away_features = away_features

        self.away_team_data = TeamData(
            game_id, date, away_team_id, away_team, away_features, away_score
        )
        self.home_team_data = TeamData(
            game_id, date, home_team_id, home_team, home_features, home_score
        )


class TeamData:
    def __init__(self, game_id, date, team_id, team_name, team_features, team_score):
        self.game_id = game_id
        self.date = date
        self.team_id = team_id
        self.team_name = team_name
        self.team_features = team_features
        self.team_score = team_score

    def get_training_data(self):
        score = float(self.team_score)
        features = [float(x) for x in self.team_features]
        return score, features

    def get_csv_data(self):
        vals = (
            [self.game_id, self.date, self.team_id, self.team_name]
            + [self.team_score]
            + self.team_features
        )
        # Convert all to str
        for i in range(len(vals)):
            vals[i] = str(vals[i])
        return vals

# Take a list of feature names and save them
def save_feature_names(feature_names):
    feature_len = len(feature_names)
    # Save as feature_len_names.txt
    file_name = f"./csv_data/feature_names_{feature_len}.txt"
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(",".join(feature_names))

def get_feature_names(feature_list):
    feature_len = len(feature_list)
    file_name = f"./csv_data/feature_names_{feature_len}.txt"
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            return row


def csv_to_teamdata(row):
    game_id = row[0]
    date = row[1]
    team_id = row[2]
    team_name = row[3]
    team_score = row[4]
    team_features = row[5:]
    return TeamData(game_id, date, team_id, team_name, team_features, team_score)
