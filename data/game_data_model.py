class BaseballGameData:
    def __init__(self, game_id, date, home_team_id, home_team, home_score, home_features, away_team_id, away_team, away_score, away_features):
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

        self.away_team_data = TeamData(game_id, date, away_team_id, away_team, away_features, away_score)
        self.home_team_data = TeamData(game_id, date, home_team_id, home_team, home_features, home_score)

class TeamData:
    def __init__(self, game_id, date, team_id, team_name, team_features, team_score):
        self.game_id = game_id
        self.date = date
        self.team_id = team_id
        self.team_name = team_name
        self.team_features = team_features
        self.team_score = team_score
    
    def get_training_data(self):
        return self.team_score, self.team_features
    
    def get_csv_data(self):
        return [self.game_id, self.date, self.team_id, self.team_name] + [self.team_score] + self.team_features
    

def csv_to_teamdata(row):
    game_id = row[0]
    date = row[1]
    team_id = row[2]
    team_name = row[3]
    team_score = row[4]
    team_features = row[5:]
    return TeamData(game_id, date, team_id, team_name, team_features, team_score)