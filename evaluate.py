import torch
import os
import csv
from data.baseball_data_model import csv_to_teamdata, TeamData
from model import BaseballModel


class GameResultData:
    def __init__(self, data_path):
        self.games = {}

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found.")
        with open(data_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                team_data: TeamData = csv_to_teamdata(row)

                # check if game id is in self.games
                game_id = team_data.game_id
                team_name = team_data.team_name
                team_score = float(team_data.team_score)
                date = team_data.date

                team_features = team_data.team_features
                team_features = [float(x) for x in team_features]
                if game_id not in self.games:
                    self.games[game_id] = {team_name: (team_score, team_features, date)}
                else:
                    self.games[game_id][team_name] = (team_score, team_features, date)

    def get_games(self):
        return self.games


class TeamScore:
    def __init__(self, game_id, date, team_name, score, predicted_score):
        self.game_id = game_id
        self.date = date
        self.team_name = team_name
        self.score = score
        self.predicted_score = predicted_score

    def to_csv(self):
        return f"{self.game_id},{self.date},{self.team_name},{self.score},{self.predicted_score}"

    @staticmethod
    def from_csv(csv_str):
        game_id, date, team_name, score, predicted_score = csv_str.split(",")
        return TeamScore(game_id, date, team_name, score, predicted_score)


def check_winner(results: list[TeamScore], verbose=False):
    assert len(results) == 2
    assert results[0].game_id == results[1].game_id
    assert results[0].date == results[1].date
    if results[0].predicted_score > results[1].predicted_score:
        predicted_winner = 0
    else:
        predicted_winner = 1

    if results[0].score > results[1].score:
        actual_winner = 0
    else:
        actual_winner = 1

    correct = actual_winner == predicted_winner
    predicted_loser = 0 if predicted_winner == 1 else 1
    actual_loser = 0 if actual_winner == 1 else 1
    if not correct and verbose:
        print(
            f"MISS for {results[0].date}: \n\
            Predicted Winner=[{results[predicted_winner].team_name}, Score:[{results[predicted_winner].predicted_score}vs{results[predicted_loser].predicted_score}]]\n\
            Actual Winner=[{results[actual_winner].team_name}, Score: [{results[actual_winner].score}vs{results[actual_loser].score}]]\n\n"
        )
    return correct


def evaluate(model: BaseballModel, data_path: str):
    if type(data_path) == list:
        data_path = data_path[0]
    correct = 0
    game_data = GameResultData(data_path).get_games()
    model.prepare_eval()
    with torch.no_grad():
        for game_id, team_results in game_data.items():
            results = []
            for team in team_results.keys():
                score, features, date = team_results[team]
                predicted_score = model.predict(features)
                results.append(TeamScore(game_id, date, team, score, predicted_score))

            correct += check_winner(results)
    print(f"Accuracy: {correct/len(game_data)}")
