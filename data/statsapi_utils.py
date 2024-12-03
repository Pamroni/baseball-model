def get_date(game_response):
    date = game_response["gameData"]["datetime"]["officialDate"]
    # MM/DD/YYYY
    query_date = f"{date[5:7]}/{date[8:10]}/{date[0:4]}"
    time = f'{game_response["gameData"]["datetime"]["time"]}{game_response["gameData"]["datetime"]["ampm"]}'
    return query_date, f"{date} {time}"


def team_info(game_response):
    away_team = game_response["gameData"]["teams"]["away"]["name"]
    away_team_id = game_response["gameData"]["teams"]["away"]["id"]
    home_team = game_response["gameData"]["teams"]["home"]["name"]
    home_team_id = game_response["gameData"]["teams"]["home"]["id"]
    return away_team, away_team_id, home_team, home_team_id


def get_runs(game_response, innings=None):
    if innings:
        away_runs = 0
        home_runs = 0
        for inning in game_response["liveData"]["linescore"]["innings"]:
            if inning["num"] <= innings:
                away_runs += inning["away"]["runs"]
                home_runs += inning["home"]["runs"]
    else:
        away_runs = game_response["liveData"]["linescore"]["teams"]["away"]["runs"]
        home_runs = game_response["liveData"]["linescore"]["teams"]["home"]["runs"]

    return away_runs, home_runs
