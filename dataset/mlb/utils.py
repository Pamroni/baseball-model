import statsapi


def get_season_games(year, skip_games=None):
    game_ids = []
    months = [
        ("04", "30"),
        ("05", "31"),
        ("06", "30"),
        ("07", "31"),
        ("08", "31"),
        ("09", "30"),
    ]
    for month, end_day in months:
        start_date = "01"
        if month == "04" and skip_games is not None:
            start_date = str(skip_games)
            if len(start_date) == 1:
                start_date = "0" + start_date

        games = statsapi.schedule(
            start_date=f"{month}/{start_date}/{year}",
            end_date=f"{month}/{end_day}/{year}",
        )
        for game in games:
            game_id = game["game_id"]
            if (
                game["game_type"] == "R"
                and game["status"] == "Final"
                and game["away_score"] != game["home_score"]
            ):
                game_ids.append(game_id)

    return game_ids


def get_game_response(game_id):
    game_response = statsapi.get("game", {"gamePk": game_id})
    return game_response


def get_away_lineup(game_response):
    away_lineup = game_response["liveData"]["boxscore"]["teams"]["away"]["battingOrder"]
    away_starting_pitcher_id = game_response["liveData"]["boxscore"]["teams"]["away"][
        "pitchers"
    ]
    if len(away_starting_pitcher_id) > 0:
        away_starting_pitcher_id = away_starting_pitcher_id[0]
    else:
        away_starting_pitcher_id = None
    return away_lineup, away_starting_pitcher_id


def get_home_lineup(game_response):
    home_lineup = game_response["liveData"]["boxscore"]["teams"]["home"]["battingOrder"]
    home_starting_pitcher_id = game_response["liveData"]["boxscore"]["teams"]["home"][
        "pitchers"
    ]
    if len(home_starting_pitcher_id) > 0:
        home_starting_pitcher_id = home_starting_pitcher_id[0]
    else:
        home_starting_pitcher_id = None
    return home_lineup, home_starting_pitcher_id


def get_response_date(game_response):
    date = game_response["gameData"]["datetime"]["officialDate"]
    return date


def get_date(game_response):
    date = get_response_date(game_response)
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
