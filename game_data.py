import statsapi


def game_extract(schedule_game):
    away_id, away_name = schedule_game["away_id"], schedule_game["away_name"]
    home_id, home_name = schedule_game["home_id"], schedule_game["home_name"]

    venue_id = schedule_game["venue_id"]


if __name__ == "__main__":
    games = statsapi.schedule(team="117", start_date="04/01/2024")
    game_extract(games[0])
