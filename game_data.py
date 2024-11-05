import statsapi


def get_away_team_lineup(game_id):
    # This returns the lineup for the away team in the game
    statsapi.lineup(game_id, "away")

def get_games():
    # This returns the games for the 2024 season - each one has a game_id flag
    # Filter on status: Final ?
    len(statsapi.schedule(start_date="04/01/2024", end_date="09/30/2024"))


def game_extract(schedule_game):
    away_id, away_name = schedule_game["away_id"], schedule_game["away_name"]
    home_id, home_name = schedule_game["home_id"], schedule_game["home_name"]

    venue_id = schedule_game["venue_id"]


if __name__ == "__main__":
    games = statsapi.schedule(team="117", start_date="04/01/2024")
    game_extract(games[0])
