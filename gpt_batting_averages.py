# lineup_batting_averages.py

from pybaseball import batting_stats, playerid_reverse_lookup
import requests
import pandas as pd


def get_game_lineup(game_pk):
    """
    Fetches the active lineup for a given MLB game using the MLB Stats API.
    """
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    response = requests.get(url)
    data = response.json()
    lineups = {}

    # Get home team lineup
    home_players = data["teams"]["home"]["players"]
    home_lineup = []
    for player_id, player_info in home_players.items():
        if (
            player_info["stats"].get("batting")
            and player_info["position"]["code"] != "1"
        ):
            player_name = player_info["person"]["fullName"]
            mlb_player_id = player_info["person"]["id"]
            home_lineup.append({"name": player_name, "mlb_id": mlb_player_id})
    lineups["home"] = home_lineup

    # Get away team lineup
    away_players = data["teams"]["away"]["players"]
    away_lineup = []
    for player_id, player_info in away_players.items():
        if (
            player_info["stats"].get("batting")
            and player_info["position"]["code"] != "1"
        ):
            player_name = player_info["person"]["fullName"]
            mlb_player_id = player_info["person"]["id"]
            away_lineup.append({"name": player_name, "mlb_id": mlb_player_id})
    lineups["away"] = away_lineup

    return lineups


def get_player_batting_average_by_id(player_id_mlb, stats):
    """
    Retrieves the batting average for a player using their MLBAM ID.
    """
    # Convert MLBAM ID to FanGraphs ID used by pybaseball
    player_ids = playerid_reverse_lookup([player_id_mlb], key_type="mlbam")
    if not player_ids.empty:
        fg_id = player_ids.iloc[0]["key_fangraphs"]
        player_stats = stats[stats["playerid"] == fg_id]
        if not player_stats.empty:
            batting_avg = player_stats.iloc[0]["AVG"]
            return batting_avg
    return None


def get_lineup_batting_averages(game_pk):
    """
    Combines the lineup data with batting averages.
    """
    lineup = get_game_lineup(game_pk)
    lineup_batting_avgs = {}
    current_year = pd.Timestamp.now().year
    stats = batting_stats(current_year)

    for team in ["home", "away"]:
        lineup_batting_avgs[team] = []
        for player in lineup[team]:
            avg = get_player_batting_average_by_id(player["mlb_id"], stats)
            lineup_batting_avgs[team].append({"Player": player["name"], "AVG": avg})
    return lineup_batting_avgs


def get_today_games():
    """
    Fetches today's MLB games.
    """
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule/games/?sportId=1&date={today}"
    response = requests.get(url)
    data = response.json()
    games = []

    for date_info in data["dates"]:
        for game in date_info["games"]:
            game_info = {
                "game_pk": game["gamePk"],
                "home_team": game["teams"]["home"]["team"]["name"],
                "away_team": game["teams"]["away"]["team"]["name"],
            }
            games.append(game_info)
    return games


def main():
    # Fetch today's games
    games = get_today_games()
    if not games:
        print("No games found for today.")
        return

    # List available games
    print("Today's MLB Games:")
    for idx, game in enumerate(games):
        print(
            f"{idx + 1}. Game PK: {game['game_pk']}, {game['away_team']} at {game['home_team']}"
        )

    # Select a game
    game_selection = (
        int(input("Select a game by entering the corresponding number: ")) - 1
    )
    if game_selection < 0 or game_selection >= len(games):
        print("Invalid selection.")
        return

    selected_game_pk = games[game_selection]["game_pk"]

    # Get lineup batting averages
    lineup_batting_avgs = get_lineup_batting_averages(selected_game_pk)

    # Display the results
    for team in ["home", "away"]:
        team_name = games[game_selection][f"{team}_team"]
        print(f"\n{team_name} Lineup Batting Averages:")
        for player_info in lineup_batting_avgs[team]:
            avg = player_info["AVG"]
            avg_display = f"{avg:.3f}" if avg is not None else "N/A"
            print(f"{player_info['Player']}: {avg_display}")


if __name__ == "__main__":
    main()
