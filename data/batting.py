import statsapi
from datetime import datetime, timedelta

BATTER_STATS = [
    "gamesPlayed",
    "flyOuts",
    "groundOuts",
    "airOuts",
    "runs",
    "doubles",
    "triples",
    "homeRuns",
    "strikeOuts",
    "baseOnBalls",
    "intentionalWalks",
    "hits",
    "hitByPitch",
    "avg",
    "atBats",
    "obp",
    "slg",
    "ops",
    "caughtStealing",
    "stolenBases",
    "stolenBasePercentage",
    "groundIntoDoublePlay",
    "groundIntoTriplePlay",
    "numberOfPitches",
    "plateAppearances",
    "totalBases",
    "rbi",
    "leftOnBase",
    "sacBunts",
    "sacFlies",
    "babip",
    "groundOutsToAirouts",
    "catchersInterference",
    "atBatsPerHomeRun",
]

ZERO_VALUES = ['.---', '-.--']
def get_batter_stats(player_id, game_date, year, last_x=5):
    season_start_date = f"01/01/{year}"
    end_date = datetime.strptime(game_date, "%m/%d/%Y")
    last_x_start_date = (end_date - timedelta(days=last_x)).strftime("%m/%d/%Y")

    player_season_stats = statsapi.player_stat_data(
        player_id, group="[hitting]", type=f"[byDateRange],startDate={season_start_date},endDate={game_date},currentTeam"
    )["stats"]
    player_last_x_dates_stats = statsapi.player_stat_data(
        player_id, group="[hitting]", type=f"[byDateRange],startDate={last_x_start_date},endDate={game_date},currentTeam"
    )["stats"]
    season = {}
    last_x_batting = {}
    
    if len(player_season_stats) > 0:
        season = player_season_stats[0]["stats"]
    
    if len(player_last_x_dates_stats) > 0:
        last_x_batting = player_last_x_dates_stats[0]["stats"]

    # Turn feature set into a numpy array
    features = []
    for val in BATTER_STATS:
        # First do the season stats:
        season_val = season.get(val, 0.0)
        if season_val in ZERO_VALUES:
            season_val = 0.0
        features.append(float(season_val))

    # Now do the last x games stats:
    for val in BATTER_STATS:
        last_x_val = last_x_batting.get(val, 0.0)
        if last_x_val in ZERO_VALUES:
            last_x_val = 0.0
        features.append(float(last_x_val))

    return features