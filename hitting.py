import statsapi
import numpy as np

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

ZERO_VALUES = ['.---']

def get_batter_stats(player_id):
    player = statsapi.player_stat_data(
        player_id, group="[hitting]", type="[season,lastXGames],limit=5"
    )
    stats = player["stats"]
    season = None
    last_x_batting = None
    for stat in stats:
        if stat["group"] == "hitting" and stat["type"] == "season":
            season = stat["stats"]
        elif stat["group"] == "hitting" and stat["type"] == "lastXGames":
            last_x_batting = stat["stats"]

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