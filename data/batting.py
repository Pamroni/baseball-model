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

MATCHUP_STATS = [
    "gamesPlayed",
    "groundOuts",
    "airOuts",
    "doubles",
    "triples",
    "homeRuns",
    "strikeOuts",
    "baseOnBalls",
    "intentionalWalks",
    "hits",
    "hitByPitch",
    "atBats",
    "groundIntoDoublePlay",
    "groundIntoTriplePlay",
    "numberOfPitches",
    "plateAppearances",
    "totalBases",
    "rbi",
    "leftOnBase",
    "sacBunts",
    "sacFlies",
    "catchersInterference",
]

ZERO_VALUES = [".---", "-.--", "-"]


def get_matchup_stats(batter_id, pitcher_id, season):
    game_response = statsapi.player_stat_data(
        personId=batter_id,
        group="[hitting]",
        type=f"[vsPlayer],opposingPlayerId={pitcher_id}",
    )

    versus_stats = [0 for _ in MATCHUP_STATS]
    for stat in game_response["stats"]:
        if stat["type"] == "vsPlayer" and int(stat["season"]) < int(season):
            historic_stat = stat["stats"]
            for i, key in enumerate(MATCHUP_STATS):
                stat_val = float(historic_stat[key])
                versus_stats[i] += stat_val

    return versus_stats


def get_batter_stats(player_id, game_date, year, pitcher_id, last_x=5):
    season_start_date = f"01/01/{year}"
    t_minus_1_game_date = datetime.strptime(game_date, "%m/%d/%Y") - timedelta(days=1)
    last_x_start_date = (t_minus_1_game_date - timedelta(days=last_x)).strftime(
        "%m/%d/%Y"
    )
    t_minus_1_game_date = t_minus_1_game_date.strftime("%m/%d/%Y")

    player_season_stats = statsapi.player_stat_data(
        player_id,
        group="[hitting]",
        type=f"[byDateRange],startDate={season_start_date},endDate={t_minus_1_game_date}",
    )["stats"]
    player_last_x_dates_stats = statsapi.player_stat_data(
        player_id,
        group="[hitting]",
        type=f"[byDateRange],startDate={last_x_start_date},endDate={t_minus_1_game_date}",
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

    # matchup stats
    matchup_stats = get_matchup_stats(player_id, pitcher_id, year)
    features.extend(matchup_stats)

    # Get feature names
    feature_names = []
    prefix_list = ["season", f"last_{last_x}_days"]
    for prefix in prefix_list:
        for key in BATTER_STATS:
            name = f"batter_{prefix}_{key}"
            feature_names.append(name)

    # matchup feature names
    for key in MATCHUP_STATS:
        name = f"batter_matchup_{key}"
        feature_names.append(name)

    return features, feature_names
