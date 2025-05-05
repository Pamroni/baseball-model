import statsapi

FANGRAPHS_TEAM_ID = {
    "Arizona Diamondbacks": {'abbr': 'ARI', 'team_num': 15},
    "Atlanta Braves": {'abbr': 'ATL', 'team_num': 16},
    "Baltimore Orioles": {'abbr': 'BAL', 'team_num': 2},
    "Boston Red Sox": {'abbr': 'BOS', 'team_num': 3},
    "Chicago White Sox": {'abbr': 'CHW', 'team_num': 4},
    "Chicago Cubs": {'abbr': 'CHC', 'team_num': 17},
    "Cincinnati Reds": {'abbr': 'CIN', 'team_num': 18},
    "Cleveland Guardians": {'abbr': 'CLE', 'team_num': 5},
    "Colorado Rockies": {'abbr': 'COL', 'team_num': 19},
    "Detroit Tigers": {'abbr': 'DET', 'team_num': 6},
    "Houston Astros": {'abbr': 'HOU', 'team_num': 21},
    "Kansas City Royals": {'abbr': 'KCR', 'team_num': 7},
    "Los Angeles Angels": {'abbr': 'LAA', 'team_num': 1},
    "Los Angeles Dodgers": {'abbr': 'LAD', 'team_num': 22},
    "Miami Marlins": {'abbr': 'MIA', 'team_num': 20},
    "Milwaukee Brewers": {'abbr': 'MIL', 'team_num': 23},
    "Minnesota Twins": {'abbr': 'MIN', 'team_num': 8},
    "New York Yankees": {'abbr': 'NYY', 'team_num': 9},
    "New York Mets": {'abbr': 'NYM', 'team_num': 25},
    "Oakland Athletics": {'abbr': 'OAK', 'team_num': 10},
    "Philadelphia Phillies": {'abbr': 'PHI', 'team_num': 26},
    "Pittsburgh Pirates": {'abbr': 'PIT', 'team_num': 27},
    "San Diego Padres": {'abbr': 'SDP', 'team_num': 29},
    "San Francisco Giants": {'abbr': 'SFG', 'team_num': 30},
    "Seattle Mariners": {'abbr': 'SEA', 'team_num': 11},
    "St. Louis Cardinals": {'abbr': 'STL', 'team_num': 28},
    "Tampa Bay Rays": {'abbr': 'TBR', 'team_num': 12},
    "Texas Rangers": {'abbr': 'TEX', 'team_num': 13},
    "Toronto Blue Jays": {'abbr': 'TOR', 'team_num': 14},
    "Washington Nationals": {'abbr': 'WSN', 'team_num': 24},
    "Montreal Expos": {'abbr': 'MON', 'team_num': 24},
    "Cleveland Indians": {'abbr': 'CLE', 'team_num': 5},
    "Tampa Bay Devil Rays": {'abbr': 'TBD', 'team_num': 12},
    "Anaheim Angels": {'abbr': 'ANA', 'team_num': 1},
    "Florida Marlins": {'abbr': 'FLA', 'team_num': 20},
}

def get_team_id(team_name):
    """
    Get the Fangraphs team ID for a given MLB team name.
    """
    if team_name in FANGRAPHS_TEAM_ID:
        return FANGRAPHS_TEAM_ID[team_name]['team_num']
    else:
        raise ValueError(f"Team name '{team_name}' not found in Fangraphs team ID mapping.")


def convert_percentage_to_float(percentage_str):
    return float(percentage_str.strip("%")) / 100


def get_player_stats(mlb_player_id, team_df):
    data = statsapi.player_stat_data(mlb_player_id)
    full_name = f"{data['first_name']} {data['last_name']}"
    player_stats = team_df[team_df["Name"] == full_name]
    if len(player_stats) == 0:
        # Return 0s the size of the DF
        player_stats = team_df.iloc[0].copy()
        player_stats[:] = 0

    return player_stats.to_numpy().flatten()[1:]