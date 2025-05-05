import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from .utils import convert_percentage_to_float

ADVANCED_PITCHING_COLUMNS = [
    "DELETE",
    "Name",
    "DELETE",
    "K/9",
    "BB/9",
    "K/BB",
    "HR/9",
    "DELETE",
    "K%",
    "BB%",
    "K-BB%",
    "DELETE",
    "AVG",
    "WHIP",
    "BABIP",
    "LOB%",
    "DELETE",
    "ERA-",
    "FIP-",
    "xFIP-",
    "DELETE",
    "ERA",
    "FIP",
    "E-F",
    "DELETE",
    "xFIP",
    "SIERA",
]
ADVANCED_PITCHING_PERCENTAGE_COLUMNS = ["K%", "BB%", "K-BB%", "LOB%"]


def get_pitching_season_stats(game_date, team, type="starter"):
    if type not in ["starter", "reliever"]:
        raise ValueError("Type must be either 'starter' or 'reliever'")
    type = type[0:3]
    season = game_date.split("-")[0]
    start_date = f"{season}-03-01"
    t_minus_1 = (pd.to_datetime(game_date) - pd.DateOffset(days=1)).date()
    hit_url = f"https://www.fangraphs.com/leaders/major-league?startdate={start_date}&enddate={t_minus_1}&season={season}&season1={season}&month=1000&ind=0&pageitems=200&team={team}&type=1&qual=1&stats={type}"

    team_data = []
    page = None
    while page is None:
        try:
            page = requests.get(hit_url, timeout=15)
        except Exception as e:
            print(f"Error fetching data, retrying after 5 seconds: {e}")
            time.sleep(5)
            continue
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("div", class_="table-scroll")
    rows = table.find("tbody").find_all("tr")

    for row in rows:
        team_cell = row.find("td", {"data-stat": "Name"})
        if team_cell:
            row_data = [cell.text.strip() for cell in row.find_all("td")]
            team_data.append(row_data)

    team_df = pd.DataFrame(team_data, columns=ADVANCED_PITCHING_COLUMNS)
    team_df = team_df.drop(columns=["DELETE"])
    for col in ADVANCED_PITCHING_PERCENTAGE_COLUMNS:
        team_df[col] = team_df[col].apply(convert_percentage_to_float)
    team_df.iloc[:, 1:] = team_df.iloc[:, 1:].apply(pd.to_numeric)
    return team_df


def get_last_x_starting_pitcher_stats(game_date, team, x_days_up_to=8, type="starter"):
    if type not in ["starter", "reliever"]:
        raise ValueError("Type must be either 'starter' or 'reliever'")
    type = type[0:3]
    season = game_date.split("-")[0]
    t_minus_1 = (pd.to_datetime(game_date) - pd.DateOffset(days=1)).date()
    x_back = (pd.to_datetime(game_date) - pd.DateOffset(days=x_days_up_to)).date()
    hit_url = f"https://www.fangraphs.com/leaders/major-league?startdate={x_back}&enddate={t_minus_1}&season={season}&season1={season}&month=1000&ind=0&pageitems=200&team={team}&type=1&qual=1&stats={type}"

    team_data = []
    page = None
    while page is None:
        try:
            page = requests.get(hit_url, timeout=15)
        except Exception as e:
            print(f"Error fetching data, retrying after 5 seconds: {e}")
            time.sleep(5)
            continue
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("div", class_="table-scroll")
    rows = table.find("tbody").find_all("tr")

    for row in rows:
        team_cell = row.find("td", {"data-stat": "Name"})
        if team_cell:
            row_data = [cell.text.strip() for cell in row.find_all("td")]
            team_data.append(row_data)

    team_df = pd.DataFrame(team_data, columns=ADVANCED_PITCHING_COLUMNS)
    team_df = team_df.drop(columns=["DELETE"])

    for col in ADVANCED_PITCHING_PERCENTAGE_COLUMNS:
        team_df[col] = team_df[col].apply(convert_percentage_to_float)
    team_df.iloc[:, 1:] = team_df.iloc[:, 1:].apply(pd.to_numeric)
    return team_df


def remove_bullpen_yesterday(bullpen_long_time, bullpen_yesterday):
    # remove all from bullpen_long_time that are in bullpen_yesterday
    return bullpen_long_time[~bullpen_long_time["Name"].isin(bullpen_yesterday["Name"])]


def get_pitcher_team_features(pitcher_df):
    # Drop the name column and average the results
    dropped = pitcher_df.drop(columns=["Name"])
    return dropped.mean(axis=0).to_numpy().flatten()
