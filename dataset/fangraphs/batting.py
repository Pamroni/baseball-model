import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from .utils import convert_percentage_to_float

ADVANCED_HITTING_COLUMNS = [
    "DELETE",
    "Name",
    "DELETE",
    "PA",
    "BB%",
    "K%",
    "BB/K",
    "DELETE",
    "AVG",
    "OBP",
    "SLG",
    "OPS",
    "DELETE",
    "ISO",
    "Spd",
    "BABIP",
    "DELETE",
    "DELETE",
    "DELETE",
    "DELETE",
    "wSB",
    "DELETE",
    "wRC",
    "wRAA",
    "wOBA",
    "wRC+",
]
ADVACNED_HITTING_PERCENT_COLUMNS = ["BB%", "K%"]


def get_batting_season_stats(game_date, team):
    season = game_date.split("-")[0]
    start_date = f"{season}-03-01"
    t_minus_1 = (pd.to_datetime(game_date) - pd.DateOffset(days=1)).date()
    hit_url = f"https://www.fangraphs.com/leaders/major-league?startdate={start_date}&enddate={t_minus_1}&season={season}&season1={season}&month=1000&ind=0&pageitems=200&team={team}&type=1&qual=1"

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

    team_df = pd.DataFrame(team_data, columns=ADVANCED_HITTING_COLUMNS)
    team_df = team_df.drop(columns=["DELETE"])

    for col in ADVACNED_HITTING_PERCENT_COLUMNS:
        team_df[col] = team_df[col].apply(convert_percentage_to_float)
    team_df.iloc[:, 1:] = team_df.iloc[:, 1:].apply(pd.to_numeric)
    return team_df


def get_two_year_stats(game_date, team):
    start_date = (pd.to_datetime(game_date) - pd.DateOffset(days=730)).date()
    t_minus_1 = (pd.to_datetime(game_date) - pd.DateOffset(days=1)).date()
    hit_url = f"https://www.fangraphs.com/leaders/major-league?startdate={start_date}&enddate={t_minus_1}&month=1000&ind=0&pageitems=200&team={team}&type=1&qual=1"

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

    team_df = pd.DataFrame(team_data, columns=ADVANCED_HITTING_COLUMNS)
    team_df = team_df.drop(columns=["DELETE"])

    for col in ADVACNED_HITTING_PERCENT_COLUMNS:
        team_df[col] = team_df[col].apply(convert_percentage_to_float)
    team_df.iloc[:, 1:] = team_df.iloc[:, 1:].apply(pd.to_numeric)
    return team_df


def get_last_x_batter_stats(game_date, team, x_days_up_to=8):
    season = game_date.split("-")[0]
    t_minus_1 = (pd.to_datetime(game_date) - pd.DateOffset(days=1)).date()
    week_back = (pd.to_datetime(game_date) - pd.DateOffset(days=x_days_up_to)).date()
    hit_url = f"https://www.fangraphs.com/leaders/major-league?startdate={week_back}&enddate={t_minus_1}&season={season}&season1={season}&month=1000&ind=0&pageitems=200&team={team}&type=1&qual=1"

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

    team_df = pd.DataFrame(team_data, columns=ADVANCED_HITTING_COLUMNS)
    team_df = team_df.drop(columns=["DELETE"])

    for col in ADVACNED_HITTING_PERCENT_COLUMNS:
        team_df[col] = team_df[col].apply(convert_percentage_to_float)
    team_df.iloc[:, 1:] = team_df.iloc[:, 1:].apply(pd.to_numeric)
    return team_df
