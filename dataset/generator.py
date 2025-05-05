import concurrent.futures
import time
import pandas as pd
from tqdm import tqdm
from .mlb.utils import get_season_games
from .skeleton_dataset import Dataset
from .fangraphs.fangraphs_dataset import FangraphsDataset

COOLDOWN_TIME = 1
REQUESTS_ERROR_RETRY = 10

GAME_ID_CSV_INDEX = 0

def get_filename(year, prefix):
    return f"{prefix}_{year}.csv"

def get_csv_dataframe(file_name):
    try:
        df = pd.read_csv(file_name, header=None)
        return df
    except Exception:
        # Create empty file
        with open(file_name, 'w') as f:
            print(f"Created empty file {file_name}")
        return pd.DataFrame()
    
def write_to_csv(game_id, label, features, csv_file):
    csv_data = [game_id, label] + features
    with open(csv_file, 'a') as f:
        f.write(','.join(map(str, csv_data)) + '\n')

def process_year(year):
    dataset = FangraphsDataset()
    print(f"Generating data for {year}")
    # Get all games for a season
    start = time.time()
    season_game_ids = get_season_games(year, skip_games=15)
    print(f"Found {len(season_game_ids)} games in {time.time()-start} seconds for {year}")
    csv_file = get_filename(year, dataset.get_csv_file_prefix())
    # Load into Pandas to make sure we dont duplicate
    for game_id in tqdm(season_game_ids):
        retries = 0
        game_data_df = get_csv_dataframe(csv_file)
        while retries < REQUESTS_ERROR_RETRY:
            try:
                # Check if the game_id is in the CSV File
                if len(game_data_df) > 0 and game_id in game_data_df.iloc[:, GAME_ID_CSV_INDEX].values:
                    print(f"Game {game_id} already exists in {csv_file}")
                    break
            
                # Get the game data
                label, data = dataset.generate_csv_data(game_id)
                write_to_csv(game_id, label, data, csv_file)
                break
            except Exception as e:
                retries += 1
                print(f"Retrying {game_id} due to error: {e}")
                time.sleep(COOLDOWN_TIME * 4)
        if retries == REQUESTS_ERROR_RETRY:
            print(f"Failed to process game {game_id} after {REQUESTS_ERROR_RETRY} retries")

    print(f"Finished generating data for {year} in {time.time()-start} seconds")
    

if __name__ == "__main__":
    years = ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
    threaded = True
    if threaded:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_year, years)
    else:
        for year in years:
            process_year(year)