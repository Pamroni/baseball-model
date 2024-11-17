# MLB Baseball model
Initially as a research project, this serves to predict winners of baseball games. All data comes directly from the official MLB stats API. As of development, it is the offseason so live predictions aren't supported

## Initial setup
Prior to working, it's strongly advise to use a Python virtual environment. All development took place on Python 3.10.12 on an Ubuntu server.

1. `python3 -m virtualenv .venv`
2. `source ./.venv/bin/activate` or for windows, `.\.venv\scripts\activate`
3. `pip install -r requirements.txt`

Of note, there's quite a few requirements. These are required to have properly working XGBoost and Torch based NN

## Generating Data
Features can be updated and modified in the respective files:
- `data/batting.py` for batting data
- `data/pitching.py` for pitching data

Curently, data is generated by running the following:
- `python -m data.generate`

Of note, this creates a significant number of API request to the MLB server and can take a while (16-17 hours per year of data). This stores the features in the `./csv_data/*` directory as `<year>_data.csv`

## Training and evaluation
The base `model.py` class is required to use the evaluate method. Evaluation is taking a set of data in a CSV (as a path) and predicts each team's runs for a game and chooses the team with the highest run. The model then returns a score of % accuracy.


### Neural Network Model
The neural network model is a linear regression network with ReLU activation layers. There is more work to be done to this, but the baseline model is performing exceptionaly well for 2024 after training on 2021, 2022, and 2023.

To train the neural network, all that needs to be done is run:
`python nn_train.py`. There are a considerable amount of args that can be seen by using `python nn_train.py --help`. This script will train and evaluate the network. Additionally, it will store the final model to `./trained_models/baseball_nn.th` but this can be modified by the --model_path arg. 


### XGBoost Model
TO COME


