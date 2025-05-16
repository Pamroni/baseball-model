import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import dump, load
import warnings
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='baseball_model.log'
)

warnings.filterwarnings("ignore")

# Create output directory
os.makedirs("output", exist_ok=True)


def load_baseball_data(file_path):
    """Load and preprocess baseball statistics data"""
    print(f"Loading data from {file_path}...")
    logging.info(f"Loading data from {file_path}")
    start_time = time.time()

    # Load data
    df = pd.read_csv(file_path)

    # Data info
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    logging.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Handle unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # Convert date to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logging.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Check for new metrics (OPS, ISO, FIP)
    new_metrics = [
        "OPS",
        "OPS/5 Players",
        "OPS/Week",
        "ISO",
        "ISO/5 Players",
        "ISO/Week",
        "FIP",
    ]
    existing_metrics = [metric for metric in new_metrics if metric in df.columns]

    print(f"Found {len(existing_metrics)} new metrics: {', '.join(existing_metrics)}")
    logging.info(f"Found {len(existing_metrics)} new metrics: {', '.join(existing_metrics)}")

    # Add OPS if not present
    if "OPS" not in df.columns and all(col in df.columns for col in ["OBP", "SLG"]):
        df["OPS"] = df["OBP"] + df["SLG"]
        print("Added OPS metric (OBP + SLG)")
        logging.info("Added OPS metric (OBP + SLG)")

    # Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_after = df.isnull().sum().sum()

    if missing_before > 0:
        print(f"Removed {missing_before - missing_after} missing values")
        logging.info(f"Removed {missing_before - missing_after} missing values")

    # Sort by date to maintain chronological order
    if "Date" in df.columns:
        df = df.sort_values("Date")

    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    
    return df


def calculate_feature_interactions(df):
    """Calculate feature interactions that might be useful for prediction"""
    print("Calculating feature interactions...")
    logging.info("Calculating feature interactions")
    start_time = time.time()

    # Calculate offensive power metrics
    if all(col in df.columns for col in ["OBP", "SLG"]):
        df["OPS"] = df["OBP"] + df["SLG"]

    if all(col in df.columns for col in ["OPS", "WRC+"]):
        df["OffensivePower"] = df["OPS"] * df["WRC+"] / 100

    # Calculate pitching effectiveness
    if all(col in df.columns for col in ["ERA", "FIP"]):
        df["PitchingEffectiveness"] = (df["FIP"] + df["ERA"]) / 2
    elif "ERA" in df.columns:
        df["PitchingEffectiveness"] = df["ERA"]

    # Calculate batting discipline
    if all(col in df.columns for col in ["K Percentage", "BB Percentage"]):
        df["BattingDiscipline"] = df["BB Percentage"] / (df["K Percentage"] + 0.001)

    # Calculate pitcher control
    if all(col in df.columns for col in ["Opposing K/9", "Opposing BB/9"]):
        df["PitcherControl"] = df["Opposing K/9"] / (df["Opposing BB/9"] + 0.001)

    # Calculate efficiency metrics
    if all(col in df.columns for col in ["RBIs", "Total Runs"]):
        df["RunEfficiency"] = df["RBIs"] / (df["Total Runs"] + 0.001)

    # Calculate OPS vs ERA differential (hitting vs pitching strength)
    if all(col in df.columns for col in ["OPS", "ERA"]):
        df["OPS_ERA_Differential"] = df["OPS"] - (df["ERA"] / 10)  # Scale adjustment

    # Calculate combined weekly performance indicator
    if all(col in df.columns for col in ["OPS/Week", "ERA/Week"]):
        df["WeeklyPerformance"] = df["OPS/Week"] - (
            df["ERA/Week"] / 10
        )  # Scale adjustment

    # Calculate difference between actual and 5-player metrics
    if all(col in df.columns for col in ["OPS", "OPS/5 Players"]):
        df["OPS_StarterDiff"] = df["OPS"] - df["OPS/5 Players"]

    # Calculate the relative hitting strength
    if all(col in df.columns for col in ["AVG", "OBP", "SLG"]):
        df["HittingStrength"] = (df["AVG"] * 0.25) + (df["OBP"] * 0.35) + (df["SLG"] * 0.4)

    # Calculate the relative pitching strength
    if all(col in df.columns for col in ["ERA", "Opposing K/9", "Opposing BB/9"]):
        df["PitchingStrength"] = df["Opposing K/9"] - df["Opposing BB/9"] - (df["ERA"] / 5)
        
    # Calculate team net WAR
    if all(col in df.columns for col in ["WAR", "Opposing War"]):
        df["NetWAR"] = df["WAR"] - df["Opposing War"]
    
    # Calculate short-term momentum metrics
    if all(col in df.columns for col in ["WRC+", "WRC+/Week"]):
        df["OffensiveMomentum"] = df["WRC+/Week"] - df["WRC+"]
        
    if all(col in df.columns for col in ["ERA", "ERA/Week"]):
        df["PitchingMomentum"] = df["ERA"] - df["ERA/Week"]  # Reversed because lower ERA is better

    # Print newly created features
    new_features = [
        "OPS",
        "OffensivePower",
        "PitchingEffectiveness",
        "BattingDiscipline",
        "PitcherControl",
        "RunEfficiency",
        "OPS_ERA_Differential",
        "WeeklyPerformance",
        "OPS_StarterDiff",
        "HittingStrength",
        "PitchingStrength",
        "NetWAR",
        "OffensiveMomentum",
        "PitchingMomentum"
    ]

    created_features = [feat for feat in new_features if feat in df.columns]
    print(f"Created {len(created_features)} interaction features")
    logging.info(f"Created {len(created_features)} interaction features: {', '.join(created_features)}")
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    
    return df


def prepare_features_and_target(df, target_col="Win?"):
    """Prepare features and target variable for modeling"""
    print("Preparing features and target variable...")
    logging.info("Preparing features and target variable")
    start_time = time.time()

    # Calculate interaction features
    df = calculate_feature_interactions(df)

    # Team columns to keep separately
    team_cols = ["Offensive Team", "Defensive Team"]

    # Extract teams if both columns exist
    if all(col in df.columns for col in team_cols):
        teams = df[team_cols].copy()
        if "Date" in df.columns:
            teams["Date"] = df["Date"]
    else:
        teams = None
        team_cols = [col for col in team_cols if col in df.columns]

    # Define columns to exclude from features
    # VERY IMPORTANT: Explicitly exclude "Runs Scored" to prevent data leakage
    exclude_cols = team_cols + ["Date", target_col, "Runs Scored"]
    exclude_cols = [col for col in exclude_cols if col in df.columns]

    # Select numeric features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    # Target variable
    if target_col in df.columns:
        y = df[target_col].copy()
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # Data integrity check: ensure no leakage of game outcome
    if 'Runs Scored' in X.columns:
        print("WARNING: 'Runs Scored' is in the feature set - this will cause data leakage!")
        logging.warning("'Runs Scored' found in feature set - removing to prevent data leakage")
        X = X.drop('Runs Scored', axis=1)
    
    print(f"Prepared {X.shape[1]} features for modeling")
    logging.info(f"Prepared {X.shape[1]} features for modeling")

    # Create correlation with target
    corr_with_target = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Correlation": [X[col].corr(y) for col in feature_cols],
        }
    ).sort_values("Correlation", ascending=False)

    # Save correlation analysis
    corr_with_target.to_csv("output/target_correlation.csv", index=False)

    # Print top correlated features
    print("\nTop 10 features by correlation with target:")
    print(corr_with_target.head(10).to_string(index=False))
    logging.info(f"Top 3 features by correlation: {', '.join(corr_with_target.head(3)['Feature'].tolist())}")
    
    print(f"Feature preparation completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Feature preparation completed in {time.time() - start_time:.2f} seconds")

    return X, y, teams


def backtest_huber_regressor(X, y, n_splits=5, epsilon=1.35, alpha=0.0001):
    """Perform backtesting of HuberRegressor with time series splits"""
    print(f"\nPerforming backtesting with {n_splits} time series splits...")
    logging.info(f"Starting backtesting with {n_splits} time series splits")
    start_time = time.time()

    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Results storage
    results = []
    all_predictions = []
    all_actual = []
    fold_boundaries = []
    current_idx = 0

    # Scale features
    scaler = StandardScaler()

    # For each fold
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        fold_start_time = time.time()
        print(f"\nFold {i+1}/{n_splits}:")
        print(f"Training samples: {len(train_idx)}, Testing samples: {len(test_idx)}")
        logging.info(f"Fold {i+1}/{n_splits}: Training={len(train_idx)}, Testing={len(test_idx)}")

        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Feature selection
        print("Performing feature selection...")
        selector = SelectFromModel(HuberRegressor(epsilon=epsilon, alpha=alpha / 10))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        # Get selected features for this fold
        selected_mask = selector.get_support()
        selected_names = X.columns[selected_mask].tolist()
        print(f"Selected {len(selected_names)} features")
        logging.info(f"Fold {i+1}: Selected {len(selected_names)} features")

        # Print selected features for this fold
        print("Selected features for this fold:")
        features_str = ", ".join(selected_names[:5])
        if len(selected_names) > 5:
            features_str += f", ... and {len(selected_names) - 5} more"
        print(features_str)
        logging.info(f"Top selected features: {', '.join(selected_names[:3])}")

        # Train HuberRegressor
        print("Training HuberRegressor model...")
        model = HuberRegressor(epsilon=epsilon, max_iter=2000, alpha=alpha)
        model.fit(X_train_selected, y_train)

        # Get coefficients
        coefficients = model.coef_

        # Make predictions
        predictions = model.predict(X_test_selected)

        # Binary predictions (win/loss)
        binary_preds = (predictions >= 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, binary_preds)
        precision = precision_score(y_test, binary_preds, zero_division=0)
        recall = recall_score(y_test, binary_preds, zero_division=0)
        f1 = f1_score(y_test, binary_preds, zero_division=0)

        # Try to calculate ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, predictions)
        except:
            roc_auc = np.nan

        # Confusion matrix
        cm = confusion_matrix(y_test, binary_preds)
        true_neg = cm[0][0] if cm.shape == (2, 2) else 0
        false_pos = cm[0][1] if cm.shape == (2, 2) else 0
        false_neg = cm[1][0] if cm.shape == (2, 2) else 0
        true_pos = cm[1][1] if cm.shape == (2, 2) else 0

        # Store results
        results.append(
            {
                "Fold": i + 1,
                "Train Size": len(train_idx),
                "Test Size": len(test_idx),
                "Selected Features": len(selected_names),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
                "True Positives": true_pos,
                "False Positives": false_pos,
                "True Negatives": true_neg,
                "False Negatives": false_neg,
            }
        )

        # Store predictions and actual values
        all_predictions.extend(predictions)
        all_actual.extend(y_test)

        # Store fold boundary
        fold_boundaries.append(current_idx + len(test_idx))
        current_idx += len(test_idx)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if not np.isnan(roc_auc):
            print(f"ROC AUC: {roc_auc:.4f}")
            
        print(f"Fold {i+1} completed in {time.time() - fold_start_time:.2f} seconds")
        logging.info(f"Fold {i+1} metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    logging.info("Training final model on all data")
    final_start_time = time.time()

    # Scale all features
    X_scaled = scaler.fit_transform(X)

    # Final feature selection
    final_selector = SelectFromModel(HuberRegressor(epsilon=epsilon, alpha=alpha / 10))
    X_selected = final_selector.fit_transform(X_scaled, y)

    # Get selected features
    final_selected_mask = final_selector.get_support()
    final_selected_features = X.columns[final_selected_mask].tolist()
    print(f"Final model uses {len(final_selected_features)} selected features")
    logging.info(f"Final model uses {len(final_selected_features)} selected features")

    # Train final model
    final_model = HuberRegressor(epsilon=epsilon, max_iter=2000, alpha=alpha)
    final_model.fit(X_selected, y)

    # Save models and components
    dump(scaler, "output/scaler.joblib")
    dump(final_selector, "output/feature_selector.joblib")
    dump(final_model, "output/huber_model.joblib")

    # Save selected features
    pd.DataFrame({"Selected Features": final_selected_features}).to_csv(
        "output/selected_features.csv", index=False
    )

    # Create feature importance for final model
    feature_importance = pd.DataFrame(
        {"Feature": final_selected_features, "Coefficient": final_model.coef_}
    )
    feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
    feature_importance = feature_importance.sort_values(
        "Abs_Coefficient", ascending=False
    )
    feature_importance.to_csv("output/feature_importance.csv", index=False)
    
    print(f"Final model training completed in {time.time() - final_start_time:.2f} seconds")
    print(f"Total backtesting completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Backtesting completed in {time.time() - start_time:.2f} seconds")

    return {
        "results": pd.DataFrame(results),
        "predictions": all_predictions,
        "actual": all_actual,
        "fold_boundaries": fold_boundaries,
        "selected_features": final_selected_features,
        "feature_importance": feature_importance,
        "scaler": scaler,
        "selector": final_selector,
        "model": final_model,
    }


def visualize_backtesting_results(backtest_results):
    """Visualize and analyze backtesting results"""
    print("\nVisualizing backtesting results...")
    logging.info("Visualizing backtesting results")
    start_time = time.time()

    # Extract data from results
    results_df = backtest_results["results"]
    predictions = backtest_results["predictions"]
    actual = backtest_results["actual"]
    fold_boundaries = backtest_results["fold_boundaries"]
    feature_importance = backtest_results["feature_importance"]

    # 1. Performance metrics by fold
    plt.figure(figsize=(12, 6))
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        plt.plot(results_df["Fold"], results_df[metric], marker="o", label=metric)
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Performance Metrics by Fold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/performance_by_fold.png")

    # 2. Predictions vs Actual
    plt.figure(figsize=(14, 6))
    plt.scatter(range(len(actual)), actual, label="Actual", alpha=0.6, color="blue")
    plt.scatter(
        range(len(predictions)), predictions, label="Predicted", alpha=0.6, color="red"
    )

    # Add fold boundaries
    for boundary in fold_boundaries[:-1]:
        plt.axvline(x=boundary, color="green", linestyle="--", alpha=0.5)

    # Add threshold line
    plt.axhline(y=0.5, color="black", linestyle="-", alpha=0.3)

    plt.xlabel("Sample")
    plt.ylabel("Win Probability")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/predictions_vs_actual.png")

    # 3. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm_data = results_df[
        ["True Positives", "False Positives", "True Negatives", "False Negatives"]
    ].sum()
    cm = np.array(
        [
            [cm_data["True Negatives"], cm_data["False Positives"]],
            [cm_data["False Negatives"], cm_data["True Positives"]],
        ]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix (All Folds)")
    plt.tight_layout()
    plt.savefig("output/confusion_matrix.png")

    # 4. Top Feature Importance
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(feature_importance))
    top_features = feature_importance.head(top_n)

    # Colors based on coefficient sign
    colors = ["red" if x < 0 else "green" for x in top_features["Coefficient"]]

    plt.barh(top_features["Feature"], top_features["Coefficient"], color=colors)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Features by Importance")
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.5)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")

    # 5. Threshold Analysis
    # Calculate optimal threshold
    thresholds = np.linspace(0.2, 0.8, 25)
    threshold_metrics = []

    for threshold in thresholds:
        binary_preds = [1 if p >= threshold else 0 for p in predictions]
        accuracy = accuracy_score(actual, binary_preds)
        precision = precision_score(actual, binary_preds, zero_division=0)
        recall = recall_score(actual, binary_preds, zero_division=0)

        # Calculate profit assuming 2.0 odds (-100 for loss, +100 for win)
        true_positives = sum(
            1 for a, p in zip(actual, binary_preds) if a == 1 and p == 1
        )
        false_positives = sum(
            1 for a, p in zip(actual, binary_preds) if a == 0 and p == 1
        )
        profit = (true_positives * 100) - (false_positives * 100)
        roi = (
            profit / ((true_positives + false_positives) * 100)
            if (true_positives + false_positives) > 0
            else 0
        )
        total_bets = true_positives + false_positives

        threshold_metrics.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Total Bets": total_bets,
                "Win Rate": true_positives / total_bets if total_bets > 0 else 0,
                "Profit ($)": profit,
                "ROI": roi,
            }
        )

    threshold_df = pd.DataFrame(threshold_metrics)
    threshold_df.to_csv("output/threshold_analysis.csv", index=False)

    # Plot threshold analysis
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Accuracy"],
        marker="o",
        label="Accuracy",
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        marker="s",
        label="Precision",
    )
    plt.plot(
        threshold_df["Threshold"], threshold_df["Recall"], marker="^", label="Recall"
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Win Rate"],
        marker="*",
        label="Win Rate",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Performance Metrics by Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Profit ($)"],
        marker="o",
        color="green",
        label="Profit ($)",
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Total Bets"],
        marker="s",
        color="orange",
        label="Total Bets",
    )

    # Add reference line at 0 profit
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)

    # Find and mark optimal threshold for profit
    optimal_idx = threshold_df["Profit ($)"].idxmax()
    optimal_threshold = threshold_df.loc[optimal_idx, "Threshold"]
    optimal_profit = threshold_df.loc[optimal_idx, "Profit ($)"]

    plt.scatter([optimal_threshold], [optimal_profit], color="red", s=100, zorder=5)
    plt.annotate(
        f"Optimal: {optimal_threshold:.2f}",
        xy=(optimal_threshold, optimal_profit),
        xytext=(optimal_threshold + 0.05, optimal_profit - 50),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=7),
    )

    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title("Profit and Total Bets by Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/threshold_analysis.png")
    
    print(f"Visualization completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Visualization completed in {time.time() - start_time:.2f} seconds")
    logging.info(f"Optimal threshold: {optimal_threshold:.4f}, profit: ${optimal_profit:.2f}")

    return threshold_df, optimal_threshold


def save_prediction_model(backtest_results, optimal_threshold):
    """Save prediction model components and example usage code"""
    # Save optimal threshold
    with open("output/optimal_threshold.txt", "w") as f:
        f.write(f"{optimal_threshold}")

    # Create example prediction code
    example_code = f"""
# Example code for using the trained HuberRegressor model for baseball game prediction

import pandas as pd
import numpy as np
from joblib import load
from datetime import datetime, timedelta

# Load model components
scaler = load('output/scaler.joblib')
feature_selector = load('output/feature_selector.joblib')
model = load('output/huber_model.joblib')

# Load optimal threshold
with open('output/optimal_threshold.txt', 'r') as f:
    optimal_threshold = float(f.read().strip())

def predict_game_outcome(game_stats):
    '''Predict the win probability for a new game'''
    # Ensure game_stats has the same features as the training data
    # Scale the features
    game_stats_scaled = scaler.transform(game_stats)
    
    # Select important features
    game_stats_selected = feature_selector.transform(game_stats_scaled)
    
    # Make prediction
    win_probability = model.predict(game_stats_selected)[0]
    
    # Decision using optimal threshold
    prediction = 'Win' if win_probability >= optimal_threshold else 'Loss'
    
    return {{
        'win_probability': win_probability,
        'prediction': prediction,
        'confidence': abs(win_probability - 0.5) * 2,  # Scale 0-1
        'bet_recommended': win_probability >= optimal_threshold
    }}

def calculate_feature_interactions(df):
    '''Calculate the same feature interactions as used in training'''
    # Calculate offensive power metrics
    if all(col in df.columns for col in ['OBP', 'SLG']):
        df['OPS'] = df['OBP'] + df['SLG']
    
    if all(col in df.columns for col in ['OPS', 'WRC+']):
        df['OffensivePower'] = df['OPS'] * df['WRC+'] / 100
    
    # Calculate pitching effectiveness
    if all(col in df.columns for col in ['ERA', 'FIP']):
        df['PitchingEffectiveness'] = (df['FIP'] + df['ERA']) / 2
    elif 'ERA' in df.columns:
        df['PitchingEffectiveness'] = df['ERA']
    
    # Calculate batting discipline
    if all(col in df.columns for col in ['K Percentage', 'BB Percentage']):
        df['BattingDiscipline'] = df['BB Percentage'] / (df['K Percentage'] + 0.001)
    
    # Calculate pitcher control
    if all(col in df.columns for col in ['Opposing K/9', 'Opposing BB/9']):
        df['PitcherControl'] = df['Opposing K/9'] / (df['Opposing BB/9'] + 0.001)
    
    # Calculate the relative hitting strength
    if all(col in df.columns for col in ['AVG', 'OBP', 'SLG']):
        df['HittingStrength'] = (df['AVG'] * 0.25) + (df['OBP'] * 0.35) + (df['SLG'] * 0.4)

    # Calculate the relative pitching strength
    if all(col in df.columns for col in ['ERA', 'Opposing K/9', 'Opposing BB/9']):
        df['PitchingStrength'] = df['Opposing K/9'] - df['Opposing BB/9'] - (df['ERA'] / 5)
        
    # Calculate team net WAR
    if all(col in df.columns for col in ['WAR', 'Opposing War']):
        df['NetWAR'] = df['WAR'] - df['Opposing War']
    
    # Add other interactions that were created during training
    # ...
    
    return df

def prepare_game_data(home_team_stats, away_team_stats):
    '''Prepare game data in the correct format for prediction'''
    # Create a DataFrame with the correct structure
    game_data = pd.DataFrame()
    
    # Make sure 'Runs Scored' is not included (prevents data leakage)
    if 'Runs Scored' in home_team_stats:
        del home_team_stats['Runs Scored']
    if 'Runs Scored' in away_team_stats:
        del away_team_stats['Runs Scored']
    
    # Set up the two rows (home team as offensive, away team as offensive)
    home_offensive = {{**home_team_stats, 'Offensive Team': home_team_stats['team'], 'Defensive Team': away_team_stats['team']}}
    away_offensive = {{**away_team_stats, 'Offensive Team': away_team_stats['team'], 'Defensive Team': home_team_stats['team']}}
    
    # Add rows to the DataFrame
    game_data = pd.concat([game_data, pd.DataFrame([home_offensive, away_offensive])])
    
    # Calculate feature interactions
    game_data = calculate_feature_interactions(game_data)
    
    # Remove team columns and any other non-feature columns
    exclude_cols = ['team', 'Offensive Team', 'Defensive Team', 'Date']
    feature_cols = [col for col in game_data.columns if col not in exclude_cols]
    
    return game_data[feature_cols]

# Example usage:
# 1. Load team statistics (can be from an API, database, etc.)
# home_team_stats = get_team_stats('NYY')  # Custom function to get team stats
# away_team_stats = get_team_stats('BOS')  # Custom function to get team stats
#
# 2. Prepare game data
# game_features = prepare_game_data(home_team_stats, away_team_stats)
#
# 3. Make prediction for home team
# home_result = predict_game_outcome(game_features.iloc[0:1])
# print(f"Home team win probability: {{home_result['win_probability']:.2f}}")
# print(f"Prediction: {{home_result['prediction']}}")
# print(f"Confidence: {{home_result['confidence']:.2f}}")
# print(f"Bet recommended: {{home_result['bet_recommended']}}")
"""

    # Save example code
    with open("output/prediction_example.py", "w") as f:
        f.write(example_code)

    print(f"\nSaved prediction model and example code to 'output' directory")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    logging.info(f"Saved prediction model with optimal threshold: {optimal_threshold:.4f}")


def perform_validation_testing(
    X, y, teams=None, validation_size=0.2, optimal_threshold=None
):
    """Perform validation testing using a holdout set of the most recent data"""
    print("\n" + "=" * 80)
    print("Validation Testing on Most Recent Data")
    print("=" * 80)
    logging.info("Starting validation testing on most recent data")
    start_time = time.time()

    # Determine split point for validation
    total_samples = len(X)
    validation_samples = int(total_samples * validation_size)
    training_samples = total_samples - validation_samples

    print(f"Total samples: {total_samples}")
    print(
        f"Training samples: {training_samples} ({(1-validation_size)*100:.1f}% of data)"
    )
    print(
        f"Validation samples: {validation_samples} ({validation_size*100:.1f}% of data)"
    )
    logging.info(f"Training: {training_samples}, Validation: {validation_samples}")

    # Split data chronologically (most recent data as validation)
    X_train, X_val = X.iloc[:training_samples], X.iloc[training_samples:]
    y_train, y_val = y.iloc[:training_samples], y.iloc[training_samples:]

    if teams is not None:
        teams_train, teams_val = (
            teams.iloc[:training_samples],
            teams.iloc[training_samples:],
        )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Feature selection
    print("\nPerforming feature selection...")
    selector = SelectFromModel(HuberRegressor(epsilon=1.35, alpha=0.0001 / 10))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)

    # Get selected features
    selected_mask = selector.get_support()
    selected_names = X.columns[selected_mask].tolist()
    print(f"Selected {len(selected_names)} features")
    logging.info(f"Selected {len(selected_names)} features for validation")

    # Print selected features
    print("Selected features for validation:")
    features_str = ", ".join(selected_names[:5])
    if len(selected_names) > 5:
        features_str += f", ... and {len(selected_names) - 5} more"
    print(features_str)
    logging.info(f"Top validation features: {', '.join(selected_names[:3])}")

    # Train model
    print("Training model on training data...")
    model = HuberRegressor(epsilon=1.35, max_iter=2000, alpha=0.0001)
    model.fit(X_train_selected, y_train)

    # Make predictions on validation set
    print("Making predictions on validation data...")
    val_predictions = model.predict(X_val_selected)

    # If no optimal threshold provided, use 0.5
    if optimal_threshold is None:
        optimal_threshold = 0.5

    # Binary predictions
    val_binary_preds = (val_predictions >= optimal_threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_val, val_binary_preds)
    precision = precision_score(y_val, val_binary_preds, zero_division=0)
    recall = recall_score(y_val, val_binary_preds, zero_division=0)
    f1 = f1_score(y_val, val_binary_preds, zero_division=0)

    # Try to calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_val, val_predictions)
    except:
        roc_auc = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_val, val_binary_preds)

    # Calculate betting metrics
    true_positives = sum(
        1 for a, p in zip(y_val, val_binary_preds) if a == 1 and p == 1
    )
    false_positives = sum(
        1 for a, p in zip(y_val, val_binary_preds) if a == 0 and p == 1
    )
    win_rate = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    total_bets = true_positives + false_positives
    profit = (true_positives * 100) - (false_positives * 100)
    roi = (
        profit / ((true_positives + false_positives) * 100)
        if (true_positives + false_positives) > 0
        else 0
    )

    # Print metrics
    print("\nValidation Performance:")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC AUC: {roc_auc:.4f}")

    print("\nValidation Betting Performance:")
    print("-" * 40)
    print(f"Win Rate: {win_rate:.4f}")
    print(f"Total Bets: {total_bets}")
    print(f"Profit: ${profit:.2f}")
    print(f"ROI: {roi:.4f}")
    
    logging.info(f"Validation metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, Win Rate={win_rate:.4f}")
    logging.info(f"Validation betting: Profit=${profit:.2f}, ROI={roi:.4f}, Bets={total_bets}")

    # Visualize validation results
    plt.figure(figsize=(14, 10))

    # 1. Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Validation Set Confusion Matrix")

    # 2. Predictions vs Actual
    plt.subplot(2, 2, 2)
    plt.scatter(range(len(y_val)), y_val, label="Actual", alpha=0.6, color="blue")
    plt.scatter(
        range(len(val_predictions)),
        val_predictions,
        label="Predicted",
        alpha=0.6,
        color="red",
    )
    plt.axhline(y=optimal_threshold, color="black", linestyle="-", alpha=0.3)
    plt.xlabel("Sample")
    plt.ylabel("Win Probability")
    plt.title("Validation: Actual vs Predicted Values")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Distribution of probabilities
    plt.subplot(2, 2, 3)
    plt.hist(val_predictions, bins=20, alpha=0.7, color="green")
    plt.axvline(x=optimal_threshold, color="red", linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predictions")
    plt.grid(True, alpha=0.3)

    # 4. Daily performance tracking
    if teams is not None and "Date" in teams.columns:
        dates = teams_val["Date"]
        date_groups = pd.DataFrame(
            {"date": dates, "actual": y_val, "predicted": val_binary_preds}
        )
        date_groups["correct"] = (
            date_groups["actual"] == date_groups["predicted"]
        ).astype(int)
        daily_performance = (
            date_groups.groupby("date")
            .agg({"correct": "mean", "predicted": "sum", "actual": "count"})
            .reset_index()
        )
        daily_performance["bets"] = daily_performance["predicted"]

        plt.subplot(2, 2, 4)
        plt.plot(
            range(len(daily_performance)),
            daily_performance["correct"],
            marker="o",
            label="Accuracy",
        )
        plt.plot(
            range(len(daily_performance)),
            daily_performance["bets"] / daily_performance["actual"],
            marker="s",
            label="Bet Rate",
        )
        plt.xlabel("Day")
        plt.ylabel("Rate")
        plt.title("Daily Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/validation_results.png")

    # Threshold analysis for validation set
    thresholds = np.linspace(0.2, 0.8, 25)
    threshold_metrics = []

    for threshold in thresholds:
        binary_preds = [1 if p >= threshold else 0 for p in val_predictions]
        accuracy = accuracy_score(y_val, binary_preds)
        precision = precision_score(y_val, binary_preds, zero_division=0)
        recall = recall_score(y_val, binary_preds, zero_division=0)

        # Calculate profit
        true_positives = sum(
            1 for a, p in zip(y_val, binary_preds) if a == 1 and p == 1
        )
        false_positives = sum(
            1 for a, p in zip(y_val, binary_preds) if a == 0 and p == 1
        )
        profit = (true_positives * 100) - (false_positives * 100)
        roi = (
            profit / ((true_positives + false_positives) * 100)
            if (true_positives + false_positives) > 0
            else 0
        )
        total_bets = true_positives + false_positives
        win_rate = true_positives / total_bets if total_bets > 0 else 0

        threshold_metrics.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Total Bets": total_bets,
                "Win Rate": win_rate,
                "Profit ($)": profit,
                "ROI": roi,
            }
        )

    val_threshold_df = pd.DataFrame(threshold_metrics)
    val_threshold_df.to_csv("output/validation_threshold_analysis.csv", index=False)

    # Plot validation threshold analysis
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(
        val_threshold_df["Threshold"],
        val_threshold_df["Accuracy"],
        marker="o",
        label="Accuracy",
    )
    plt.plot(
        val_threshold_df["Threshold"],
        val_threshold_df["Precision"],
        marker="s",
        label="Precision",
    )
    plt.plot(
        val_threshold_df["Threshold"],
        val_threshold_df["Recall"],
        marker="^",
        label="Recall",
    )
    plt.plot(
        val_threshold_df["Threshold"],
        val_threshold_df["Win Rate"],
        marker="*",
        label="Win Rate",
    )
    plt.axvline(
        x=optimal_threshold,
        color="red",
        linestyle="--",
        label=f"Optimal ({optimal_threshold:.2f})",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Validation: Performance Metrics by Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(
        val_threshold_df["Threshold"],
        val_threshold_df["Profit ($)"],
        marker="o",
        color="green",
        label="Profit ($)",
    )
    plt.plot(
        val_threshold_df["Threshold"],
        val_threshold_df["Total Bets"],
        marker="s",
        color="orange",
        label="Total Bets",
    )
    plt.axvline(
        x=optimal_threshold,
        color="red",
        linestyle="--",
        label=f"Optimal ({optimal_threshold:.2f})",
    )
    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title("Validation: Profit and Total Bets by Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/validation_threshold_analysis.png")

    # Find and report optimal threshold for validation set
    val_optimal_idx = val_threshold_df["Profit ($)"].idxmax()
    val_optimal_threshold = val_threshold_df.loc[val_optimal_idx, "Threshold"]
    val_optimal_profit = val_threshold_df.loc[val_optimal_idx, "Profit ($)"]
    val_optimal_roi = val_threshold_df.loc[val_optimal_idx, "ROI"]

    print("\nValidation Set Optimal Threshold Analysis:")
    print("-" * 40)
    print(f"Validation Optimal Threshold: {val_optimal_threshold:.4f}")
    print(f"Validation Profit at Optimal: ${val_optimal_profit:.2f}")
    print(f"Validation ROI at Optimal: {val_optimal_roi:.4f}")

    # Compare with backtesting optimal threshold
    backtesting_profit_at_val_optimal = val_threshold_df.loc[
        val_threshold_df["Threshold"].sub(optimal_threshold).abs().idxmin(),
        "Profit ($)",
    ]

    print(f"\nBacktesting Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Validation Profit at Backtesting Optimal: ${profit:.2f}")

    if val_optimal_threshold != optimal_threshold:
        improvement = ((val_optimal_profit / profit) - 1) * 100
        print(f"Potential improvement by using validation optimal: {improvement:.2f}%")
        
    logging.info(f"Validation optimal threshold: {val_optimal_threshold:.4f}")
    logging.info(f"Validation profit at optimal: ${val_optimal_profit:.2f}, ROI: {val_optimal_roi:.4f}")
    logging.info(f"Validation testing completed in {time.time() - start_time:.2f} seconds")

    return {
        "validation_accuracy": accuracy,
        "validation_precision": precision,
        "validation_recall": recall,
        "validation_f1": f1,
        "validation_profit": profit,
        "validation_roi": roi,
        "validation_win_rate": win_rate,
        "validation_total_bets": total_bets,
        "optimal_threshold": val_optimal_threshold,
        "optimal_profit": val_optimal_profit,
        "optimal_roi": val_optimal_roi,
    }


def main():
    """Main function to run the enhanced baseball win prediction system"""
    start_time = time.time()
    print("=" * 80)
    print("Enhanced Baseball Win Prediction System")
    print("=" * 80)
    logging.info("Starting Enhanced Baseball Win Prediction System")

    # Record the start time for performance tracking
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"Run start time: {timestamp}")

    # 1. Load and preprocess data
    data_path = "./stats_updated.csv"
    if not os.path.exists(data_path):
        data_path = "./stats.csv"
        print(f"stats_updated.csv not found, using {data_path} instead")
        logging.warning(f"stats_updated.csv not found, using {data_path} instead")
        
    df = load_baseball_data(data_path)

    # 2. Prepare features and target
    X, y, teams = prepare_features_and_target(df)

    # 3. Perform backtesting
    backtest_results = backtest_huber_regressor(X, y, n_splits=5)

    # 4. Visualize and analyze results
    threshold_df, optimal_threshold = visualize_backtesting_results(backtest_results)

    # 5. Save prediction model
    save_prediction_model(backtest_results, optimal_threshold)

    # 6. Print summary statistics
    results_df = backtest_results["results"]

    print("\nPerformance Summary:")
    print("-" * 40)
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        avg = results_df[metric].mean()
        std = results_df[metric].std()
        print(f"{metric}: {avg:.4f} ± {std:.4f}")
        logging.info(f"{metric}: {avg:.4f} ± {std:.4f}")

    # Profitability at optimal threshold
    optimal_row = threshold_df.loc[
        threshold_df["Threshold"].sub(optimal_threshold).abs().idxmin()
    ]
    print("\nProfitability at Optimal Threshold:")
    print("-" * 40)
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Win Rate: {optimal_row['Win Rate']:.4f}")
    print(f"Total Bets: {optimal_row['Total Bets']:.0f}")
    print(f"Profit: ${optimal_row['Profit ($)']:.2f}")
    print(f"ROI: {optimal_row['ROI']:.4f}")
    
    logging.info(f"Optimal Threshold: {optimal_threshold:.4f}")
    logging.info(f"Win Rate: {optimal_row['Win Rate']:.4f}")
    logging.info(f"Profit: ${optimal_row['Profit ($)']:.2f}, ROI: {optimal_row['ROI']:.4f}")

    # 7. Run validation testing on most recent data
    validation_size = 0.15  # Use 15% of most recent data for validation
    validation_results = perform_validation_testing(
        X, y, teams, validation_size, optimal_threshold
    )

    # 8. Report final performance and execution time
    total_runtime = time.time() - start_time
    print("\nAll results and visualizations have been saved to the 'output' directory.")
    print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    print("=" * 80)
    
    logging.info(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    logging.info("Baseball Win Prediction System completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        print("See baseball_model.log for details")