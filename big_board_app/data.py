import pandas as pd

from .config import DATA_FILE, PLAYER_COLUMNS, STATS_COLUMNS


def load_prospect_data(path=DATA_FILE):
    """Load the prospect CSV and keep only the columns used by the app."""
    df = pd.read_csv(path)

    for column in PLAYER_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    return df[PLAYER_COLUMNS]


def get_player_info(df, player_name):
    if df is None or df.empty or player_name not in df["name"].values:
        return {
            "age": "N/A",
            "measurements": "N/A",
            "position": "N/A",
            "team": "N/A",
        }

    player_data = df[df["name"] == player_name].iloc[0]
    return {
        "age": player_data.get("age_at_draft", "N/A"),
        "measurements": player_data.get("measurements", "N/A"),
        "position": player_data.get("position", "N/A"),
        "team": player_data.get("team", "N/A"),
    }


def get_player_stats(df, player_name):
    if df is None or df.empty or player_name not in df["name"].values:
        return {}

    player_data = df[df["name"] == player_name].iloc[0]
    return {column: player_data.get(column, 0) for column in STATS_COLUMNS}
