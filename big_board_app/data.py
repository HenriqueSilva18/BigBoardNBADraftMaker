import pandas as pd

from .config import DATA_FILE, PLAYER_COLUMNS, STATS_COLUMNS


def normalize_name(name):
    return str(name).strip().casefold()


def load_prospect_data(path=DATA_FILE):
    """Load the prospect CSV and keep only the columns used by the app."""
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]

    for column in PLAYER_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    for column in STATS_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df[PLAYER_COLUMNS]


def get_player_info(df, player_name):
    if df is None or df.empty:
        return {
            "age": "N/A",
            "measurements": "N/A",
            "position": "N/A",
            "team": "N/A",
        }

    matches = df[df["name"].map(normalize_name) == normalize_name(player_name)]
    if matches.empty:
        return {
            "age": "N/A",
            "measurements": "N/A",
            "position": "N/A",
            "team": "N/A",
        }

    player_data = matches.iloc[0]
    return {
        "age": player_data.get("age_at_draft", "N/A"),
        "measurements": player_data.get("measurements", "N/A"),
        "position": player_data.get("position", "N/A"),
        "team": player_data.get("team", "N/A"),
    }


def get_player_stats(df, player_name):
    if df is None or df.empty:
        return {}

    matches = df[df["name"].map(normalize_name) == normalize_name(player_name)]
    if matches.empty:
        return {}

    player_data = matches.iloc[0]
    return {
        column: pd.to_numeric(player_data.get(column, pd.NA), errors="coerce")
        for column in STATS_COLUMNS
    }
