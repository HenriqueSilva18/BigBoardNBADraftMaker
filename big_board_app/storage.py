import json

import pandas as pd

from .config import BOARD_COLUMNS, BOARD_SAVE_FILE, EVAL_CATEGORIES, RANK_COLUMN, SCORE_COLUMN


def create_empty_board():
    return pd.DataFrame(columns=BOARD_COLUMNS)


def order_big_board(big_board):
    board = big_board.copy()
    if board.empty:
        return board

    ranks = pd.to_numeric(board.get(RANK_COLUMN), errors="coerce")
    if ranks.notna().any():
        board[RANK_COLUMN] = ranks
        return board.sort_values(
            by=[RANK_COLUMN, SCORE_COLUMN, "Name"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)

    return board.sort_values(
        by=[SCORE_COLUMN, "Name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def normalize_ranks(big_board):
    board = order_big_board(big_board)
    if not board.empty:
        board[RANK_COLUMN] = range(1, len(board) + 1)
    return board


def normalize_big_board(df):
    if df is None:
        return create_empty_board()

    board = df.copy()

    for column in BOARD_COLUMNS:
        if column not in board.columns:
            if column in EVAL_CATEGORIES:
                board[column] = 5
            elif column == RANK_COLUMN:
                board[column] = pd.NA
            else:
                board[column] = "N/A"

    board = board.loc[:, ~board.columns.duplicated()]
    board = board[BOARD_COLUMNS]
    board[EVAL_CATEGORIES] = (
        board[EVAL_CATEGORIES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(5)
        .astype(int)
    )
    board[SCORE_COLUMN] = (
        pd.to_numeric(board[SCORE_COLUMN], errors="coerce")
        .fillna(0)
        .round(2)
    )
    board[RANK_COLUMN] = pd.to_numeric(board[RANK_COLUMN], errors="coerce")

    board = board.drop_duplicates(subset="Name", keep="first").reset_index(drop=True)
    return normalize_ranks(board)


try:
    import js
    IS_BROWSER = True
except ImportError:
    IS_BROWSER = False


def load_big_board_from_json(fileobj=None, filename=BOARD_SAVE_FILE):
    if fileobj is not None:
        raw = fileobj.read()
    else:
        raw = None
        if IS_BROWSER:
            try:
                saved_data = js.window.localStorage.getItem("big_board_save_data")
                if saved_data:
                    raw = saved_data.encode("utf-8")
            except Exception:
                pass
        
        if raw is None:
            if filename.exists():
                raw = filename.read_bytes()
            else:
                return None

    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = [data]

    return normalize_big_board(pd.DataFrame(data))


def save_big_board_to_file(big_board, filename=BOARD_SAVE_FILE):
    board = normalize_big_board(big_board)
    board.to_json(filename, orient="records", indent=2)
    
    if IS_BROWSER:
        try:
            json_data = board.to_json(orient="records", indent=2)
            js.window.localStorage.setItem("big_board_save_data", json_data)
        except Exception:
            pass


def big_board_to_json_bytes(big_board):
    board = normalize_big_board(big_board)
    return board.to_json(orient="records", indent=2).encode("utf-8")


def save_big_board_to_txt(big_board):
    board = normalize_big_board(big_board)
    if board.empty:
        return "Big Board is empty. Nothing to save."

    lines = ["NBA Draft Big Board 2026 Rankings\n"]
    board = order_big_board(board).reset_index(drop=True)
    max_name_len = board["Name"].astype(str).str.len().max()

    for index, row in board.iterrows():
        rank = int(row.get(RANK_COLUMN, index + 1))
        name = str(row.get("Name", "N/A")).ljust(max_name_len)
        position = row.get("Position", "N/A")
        team = row.get("College/Team", "N/A")
        lines.append(f"{rank:2}. {name} - {position} - {team}")

    return "\n".join(lines)
