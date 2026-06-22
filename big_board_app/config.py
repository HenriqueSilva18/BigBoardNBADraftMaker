import sys
import shutil
from pathlib import Path

APP_TITLE = "Draft Room 2026"
APP_SUBTITLE = "Build, tune, and compare your NBA Draft board."

DATA_FILE = Path("nba_prospects_2026_stats.csv")

if "pyodide" in sys.modules:
    BOARD_SAVE_FILE = Path("/mnt/big_board_save.json")
    if not BOARD_SAVE_FILE.exists() and Path("big_board_save.json").exists():
        # Copy the default empty or pre-existing board to the persistent mount on first load
        shutil.copy("big_board_save.json", BOARD_SAVE_FILE)
else:
    BOARD_SAVE_FILE = Path("big_board_save.json")

SCORE_COLUMN = "Média Ponderada"
RANK_COLUMN = "Rank"

EVAL_CATEGORIES = [
    "Athleticism",
    "Dribbling",
    "Shooting",
    "Perimeter Defense",
    "Passing",
    "Scoring",
    "Interior Defense",
    "Basketball IQ",
    "Intangibles",
]

WEIGHTS = {
    "Athleticism": 1 / 8,
    "Scoring": 1 / 8,
    "Shooting": 1 / 8,
    "Dribbling": 1 / 8,
    "Passing": 1 / 8,
    "Perimeter Defense": 1 / 16,
    "Interior Defense": 1 / 16,
    "Basketball IQ": 1 / 8,
    "Intangibles": 1 / 8,
}

BOARD_BASE_COLUMNS = [
    RANK_COLUMN,
    "Name",
    "Age",
    "Measurements",
    "Position",
    "College/Team",
    "Tier",
    SCORE_COLUMN,
]

BOARD_COLUMNS = BOARD_BASE_COLUMNS + EVAL_CATEGORIES

SOURCE_COLUMNS = [
    "name",
    "team",
    "year",
    "position",
    "measurements",
    "weight",
    "mock_draft",
    "big_board",
    "age_at_draft",
    "birthdate",
    "nation",
    "hometown",
    "high_school",
    "espn_100",
    "strengths",
    "weaknesses",
    "max_vert",
    "lane_agil",
    "shuttle",
    "3_4sprint",
    "reach",
    "wingspan",
    "games",
    "minutes_per_game",
    "fgm_fga",
    "fg_pct",
    "3pm_3pa",
    "3p_pct",
    "ftm_fta",
    "ft_pct",
    "rebounds_pg",
    "assists_pg",
    "blocks_pg",
    "steals_pg",
    "turnovers_pg",
    "personal_fouls_pg",
    "points_pg",
    "games_per36",
    "minutes_per36",
    "fgm_fga_per36",
    "fg_pct_per36",
    "3pm_3pa_per36",
    "3p_pct_per36",
    "ftm_fta_per36",
    "ft_pct_per36",
    "rebounds_per36",
    "assists_per36",
    "blocks_per36",
    "steals_per36",
    "turnovers_per36",
    "personal_fouls_per36",
    "points_per36",
    "ts_per",
    "efg_per",
    "3pa_rate",
    "fta_rate",
    "nba_3p_per",
    "usg_per",
    "ast_per_usg",
    "ast_per_to",
    "per",
    "ows_per_40",
    "dws_per_40",
    "ws_per_40",
    "ortg",
    "drtg",
    "obpm",
    "dbpm",
    "bpm",
]

PLAYER_COLUMNS = [
    "name",
    "team",
    "year",
    "position",
    "measurements",
    "age_at_draft",
    "nation",
    "wingspan",
    "games",
    "minutes_per_game",
    "3p_pct",
    "ft_pct",
    "rebounds_per36",
    "assists_per36",
    "blocks_per36",
    "steals_per36",
    "turnovers_per36",
    "points_per36",
    "ts_per",
    "3pa_rate",
    "fta_rate",
    "usg_per",
    "ast_per_usg",
    "ast_per_to",
    "obpm",
    "dbpm",
    "bpm",
]

STATS_COLUMNS = [
    "games",
    "points_per36",
    "assists_per36",
    "rebounds_per36",
    "steals_per36",
    "blocks_per36",
    "turnovers_per36",
    "obpm",
    "dbpm",
    "bpm",
    "usg_per",
    "ts_per",
    "ast_per_to",
    "3p_pct",
    "3pa_rate",
    "fta_rate",
]

STATS_RENAME_MAP = {
    "games": "GP",
    "points_per36": "PTS/36",
    "assists_per36": "AST/36",
    "rebounds_per36": "REB/36",
    "steals_per36": "STL/36",
    "blocks_per36": "BLK/36",
    "turnovers_per36": "TO/36",
    "obpm": "OBPM",
    "dbpm": "DBPM",
    "bpm": "BPM",
    "usg_per": "USG%",
    "ts_per": "TS%",
    "ast_per_to": "AST/TO",
    "3p_pct": "3PT%",
    "3pa_rate": "3PA Rate",
    "fta_rate": "FTA Rate",
}
