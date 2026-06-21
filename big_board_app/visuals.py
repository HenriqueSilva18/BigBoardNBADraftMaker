import io

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EVAL_CATEGORIES, SCORE_COLUMN, STATS_RENAME_MAP


def apply_extreme_highlighting(df_display, df_numeric, lower_is_better=()):
    styles = pd.DataFrame("", index=df_display.index, columns=df_display.columns)

    for column in df_numeric.columns:
        series = df_numeric[column].dropna()
        if series.empty:
            continue

        max_idx = series.idxmax()
        min_idx = series.idxmin()

        if column in lower_is_better:
            styles.loc[min_idx, column] = "background-color: #d8f5df; color: #12351f"
            styles.loc[max_idx, column] = "background-color: #ffe0df; color: #4a1512"
        else:
            styles.loc[max_idx, column] = "background-color: #d8f5df; color: #12351f"
            styles.loc[min_idx, column] = "background-color: #ffe0df; color: #4a1512"

    return styles


def build_stats_table(stats_records, selected_players):
    stats_df = pd.DataFrame(stats_records)
    if stats_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    stats_df.insert(0, "Name", selected_players)
    stats_df.set_index("Name", inplace=True)

    available = [column for column in STATS_RENAME_MAP if column in stats_df.columns]
    display_df = stats_df[available].rename(columns=STATS_RENAME_MAP)
    display_df = display_df.replace(["", "-", "N/A", None], np.nan)
    numeric_df = display_df.apply(pd.to_numeric, errors="coerce")

    for column in ["TS%", "3PT%"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda value: f"{value * 100:.1f}%" if pd.notnull(value) else "-"
            )

    if "USG%" in display_df.columns:
        display_df["USG%"] = display_df["USG%"].apply(
            lambda value: f"{value:.1f}%" if pd.notnull(value) else "-"
        )

    for column in display_df.columns:
        if column in ["3PA Rate", "FTA Rate"]:
            display_df[column] = display_df[column].apply(
                lambda value: f"{value:.3f}" if pd.notnull(value) else "-"
            )
        elif column in ["PTS/36", "AST/36", "REB/36", "STL/36", "BLK/36", "TO/36", "OBPM", "DBPM", "BPM"]:
            display_df[column] = display_df[column].apply(
                lambda value: f"{value:.1f}" if pd.notnull(value) else "-"
            )
        elif column not in ["TS%", "3PT%", "USG%", "GP"]:
            display_df[column] = display_df[column].apply(
                lambda value: f"{value:.2f}" if pd.notnull(value) else "-"
            )

    return display_df, numeric_df


def build_eval_table(comparison_data):
    comp_table = comparison_data.set_index("Name")[EVAL_CATEGORIES + [SCORE_COLUMN]].copy()
    numeric_df = comp_table.apply(pd.to_numeric, errors="coerce")

    for column in comp_table.columns:
        if column == SCORE_COLUMN:
            comp_table[column] = comp_table[column].apply(
                lambda value: f"{value:.1f}" if pd.notnull(value) else "-"
            )
        else:
            comp_table[column] = comp_table[column].apply(
                lambda value: f"{int(value)}" if pd.notnull(value) else "-"
            )

    return comp_table, numeric_df


def create_overlaid_radar_chart(players_data, figsize=(6, 6)):
    if players_data.empty:
        return None

    categories = EVAL_CATEGORIES
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    colors = ["#1f6f50", "#b83232", "#2f8065", "#c74f4a", "#7da453"]
    for index, (_, player) in enumerate(players_data.iterrows()):
        values = player[categories].tolist()
        values += values[:1]
        color = colors[index % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2.2, label=player["Name"], color=color)
        ax.fill(angles, values, alpha=0.16, color=color)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 10)
    ax.grid(color="#cfd8d2", linestyle="--", linewidth=0.6)
    ax.spines["polar"].set_color("#8d9c93")

    outline = [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
    for angle, category in zip(angles[:-1], categories):
        ax.text(
            angle,
            11.65,
            category,
            horizontalalignment="center",
            verticalalignment="center",
            size=9,
            color="#18211b",
            path_effects=outline,
            zorder=10,
        )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.28, 1.12),
        labelcolor="#18211b",
        fontsize=9,
        frameon=False,
    )
    return fig


def figure_to_svg(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue().decode("utf-8")
