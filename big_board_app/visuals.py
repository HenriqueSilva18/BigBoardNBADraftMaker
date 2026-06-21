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

        max_value = series.max()
        min_value = series.min()

        if max_value == min_value:
            equal_style = "background-color: #4a3b15; color: #ffe6a6; font-weight: 850"
            for index in series.index:
                styles.loc[index, column] = equal_style
            continue

        if column in lower_is_better:
            best_mask = series == min_value
            worst_mask = series == max_value
        else:
            best_mask = series == max_value
            worst_mask = series == min_value

        for index in series[best_mask].index:
            styles.loc[index, column] = "background-color: #173f2a; color: #d7ffe5; font-weight: 850"
        for index in series[worst_mask].index:
            styles.loc[index, column] = "background-color: #4a1f1f; color: #ffd9d9; font-weight: 850"

    return styles


def build_stats_table(stats_records, selected_players):
    stats_df = pd.DataFrame(stats_records)
    if stats_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    stats_df.insert(0, "Name", selected_players)

    available = [column for column in STATS_RENAME_MAP if column in stats_df.columns]
    display_df = stats_df[["Name"] + available].rename(columns=STATS_RENAME_MAP)
    display_df = display_df.replace(["", "-", "N/A", None], np.nan)
    numeric_df = pd.DataFrame(np.nan, index=display_df.index, columns=display_df.columns)
    stat_columns = [column for column in display_df.columns if column != "Name"]
    numeric_df[stat_columns] = display_df[stat_columns].apply(pd.to_numeric, errors="coerce")

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
        elif column not in ["Name", "TS%", "3PT%", "USG%", "GP"]:
            display_df[column] = display_df[column].apply(
                lambda value: f"{value:.2f}" if pd.notnull(value) else "-"
            )

    return display_df, numeric_df


def build_eval_table(comparison_data):
    comp_table = comparison_data[["Name"] + EVAL_CATEGORIES + [SCORE_COLUMN]].copy()
    numeric_df = pd.DataFrame(np.nan, index=comp_table.index, columns=comp_table.columns)
    numeric_columns = [column for column in comp_table.columns if column != "Name"]
    numeric_df[numeric_columns] = comp_table[numeric_columns].apply(pd.to_numeric, errors="coerce")

    for column in comp_table.columns:
        if column == "Name":
            continue
        if column == SCORE_COLUMN:
            comp_table[column] = comp_table[column].apply(
                lambda value: f"{value:.1f}" if pd.notnull(value) else "-"
            )
        else:
            comp_table[column] = comp_table[column].apply(
                lambda value: f"{int(value)}" if pd.notnull(value) else "-"
            )

    return comp_table, numeric_df


def create_overlaid_radar_chart(players_data, figsize=(5.4, 5.4)):
    if players_data.empty:
        return None

    categories = EVAL_CATEGORIES
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.4, 5.0))
    fig.patch.set_facecolor("#07100b")
    ax = fig.add_axes([0.05, 0.05, 0.58, 0.90], projection="polar")
    ax.set_facecolor("#0d160f")

    colors = ["#36c782", "#f2c66d", "#6aa9ff", "#ff7a7a", "#b88cff"]
    for index, (_, player) in enumerate(players_data.iterrows()):
        values = player[categories].tolist()
        values += values[:1]
        color = colors[index % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2.6, label=player["Name"], color=color)
        ax.fill(angles, values, alpha=0.13, color=color)

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#718276", fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 10)
    ax.grid(color="#2d4436", linestyle="-", linewidth=0.75, alpha=0.85)
    ax.spines["polar"].set_color("#496253")
    ax.spines["polar"].set_linewidth(1.2)

    outline = [path_effects.Stroke(linewidth=2.4, foreground="#07100b"), path_effects.Normal()]
    for angle, category in zip(angles[:-1], categories):
        ax.text(
            angle,
            10.95,
            category,
            horizontalalignment="center",
            verticalalignment="center",
            size=9,
            color="#f4f1e8",
            path_effects=outline,
            zorder=10,
        )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.45, 0.5),
        ncol=1,
        labelcolor="#f4f1e8",
        fontsize=11,
        frameon=True,
        facecolor="#101610",
        edgecolor="#2d4436",
        borderpad=0.8,
        labelspacing=0.7,
    )
    return fig


def figure_to_svg(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=False, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue().decode("utf-8")
