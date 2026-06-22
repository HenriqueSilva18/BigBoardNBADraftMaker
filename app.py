from html import escape
from textwrap import dedent

import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items

from big_board_app.config import (
    APP_SUBTITLE,
    APP_TITLE,
    BOARD_COLUMNS,
    EVAL_CATEGORIES,
    RANK_COLUMN,
    SCORE_COLUMN,
    WEIGHTS,
)
from big_board_app.data import get_player_info, get_player_stats, load_prospect_data
from big_board_app.export_image import board_to_png_bytes
from big_board_app.scoring import calculate_weighted_average, get_tier
from big_board_app.storage import (
    big_board_to_json_bytes,
    create_empty_board,
    load_big_board_from_json,
    normalize_big_board,
    order_big_board,
    save_big_board_to_file,
    save_big_board_to_txt,
)
from big_board_app.theme import inject_theme
from big_board_app.visuals import (
    apply_extreme_highlighting,
    build_eval_table,
    build_stats_table,
    create_overlaid_radar_chart,
    figure_to_svg,
)


st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🏀")
inject_theme()


def cached_prospect_data():
    return load_prospect_data()


def safe_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return escape(str(value))


def html(content):
    cleaned = "\n".join(line.strip() for line in dedent(content).strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def get_board():
    return normalize_big_board(st.session_state.big_board)


def set_board(board):
    st.session_state.big_board = normalize_big_board(board)


def initialize_state():
    if "big_board" not in st.session_state:
        st.session_state.big_board = create_empty_board()

    for category in EVAL_CATEGORIES:
        st.session_state.setdefault(f"slider_{category}", 5)

    st.session_state.setdefault("player_select", "")
    st.session_state.setdefault("last_selected_player", "")
    st.session_state.setdefault("auto_loaded", False)
    st.session_state.setdefault("last_uploaded_board", "")


def auto_load_saved_board():
    if st.session_state.auto_loaded:
        return

    try:
        saved_board = load_big_board_from_json()
    except Exception as error:
        st.warning(f"Saved board could not be loaded: {error}")
        saved_board = None

    if saved_board is not None:
        set_board(saved_board)

    st.session_state.auto_loaded = True


def reset_sliders():
    for category in EVAL_CATEGORIES:
        st.session_state[f"slider_{category}"] = 5


def apply_pending_editor_resets():
    if st.session_state.get("reset_sliders_flag", False):
        reset_sliders()
        st.session_state.reset_sliders_flag = False

    if st.session_state.get("clear_selection", False):
        st.session_state.player_select = ""
        st.session_state.last_selected_player = ""
        reset_sliders()
        st.session_state.clear_selection = False


def render_hero(df):
    board = get_board()
    prospect_count = 0 if df is None else len(df)
    board_count = len(board)
    top_name = "No board yet"

    if not board.empty:
        top_name = order_big_board(board).iloc[0].get("Name", "No board yet")

    html(
        f"""
        <section class="draft-hero">
            <div class="hero-kicker">Scouting workspace</div>
            <h1>{APP_TITLE} by Ric</h1>
            <p>{APP_SUBTITLE}</p>
        </section>
        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Prospects loaded</div>
                <div class="kpi-value">{prospect_count}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Players on board</div>
                <div class="kpi-value">{board_count}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Draft class</div>
                <div class="kpi-value">2026</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Current 1st</div>
                <div class="kpi-value">{safe_text(top_name)}</div>
            </div>
        </section>
        """
    )


def render_sidebar():
    board = get_board()

    with st.sidebar:
        st.markdown("### Board controls")

        if board.empty:
            st.info("Add your first player to enable exports.")
        else:
            st.download_button(
                label="Download full backup",
                data=big_board_to_json_bytes(board),
                file_name="big_board_backup_2026.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                label="Download rankings text",
                data=save_big_board_to_txt(board).encode("utf-8"),
                file_name="big_board_rankings_2026.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.download_button(
                label="Download board PNG",
                data=board_to_png_bytes(board),
                file_name="draft_room_2026_board.png",
                mime="image/png",
                use_container_width=True,
            )

        uploaded_file = st.file_uploader("Load JSON backup", type=["json"])
        if uploaded_file is not None:
            fingerprint = f"{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.last_uploaded_board != fingerprint:
                try:
                    loaded_board = load_big_board_from_json(fileobj=uploaded_file)
                    if loaded_board is None or loaded_board.empty:
                        st.error("Could not read the uploaded JSON file.")
                    else:
                        set_board(loaded_board)
                        save_big_board_to_file(loaded_board)
                        st.session_state.last_uploaded_board = fingerprint
                        st.success("Big Board loaded successfully.")
                except Exception as error:
                    st.error(f"Could not load backup: {error}")

        st.markdown("---")
        st.markdown("### Evaluation weights")
        for category, weight in WEIGHTS.items():
            st.caption(f"{category}: {weight * 100:.0f}%")


def render_player_profile(player_info):
    html(
        f"""
        <div class="player-chip">
            <span>Age: {safe_text(player_info["age"])}</span>
            <span>Ht/Ws: {safe_text(player_info["measurements"])}</span>
            <span>Position: {safe_text(player_info["position"])}</span>
            <span>Team: {safe_text(player_info["team"])}</span>
        </div>
        """
    )


def select_player(df):
    if df is None or df.empty:
        return st.text_input("Player name", key="player_select")

    available_players = sorted(df["name"].dropna().unique())
    options = [""] + available_players

    if st.session_state.player_select not in options:
        st.session_state.player_select = ""

    return st.selectbox("Select player to evaluate", options, key="player_select")


def hydrate_existing_player(name, board):
    if not name or board.empty or name not in board["Name"].values:
        if st.session_state.last_selected_player != name:
            st.session_state.last_selected_player = name
        return False, None

    existing_player = board[board["Name"] == name].iloc[0]

    if st.session_state.last_selected_player != name:
        for category in EVAL_CATEGORIES:
            st.session_state[f"slider_{category}"] = int(existing_player[category])
        st.session_state.last_selected_player = name

    return True, existing_player


def render_sliders():
    scores = {}
    columns = st.columns(3)

    for index, category in enumerate(EVAL_CATEGORIES):
        with columns[index % 3]:
            scores[category] = st.slider(
                f"{category} ({WEIGHTS[category] * 100:.0f}%)",
                min_value=0,
                max_value=10,
                key=f"slider_{category}",
            )

    return scores


def build_player_record(df, name, scores, score, tier):
    player_info = get_player_info(df, name)
    return {
        "Name": name,
        "Age": player_info["age"],
        "Measurements": player_info["measurements"],
        "Position": player_info["position"],
        "College/Team": player_info["team"],
        SCORE_COLUMN: score,
        "Tier": tier,
        **scores,
    }


def save_player_record(player_record, is_existing_player):
    board = get_board()

    if is_existing_player:
        mask = board["Name"] == player_record["Name"]
        for column, value in player_record.items():
            board.loc[mask, column] = value
    else:
        player_record[RANK_COLUMN] = len(board) + 1
        board = pd.concat([board, pd.DataFrame([player_record])], ignore_index=True)

    set_board(board)
    save_big_board_to_file(st.session_state.big_board)


def remove_player(name):
    board = get_board()
    set_board(board[board["Name"] != name])
    save_big_board_to_file(st.session_state.big_board)
    st.session_state.reset_sliders_flag = True


def render_player_editor(df):
    apply_pending_editor_resets()

    html('<div class="section-label">Evaluate</div>')
    st.subheader("Scout card")

    board = get_board()
    name = select_player(df)
    is_existing_player, existing_player = hydrate_existing_player(name, board)

    if is_existing_player:
        current_board = order_big_board(board).reset_index(drop=True)
        current_rank = current_board[current_board["Name"] == name].index[0] + 1
        current_score = float(existing_player[SCORE_COLUMN])
        st.warning(
            f"{name} is already on your Big Board. Adjust the grades and update the player."
        )
        st.info(f"Current ranking: #{current_rank} | Current score: {current_score:.2f}/10")

    if name:
        render_player_profile(get_player_info(df, name))

    st.markdown("**Evaluation categories**")
    scores = render_sliders()
    score_preview = calculate_weighted_average(scores)
    tier_preview = get_tier(score_preview)

    preview_cols = st.columns(2)
    preview_cols[0].metric("Score preview", f"{score_preview}/10", tier_preview)

    if is_existing_player:
        current_score = float(existing_player[SCORE_COLUMN])
        score_change = score_preview - current_score
        delta_text = "No change" if score_change == 0 else f"{score_change:+.2f}"
        preview_cols[1].metric("Saved score", f"{current_score:.2f}/10", delta_text)

    with st.form("player_form"):
        button_label = f"Update {name}" if is_existing_player else "Add to Big Board"
        button_help = "Save this evaluation to the board."

        if is_existing_player:
            button_cols = st.columns([3, 1])
            submitted = button_cols[0].form_submit_button(
                button_label,
                type="primary",
                help=button_help,
                disabled=not bool(name),
                use_container_width=True,
            )
            remove_submitted = button_cols[1].form_submit_button(
                "Remove",
                type="secondary",
                disabled=not bool(name),
                use_container_width=True,
            )
        else:
            submitted = st.form_submit_button(
                button_label,
                type="primary",
                help=button_help,
                disabled=not bool(name),
                use_container_width=True,
            )
            remove_submitted = False

    if submitted and name:
        player_record = build_player_record(df, name, scores, score_preview, tier_preview)
        save_player_record(player_record, is_existing_player)
        st.success(f"{name} was {'updated' if is_existing_player else 'added'} successfully.")
        st.session_state.last_selected_player = ""
        st.session_state.reset_sliders_flag = True
        st.session_state.clear_selection = True
        st.rerun()

    if remove_submitted and name:
        remove_player(name)
        st.success(f"{name} was removed from the Big Board.")
        st.rerun()


def render_live_board():
    board = get_board()

    html('<div class="section-label">Live board</div>')
    st.subheader("Draft order")

    if board.empty:
        html(
            """
            <div class="empty-state">
                No players yet. Your ranked board will appear here while you grade prospects.
            </div>
            """
        )
        return

    rows = []
    ranked_board = order_big_board(board).reset_index(drop=True)
    for index, row in ranked_board.iterrows():
        rank = int(row.get(RANK_COLUMN, index + 1))
        name = safe_text(row.get("Name", "N/A"))
        position = safe_text(row.get("Position", "N/A"))
        team = safe_text(row.get("College/Team", "N/A"))
        age = safe_text(row.get("Age", "N/A"))
        measurements = safe_text(row.get("Measurements", "N/A"))
        score = float(row.get(SCORE_COLUMN, 0))
        rows.append(f"""
        <div class="board-row">
            <div class="board-rank">#{rank}</div>
            <div>
                <div class="board-name">{name}</div>
                <div class="board-meta">{position} - {team} - {age} - {measurements}</div>
            </div>
            <div class="board-score">{score:.2f}</div>
        </div>
        """)

    html(
        f"""
        <div class="board-card">
        {''.join(rows)}
        </div>
        """
    )


def get_display_board():
    board = get_board()
    if board.empty:
        return board

    display_board = order_big_board(board).reset_index(drop=True)
    return display_board[BOARD_COLUMNS]


def sortable_label(row):
    score = float(row.get(SCORE_COLUMN, 0))
    info = " - ".join(
        str(value)
        for value in [row.get("Position", "N/A"), row.get("College/Team", "N/A")]
        if pd.notna(value) and str(value) != "N/A"
    )
    label = f"{row['Name']} | {info}" if info else str(row["Name"])
    return f"{label} | {score:.2f}"


def render_drag_board(display_board):
    label_to_name = {
        sortable_label(row): row["Name"]
        for _, row in display_board.iterrows()
    }
    items = list(label_to_name.keys())
    custom_style = """
    .sortable-component {
        background: transparent;
    }
    .sortable-container {
        background: #0d120f;
        border: 1px solid #2c3a31;
        border-radius: 8px;
        padding: 0.75rem;
        max-height: 640px;
        overflow-y: auto;
    }
    .sortable-container-header {
        display: none;
    }
    .sortable-container-body {
        background: transparent;
        counter-reset: board-rank;
    }
    .sortable-item {
        background: linear-gradient(180deg, #151d17, #101610);
        border: 1px solid #2d4436;
        border-radius: 8px;
        box-sizing: border-box;
        color: #f4f1e8;
        cursor: grab;
        display: flex;
        align-items: center;
        font-family: Inter, Segoe UI, Arial, sans-serif;
        font-size: 0.96rem;
        font-weight: 700;
        min-height: 56px;
        margin-bottom: 0.55rem;
        padding: 0.75rem 0.85rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
        transform: none !important;
        transition: border-color 120ms ease, box-shadow 120ms ease;
        width: 100%;
    }
    .sortable-item::before {
        color: #f2c66d;
        content: "#" counter(board-rank) "  ";
        counter-increment: board-rank;
        font-weight: 900;
    }
    .sortable-item:hover {
        border-color: #36c782;
        color: #ffffff;
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
        min-height: 56px;
        padding: 0.75rem 0.85rem;
        transform: none !important;
        width: 100%;
    }
    .sortable-item:active {
        cursor: grabbing;
        transform: none !important;
    }
    """
    sorted_items = sort_items(
        items,
        direction="vertical",
        custom_style=custom_style,
        key=f"drag_big_board_order_{hash(tuple(items))}",
    )
    sorted_names = [label_to_name[item] for item in sorted_items if item in label_to_name]

    if sorted_names != display_board["Name"].tolist():
        board = get_board()
        rank_lookup = {name: index + 1 for index, name in enumerate(sorted_names)}
        board[RANK_COLUMN] = board["Name"].map(rank_lookup).fillna(board[RANK_COLUMN])
        set_board(board)
        save_big_board_to_file(st.session_state.big_board)
        st.rerun()


def render_tier_editor(display_board):
    tier_df = display_board[["Name", "Tier"]].copy()
    edited_df = st.data_editor(
        tier_df,
        use_container_width=True,
        height=300,
        hide_index=True,
        disabled=["Name"],
    )

    current_tiers = tier_df.set_index("Name")["Tier"].astype(str)
    edited_tiers = edited_df.set_index("Name")["Tier"].astype(str)
    if not edited_tiers.equals(current_tiers):
        board = get_board()
        board["Tier"] = board["Name"].map(edited_tiers).fillna(board["Tier"])
        set_board(board)
        save_big_board_to_file(st.session_state.big_board)
        st.rerun()


def render_rankings():
    html('<div class="section-label">Rankings</div>')
    st.subheader("Drag Big Board")

    display_board = get_display_board()
    if display_board.empty:
        html(
            """
            <div class="empty-state">
                Your board is empty. Pick a player above, grade the categories, and start building.
            </div>
            """
        )
        return

    st.caption("Drag players up or down to change the board order. The order is saved automatically.")
    render_drag_board(display_board)

    with st.expander("Tier overrides"):
        render_tier_editor(display_board)

    st.download_button(
        label="Download pretty board PNG",
        data=board_to_png_bytes(get_board()),
        file_name="draft_room_2026_board.png",
        mime="image/png",
        use_container_width=True,
    )


def render_stats_comparison(df, selected_players):
    html('<div class="compare-subtitle">Stat profile</div>')
    stats_records = [get_player_stats(df, name) for name in selected_players]

    if df is None or not any(stats_records):
        st.warning("Player stats data is not available.")
        return

    display_df, numeric_df = build_stats_table(stats_records, selected_players)
    if display_df.empty:
        st.warning("There are no stats available for this selection.")
        return

    render_styled_comparison_table(display_df, numeric_df, lower_is_better={"TO/36"})


def render_radar_chart(comparison_data, selected_players):
    if len(selected_players) > 4:
        st.warning("Select up to 4 players for the radar chart.")
        return

    html('<div class="compare-subtitle">Skill shape</div>')
    fig = create_overlaid_radar_chart(comparison_data, (5.4, 5.4))
    if fig is None:
        st.warning("No radar data available.")
        return

    st.image(figure_to_svg(fig), width=860)


def render_eval_comparison(comparison_data):
    html('<div class="compare-subtitle">Grade matrix</div>')
    eval_table, numeric_df = build_eval_table(comparison_data)
    render_styled_comparison_table(eval_table, numeric_df)


def style_comparison_table(display_df, numeric_df, lower_is_better=()):
    highlights = apply_extreme_highlighting(display_df, numeric_df, lower_is_better=lower_is_better)
    return (
        display_df.style
        .hide(axis="index")
        .set_properties(
            **{
                "background-color": "#0d120f",
                "border-color": "#2d4436",
                "color": "#f4f1e8",
                "font-weight": "650",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#151d17"),
                        ("color", "#b6c2b8"),
                        ("font-weight", "800"),
                        ("border-color", "#2d4436"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-color", "#2d4436"),
                        ("padding", "0.55rem 0.65rem"),
                    ],
                },
            ]
        )
        .apply(lambda _: highlights, axis=None)
    )


def render_styled_comparison_table(display_df, numeric_df, lower_is_better=()):
    styled = style_comparison_table(display_df, numeric_df, lower_is_better=lower_is_better)
    table_html = styled.to_html()
    st.markdown(f'<div class="comparison-table-wrap">{table_html}</div>', unsafe_allow_html=True)


def best_category(row):
    grades = pd.to_numeric(row[EVAL_CATEGORIES], errors="coerce")
    if grades.dropna().empty:
        return "N/A"
    return grades.idxmax()


def render_compare_cards(comparison_data):
    cards = []
    for _, row in comparison_data.iterrows():
        rank = int(row.get(RANK_COLUMN, 0))
        name = safe_text(row.get("Name", "N/A"))
        position = safe_text(row.get("Position", "N/A"))
        team = safe_text(row.get("College/Team", "N/A"))
        age = safe_text(row.get("Age", "N/A"))
        measurements = safe_text(row.get("Measurements", "N/A"))
        score = float(row.get(SCORE_COLUMN, 0))
        strongest = safe_text(best_category(row))
        cards.append(
            f"""
            <div class="compare-card">
                <div class="compare-rank">#{rank}</div>
                <div class="compare-name">{name}</div>
                <div class="compare-meta">{position} - {team}</div>
                <div class="compare-score">{score:.2f}</div>
                <div class="compare-chip-row">
                    <span>Age {age}</span>
                    <span>{measurements}</span>
                    <span>Best: {strongest}</span>
                </div>
            </div>
            """
        )

    html(
        f"""
        <div class="compare-card-grid">
            {''.join(cards)}
        </div>
        """
    )


def render_comparison(df):
    board = get_board()
    if board.empty:
        return

    html('<div class="section-label">Compare</div>')
    st.subheader("Comparison room")

    ordered_board = order_big_board(board)
    players = ordered_board["Name"].tolist()
    default_players = players[:2] if len(players) >= 2 else []

    selected_players = st.multiselect(
        "Players to compare",
        players,
        default=default_players,
        max_selections=5,
    )

    if len(selected_players) < 2:
        st.info("Select at least two players to compare.")
        return

    comparison_data = (
        ordered_board.set_index("Name")
        .loc[selected_players]
        .reset_index()
    )
    render_compare_cards(comparison_data)

    stats_tab, radar_tab, grades_tab = st.tabs(["Stats", "Radar", "Grades"])
    with stats_tab:
        render_stats_comparison(df, selected_players)
    with radar_tab:
        render_radar_chart(comparison_data, selected_players)
    with grades_tab:
        render_eval_comparison(comparison_data)


def render_clear_board():
    board = get_board()
    if board.empty:
        return

    html('<div class="section-label">Reset</div>')
    st.subheader("Danger zone")
    if st.button("Clear all players", type="secondary"):
        set_board(create_empty_board())
        save_big_board_to_file(st.session_state.big_board)
        st.success("Big Board has been cleared.")
        st.rerun()


def main():
    initialize_state()
    auto_load_saved_board()

    try:
        df = cached_prospect_data()
    except FileNotFoundError:
        st.error("Could not find nba_prospects_2026_stats.csv in the project root.")
        df = None
    except Exception as error:
        st.error(f"Could not load prospect data: {error}")
        df = None

    render_hero(df)
    render_sidebar()

    editor_col, board_col = st.columns([1.04, 0.96], gap="large")
    with editor_col:
        render_player_editor(df)
    with board_col:
        render_live_board()

    rankings_tab, comparison_tab, reset_tab = st.tabs(["Big Board", "Compare", "Reset"])
    with rankings_tab:
        render_rankings()
    with comparison_tab:
        render_comparison(df)
    with reset_tab:
        render_clear_board()


if __name__ == "__main__":
    main()
