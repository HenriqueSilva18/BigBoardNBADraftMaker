import io

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects
import os
from io import BytesIO

# --- Configuration and Setup ---
st.set_page_config(page_title="NBA Draft Big Board", layout="wide", page_icon="🏀")

# --- Constants for new evaluation categories ---
EVAL_CATEGORIES = [
    "Athleticism", "Dribbling", "Shooting", "Perimeter Defense", "Passing",
    "Scoring", "Interior Defense", "Basketball IQ", "Intangibles"
]

# Define weights for the new categories
WEIGHTS = {
    "Athleticism": 1/9,
    "Scoring": 1/9,
    "Shooting": 1/9,
    "Dribbling": 1/9,
    "Passing": 1/9,
    "Perimeter Defense": 1/9,
    "Interior Defense": 1/9,
    "Basketball IQ": 1/9,
    "Intangibles": 1/9
}


# --- Data Loading and Processing ---
def load_data():
    try:
        # Lista de colunas padronizadas (na ordem desejada)
        columns = [
            'name','team','year','position','measurements','weight','mock_draft','big_board','age_at_draft','birthdate',
            'nation','hometown','high_school','espn_100','strengths','weaknesses','max_vert','lane_agil','shuttle','3_4sprint',
            'reach','wingspan','games','minutes_per_game','fgm_fga','fg_pct','3pm_3pa','3p_pct','ftm_fta','ft_pct','rebounds_pg',
            'assists_pg','blocks_pg','steals_pg','turnovers_pg','personal_fouls_pg','points_pg','games_per36','minutes_per36',
            'fgm_fga_per36','fg_pct_per36','3pm_3pa_per36','3p_pct_per36','ftm_fta_per36','ft_pct_per36','rebounds_per36',
            'assists_per36','blocks_per36','steals_per36','turnovers_per36','personal_fouls_per36','points_per36','ts_per',
            'efg_per','3pa_rate','fta_rate','nba_3p_per','usg_per','ast_per_usg','ast_per_to','per','ows_per_40','dws_per_40',
            'ws_per_40','ortg','drtg','obpm','dbpm','bpm'
        ]

        desired_collums = [
            'name', 'team', 'year', 'position','measurements', 'age_at_draft',
            'nation', 'wingspan', 'games','minutes_per_game', '3p_pct', 'ft_pct', 'rebounds_per36', 'assists_per36', 'blocks_per36',
            'steals_per36', 'turnovers_per36','points_per36', 'ts_per', '3pa_rate', 'fta_rate', 'usg_per', 'ast_per_usg', 'ast_per_to', 'obpm', 'dbpm', 'bpm'
        ]

        # Tenta ler o CSV já com o cabeçalho certo
        df = pd.read_csv('nba_prospects_2025_stats.csv')

        # (Se o arquivo não tiver header, use:)
        # df = pd.read_csv('players_ncaa_2025.csv', header=None, names=columns)

        # Caso o CSV tenha colunas extras ou outra ordem, filtra só as desejadas disponíveis
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]

        #colocar apenas desired_columns
        df = df[desired_collums]

        return df

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# --- Player Information and Stats Retrieval ---
def get_player_info(df, player_name):
    """Get basic player information."""
    if df is None or player_name not in df['name'].values:
        return {"idade": "N/A", "altura": "N/A", "posicao": "N/A", "equipa": "N/A"}
    player_data = df[df['name'] == player_name].iloc[0]
    return {
        "idade": player_data.get('age_at_draft', 'N/A'),
        "alt_ws": player_data.get('measurements', 'N/A'),
        "posicao": player_data.get('position', 'N/A'),
        "equipa": player_data.get('team', 'N/A')
    }


def get_player_stats(df, player_name):
    """Get detailed player statistics for comparison."""
    if df is None or player_name not in df['name'].values:
        return {}
    player_data = df[df['name'] == player_name].iloc[0]

    desired_collums = [
        'name', 'team', 'year', 'position', 'measurements', 'age_at_draft',
        'nation', 'wingspan', 'games','minutes_per_game', '3p_pct', 'ft_pct', 'rebounds_per36', 'assists_per36', 'blocks_per36',
        'steals_per36', 'points_per36', 'ts_per', '3pa_rate', 'fta_rate', 'usg_per', 'ast_per_usg', 'ast_per_to',
        'obpm', 'dbpm', 'bpm'
    ]

    stats_columns = ['games', 'points_per36', 'assists_per36', 'rebounds_per36', 'steals_per36', 'blocks_per36','turnovers_per36', 'obpm', 'dbpm', 'bpm',
                     'usg_per', 'ts_per', 'ast_per_to', '3p_pct', '3pa_rate', 'fta_rate']
    stats = {col: player_data.get(col, 0) for col in stats_columns}
    return stats


# --- Core Logic for Big Board ---
def calculate_weighted_average(scores):
    """Calculate the weighted average score based on new categories."""
    total_score = sum(scores[cat] * WEIGHTS[cat] for cat in EVAL_CATEGORIES)
    return round(total_score, 2)

def apply_highlighting_list(df_display, df_numeric):
    styles = pd.DataFrame('', index=df_display.index, columns=df_display.columns)
    for col in df_numeric.columns:
        if not df_numeric[col].dropna().empty:
            max_val = df_numeric[col].max()
            min_val = df_numeric[col].min()
            for index in df_numeric.index:
                if pd.notna(df_numeric.loc[index, col]):
                    if df_numeric.loc[index, col] == max_val and index != 'Name':
                        styles.loc[index, col] = 'background-color: #28a745'
                    elif df_numeric.loc[index, col] == min_val:
                        styles.loc[index, col] = 'background-color: #dc3545'
    return styles

 # Function to apply styles based on numeric_df
def apply_highlighting(df_display, df_numeric, cols_max, cols_min):
    styles = pd.DataFrame('', index=df_display.index, columns=df_display.columns)

    # Highlight max (green) and min (red) for columns_to_highlight
    for col in cols_max:
        if col in df_numeric.columns:
            max_idx = df_numeric[col].idxmax()
            min_idx = df_numeric[col].idxmin()
            if pd.notnull(max_idx):
                styles.loc[max_idx, col] = 'background-color: #28a745'
            if pd.notnull(min_idx):
                styles.loc[min_idx, col] = 'background-color: #dc3545'

    # Highlight min (green) and max (red) for TO/36 and BPM
    for col in cols_min:
        if col in df_numeric.columns:
            max_idx = df_numeric[col].idxmax()
            min_idx = df_numeric[col].idxmin()
            if pd.notnull(max_idx):
                styles.loc[max_idx, col] = 'background-color: #dc3545'
            if pd.notnull(min_idx):
                styles.loc[min_idx, col] = 'background-color: #28a745'

    return styles


def get_tier(score):
    """Assign a tier based on the weighted score."""
    if score >= 9.0: return "Tier 1 - Superstar"
    if score >= 8.0: return "Tier 2 - Potential All-NBA"
    if score >= 7.0: return "Tier 3 - Potential All-Star"
    if score >= 6.0: return "Tier 4 - Starter"
    return "Tier 5 - Role Player"


# --- Visualization ---
def create_overlaid_radar_chart(players_data, figsize):
    """Cria um gráfico de radar sobreposto e moderno para comparação de jogadores."""
    if players_data.empty:
        return None

    categories = EVAL_CATEGORIES
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize = figsize, subplot_kw=dict(projection='polar'))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (idx, player) in enumerate(players_data.iterrows()):
        values = player[categories].tolist()
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, label=player['Name'], color=color)
        ax.fill(angles, values, alpha=0.2, color=color)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, color='white')
    outline_effect = [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()]
    for label in ax.get_xticklabels():
        # Aplica o contorno
        label.set_path_effects(outline_effect)
        # NOVO: Define um zorder alto para o texto ficar por cima de tudo
        label.set_zorder(10)
    ax.set_ylim(0, 10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), labelcolor='black', fontsize=10, markerscale=0.8,
              handletextpad=0.5, labelspacing=0.8)
    ax.grid(color='#CCCCCC', linestyle='--', linewidth=0.5)

    spine = ax.spines['polar']

    spine.set_color('#777777')
    spine.set_zorder(0.5)
    return fig

#only save rank number, the name and the position
def save_big_board_to_txt(big_board, filename="big_board_nba_draft_2025.txt"):
    try:
        with open(filename, 'w') as f:
            f.write ("🏀 NBA Draft Big Board 2025 Rankings\n\n")
            for index, row in big_board.iterrows():
                rank = index + 1  # Rank starts at 1
                name = row['Name']
                position = row['Position']
                f.write(f"{rank}. {name} - {position}\n")
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

# --- File Operations ---
def save_big_board_to_file(big_board, filename="big_board_save.json"):
    """Save the big board DataFrame to a JSON file."""
    try:
        big_board.to_json(filename, orient='records', indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


def load_big_board_from_file(filename="big_board_save.json"):
    """Load the big board from a JSON file, ensuring all new columns exist."""
    try:
        if os.path.exists(filename):
            board = pd.read_json(filename, orient='records')
            for col in st.session_state.big_board.columns:
                if col not in board.columns:
                    board[col] = 5 if col in EVAL_CATEGORIES else 'N/A'
            return board
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# --- Main App ---
st.title("🏀 NBA Draft Big Board 2025")

df = load_data()

# --- Initialize Session State ---
if "big_board" not in st.session_state:
    columns = [
                  "Name", "Age", "Height", "Position", "College/Team", "Tier", "Média Ponderada"
              ] + EVAL_CATEGORIES
    st.session_state.big_board = pd.DataFrame(columns=columns)

for cat in EVAL_CATEGORIES:
    if f"slider_{cat}" not in st.session_state:
        st.session_state[f"slider_{cat}"] = 5

if 'player_select' not in st.session_state:
    st.session_state.player_select = ""

if "auto_loaded" not in st.session_state:
    saved_board = load_big_board_from_file()
    if saved_board is not None:
        st.session_state.big_board = saved_board
    st.session_state.auto_loaded = True

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎛️ Controls")
    st.subheader("💾 Save/Load")
    if st.button("💾 Save", use_container_width=True):
        if save_big_board_to_txt(st.session_state.big_board):
            st.success("Big Board Saved!")

    uploaded_file = st.file_uploader("📁 Load", type=['json'])
    if uploaded_file:
        loaded_board = pd.read_json(uploaded_file)
        for col in st.session_state.big_board.columns:
            if col not in loaded_board.columns:
                loaded_board[col] = 5 if col in EVAL_CATEGORIES else 'N/A'
        st.session_state.big_board = loaded_board
        st.success("Big Board Loaded!")
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Evaluation Weights")
    for cat, weight in WEIGHTS.items():
        st.text(f"• {cat}: {int(weight * 100)}%")


# --- Callbacks ---
def reset_sliders():
    """Callback to reset all evaluation sliders to their default value."""
    for category in EVAL_CATEGORIES:
        st.session_state[f"slider_{category}"] = 5


# --- Add Player Section (CORRECTED) ---
st.subheader("➕ Add Player")

# --- SOLUTION: Player selection is now OUTSIDE the form ---
# This allows us to use the on_change callback without erroring.
if df is not None:
    available_players = sorted(df['name'].dropna().unique())
    st.selectbox(
        "Select Player to Evaluate",
        [""] + available_players,
        on_change=reset_sliders,
        key="player_select"  # The selected value is stored in st.session_state.player_select
    )
else:
    st.text_input("Player Name", key="player_select")

# The form now only contains the sliders and the submit button.
with st.form("form_jogador"):
    # We get the player name from session_state, which was set by the selectbox above.
    nome = st.session_state.player_select

    if df is not None and nome:
        player_info = get_player_info(df, nome)
        st.caption(
            f"Age: {player_info['idade']} | Ht & Ws: {player_info['alt_ws']} | Position: {player_info['posicao']} | Team: {player_info['equipa']}")

    scores = {}
    cols = st.columns(3)
    for i, category in enumerate(EVAL_CATEGORIES):
        with cols[i % 3]:
            # The key for each slider now uses the session_state value
            scores[category] = st.slider(
                f"{category}",
                0, 10, key=f"slider_{category}"
            )

    st.markdown("---")
    media_preview = calculate_weighted_average(scores)
    tier_preview = get_tier(media_preview)

    submit_col, preview_col = st.columns([1, 2])
    with submit_col:
        submitted = st.form_submit_button("➕ Add to Big Board", type="primary", use_container_width=True)
    with preview_col:
        st.metric("Score Preview", f"{media_preview}/10", delta=tier_preview)

    if submitted and nome:
        if nome in st.session_state.big_board['Name'].values:
            st.error(f"⚠️ {nome} is already on the Big Board!")
        else:
            player_info = get_player_info(df, nome)
            new_player_data = {
                "Name": nome,
                "Age": player_info['idade'],
                "Height": player_info['alt_ws'],
                "Position": player_info['posicao'],
                "College/Team": player_info['equipa'],
                "Média Ponderada": media_preview,
                "Tier": tier_preview,
                **scores
            }
            new_player_df = pd.DataFrame([new_player_data])
            st.session_state.big_board = pd.concat([st.session_state.big_board, new_player_df], ignore_index=True)
            save_big_board_to_file(st.session_state.big_board)
            st.success(f"✅ {nome} added to Big Board!")
            st.rerun()

# --- Big Board Rankings Display ---
st.markdown("---")
st.subheader("📋 Big Board Rankings")

if not st.session_state.big_board.empty:
    display_board = st.session_state.big_board.sort_values(by="Média Ponderada", ascending=False).reset_index(drop=True)

    # Defina o índice para começar em 1
    display_board.index = display_board.index + 1
    display_board.index.name = "Rank"

    display_cols = ["Name", "Tier", "Média Ponderada", "Age", "Height", "Position", "College/Team"] + EVAL_CATEGORIES
    display_board = display_board[display_cols]

    st.info("💡 You can double-click a 'Tier' cell to override it.")
    edited_df = st.data_editor(
        display_board,
        use_container_width=True,
        height=400,
        disabled=display_board.columns.drop("Tier"),
        column_config={
            "Média Ponderada": st.column_config.ProgressColumn(
                "Score", format="%.2f", min_value=0, max_value=10,
            ),
        }
    )

    if not edited_df.equals(display_board):
        st.session_state.big_board = pd.merge(
            st.session_state.big_board.drop(columns=['Tier']),
            edited_df[['Name', 'Tier']],
            on='Name',
            how='left'
        )
        save_big_board_to_file(st.session_state.big_board)
        st.rerun()
    # add button to save big board to txt file
    if st.button("💾 Save Big Board to Text File"):
        if save_big_board_to_txt(st.session_state.big_board):
            st.success("Big Board saved to text file!")
        else:
            st.error("Failed to save Big Board to text file.")
else:
    st.info("👆 Add your first player to get started!")

# --- Player Comparison and Analysis ---
if not st.session_state.big_board.empty:
    st.markdown("---")
    st.subheader("🔍 Player Comparison")

    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        players = st.session_state.big_board['Name'].tolist()
        selected_players = st.multiselect(
            "Select 2-5 players to compare:",
            players,
            default=players[:2] if len(players) >= 2 else [],
            max_selections=10
        )
    with col_comp2:
        comparison_type = st.radio(
            "Comparison Type:",
            ["Stats Comparison", "Radar Chart (Max. 4 players)", "Table Comparison"],
            horizontal=True
        )

    if len(selected_players) >= 2:
        comparison_data = st.session_state.big_board[st.session_state.big_board['Name'].isin(selected_players)]

        if comparison_type == "Stats Comparison":
            st.markdown("**📊 Basketball Statistics Comparison**")
            stats_data = [get_player_stats(df, name) for name in selected_players]

            if df is not None and any(stats_data):
                stats_df = pd.DataFrame(stats_data)
                stats_df.insert(0, 'Name', selected_players)
                stats_df.set_index('Name', inplace=True)

                stats_columns = ['games', 'points_per36', 'assists_per36', 'rebounds_per36', 'steals_per36',
                                 'blocks_per36', 'turnovers_per36', 'obpm', 'dbpm', 'bpm',
                                 'usg_per', 'ts_per', 'ast_per_to', '3p_pct', '3pa_rate', 'fta_rate']

                rename_map = {'games': 'GP',
                              'points_per36': 'PTS/36',
                              'assists_per36': 'AST/36',
                              'rebounds_per36': 'REB/36',
                              'steals_per36': 'STL/36',
                              'blocks_per36': 'BLK/36',
                              'turnovers_per36': 'TO/36',
                              'obpm': 'OBPM',
                              'dbpm': 'DBPM',
                              'bpm': 'BPM',
                              'usg_per': 'USG%',
                              'ts_per': 'TS%',
                              'ast_per_to': 'AST/TO',
                              '3p_pct': '3PT%',
                              '3pa_rate': '3PA Rate',
                              'fta_rate': 'FTA Rate'
                              }

                # Select only available columns and rename
                available_rename_keys = [key for key in rename_map.keys() if key in stats_df.columns]
                display_stats_df = stats_df[available_rename_keys].rename(columns=rename_map)

                # Preprocess to handle empty or non-numeric values
                display_stats_df = display_stats_df.replace(['', '-', 'N/A', None], np.nan)

                # Create a numeric DataFrame for computations
                numeric_df = display_stats_df.apply(pd.to_numeric, errors='coerce')

                # Apply formatting to display_stats_df
                for col in ['TS%', '3PT%']:
                    if col in display_stats_df.columns:
                        display_stats_df[col] = display_stats_df[col].apply(
                            lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else '-')

                if 'USG%' in display_stats_df.columns:
                    display_stats_df['USG%'] = display_stats_df['USG%'].apply(
                        lambda x: f"{x:.1f}%" if pd.notnull(x) else '-')

                for col in display_stats_df.columns:
                    if col in ['3PA Rate', 'FTA Rate']:
                        display_stats_df[col] = display_stats_df[col].apply(
                            lambda x: f"{x:.3f}" if pd.notnull(x) else '-')
                    elif col in ['PTS/36', 'AST/36', 'REB/36', 'STL/36', 'BLK/36', 'TO/36', 'OBPM', 'DBPM', 'BPM']:
                        display_stats_df[col] = display_stats_df[col].apply(
                            lambda x: f"{x:.1f}" if pd.notnull(x) else '-')
                    elif col not in ['TS%', '3PT%', 'USG%', 'GP']:
                        display_stats_df[col] = display_stats_df[col].apply(
                            lambda x: f"{x:.2f}" if pd.notnull(x) else '-')

                # Define columns to highlight
                columns_to_highlight = [col for col in display_stats_df.columns if col not in ['TO/36']]

                # Apply styles
                styled = apply_highlighting(display_stats_df, numeric_df, columns_to_highlight, ['TO/36'])

                # Display the styled DataFrame
                st.dataframe(display_stats_df.style.apply(lambda _: styled, axis=None), use_container_width=True)
            else:
                st.warning("Player stats data (CSV) is not available.")

        elif comparison_type == "Radar Chart (Max. 4 players)" and len(selected_players) <= 4:
            st.header("Player Comparison - Radar Chart")
            fig_svg = create_overlaid_radar_chart(comparison_data, (6,6))
            buf_svg = io.BytesIO()
            fig_svg.savefig(buf_svg, format="svg", bbox_inches="tight", transparent=True)
            plt.close(fig_svg)  # Lembre-se de fechar a figura para libertar memória

            # 2. Rebobine o buffer (passo ainda essencial)
            buf_svg.seek(0)

            # 3. Extraia os bytes do buffer e DESCUDIFIQUE-OS para uma string
            svg_string = buf_svg.getvalue().decode('utf-8')

            # 4. Passe a STRING para o st.image
            # Agora o Streamlit irá reconhecer o código SVG e não usará o Pillow
            st.image(svg_string)

        elif comparison_type == "Radar Chart (Max. 4 players)" and len(selected_players) > 4:
            st.warning("Please select up to 4 players for the Radar Chart comparison.")

        elif comparison_type == "Table Comparison":
            comp_table = comparison_data.set_index('Name')[EVAL_CATEGORIES + ['Média Ponderada']]

            comp_num = comp_table.apply(pd.to_numeric, errors='coerce')

            for col in comp_table.columns:
                if col == 'Média Ponderada':
                    comp_table[col] = comp_table[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else '-')
                else:
                    comp_table[col] = comp_table[col].apply(lambda x: f"{int(x)}" if pd.notnull(x) else '-')


            styled_table = comp_table.style.apply(lambda _: apply_highlighting_list(comp_table, comp_num), axis=None)

            st.dataframe(styled_table, use_container_width=True)

# --- Clear Board Option ---
if not st.session_state.big_board.empty:
    st.markdown("---")
    if st.button("🗑️ Clear All Players", type="secondary"):
        st.session_state.big_board = st.session_state.big_board.iloc[0:0]
        save_big_board_to_file(st.session_state.big_board)
        st.success("Big Board has been cleared!")
        st.rerun()