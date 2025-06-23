import io
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects
import os

# --- Configuration and Setup ---
st.set_page_config(page_title="NBA Draft Big Board", layout="wide", page_icon="🏀")

# --- Constants for new evaluation categories ---
EVAL_CATEGORIES = [
    "Athleticism", "Dribbling", "Shooting", "Perimeter Defense", "Passing",
    "Scoring", "Interior Defense", "Basketball IQ", "Intangibles"
]

# Define weights for the new categories
WEIGHTS = {
    "Athleticism": 1/8,
    "Scoring": 1/8,
    "Shooting": 1/8,
    "Dribbling": 1/8,
    "Passing": 1/8,
    "Perimeter Defense": 1/16,
    "Interior Defense": 1/16,
    "Basketball IQ": 1/8,
    "Intangibles": 1/8
}


REQUIRED_COLS = (
    ["Name", "Age", "Measurements", "Position", "College/Team",
     "Tier", "Média Ponderada"] + EVAL_CATEGORIES
)

def load_big_board_from_json(fileobj=None, filename="big_board_save.json"):
    """
    Lê exclusivamente ficheiros JSON (orient='records').
    Devolve DataFrame limpo ou None.
    """
    try:
        # -------------------------------------------------- ler bytes
        if fileobj is not None:                 # via upload
            raw = fileobj.read()
        elif os.path.exists(filename):          # em disco
            with open(filename, "rb") as f:
                raw = f.read()
        else:
            return None

        # -------------------------------------------------- json → list[dict]
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):              # salvaguarda
            data = [data]

        df = pd.DataFrame(data)

        # -------------------------------------------------- garantir colunas
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = 5 if col in EVAL_CATEGORIES else "N/A"

        # coercionar tipos
        df[EVAL_CATEGORIES] = df[EVAL_CATEGORIES].apply(
            pd.to_numeric, errors="coerce").fillna(5).astype(int)
        df["Média Ponderada"] = pd.to_numeric(
            df["Média Ponderada"], errors="coerce").fillna(0).round(2)

        # remover duplicados
        df = df.drop_duplicates(subset="Name", keep="first").reset_index(drop=True)

        # remover colunas repetidas caso existam
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    except Exception as e:
        st.error(f"Error loading Big Board: {e}")
        return None


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
    if score >= 9.5: return "Tier 0 - All-Time Talent"
    if score >= 8.5: return "Tier 1 - Superstar"
    if score >= 7.5  : return "Tier 2 - Potential All-NBA"
    if score >= 6.5: return "Tier 3 - Potential All-Star"
    if score >= 5: return "Tier 4 - Starter"
    return "Tier 5 – Fringe NBA / G-League"


# --- Visualization ---
def create_overlaid_radar_chart(players_data, figsize):
    """Cria um gráfico de radar sobreposto e moderno para comparação de jogadores."""
    if players_data.empty:
        return None

    categories = EVAL_CATEGORIES
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
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
    ax.set_xticklabels([])  # Remove os labels padrão

    # Adiciona os labels manualmente com maior distância
    outline_effect = [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()]
    for angle, category in zip(angles[:-1], categories):
        ax.text(angle, 11.7, category, horizontalalignment='center', verticalalignment='center',
                size=10, color='white', path_effects=outline_effect, zorder=10)

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
    if big_board.empty:
        return "Big Board is empty. Nothing to save."  # Retorna uma string

    linhas = ["🏀 NBA Draft Big Board 2025 Rankings\n"]
    for index, row in big_board.iterrows():
        rank = index + 1
        nome = row.get('Name', 'N/A')
        posicao = row.get('Position', 'N/A')
        linhas.append(f"{rank}. {nome} - {posicao}")

    # O passo mais importante: retorna a string completa
    return "\n".join(linhas)

# --- File Operations ---
def save_big_board_to_file(big_board, filename="big_board_save.json"):
    """
    Guarda o DataFrame da Big Board em JSON (orient='records').
    Garante que não há colunas duplicadas.
    """
    try:
        # remove eventuais colunas duplicadas mantendo a 1.ª ocorrência
        big_board = big_board.loc[:, ~big_board.columns.duplicated()]

        big_board.to_json(filename, orient="records", indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


# --- Main App ---
st.title("🏀 NBA Draft Big Board 2025")

df = load_data()

# --- Initialize Session State ---
if "big_board" not in st.session_state:
    columns = [
                  "Name", "Age", "Measurements", "Position", "College/Team", "Tier", "Média Ponderada"
              ] + EVAL_CATEGORIES
    st.session_state.big_board = pd.DataFrame(columns=columns)

for cat in EVAL_CATEGORIES:
    if f"slider_{cat}" not in st.session_state:
        st.session_state[f"slider_{cat}"] = 5

if 'player_select' not in st.session_state:
    st.session_state.player_select = ""


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



if "auto_loaded" not in st.session_state:
    saved_board = load_big_board_from_file()
    if saved_board is not None:
        st.session_state.big_board = saved_board
    st.session_state.auto_loaded = True

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎛️ Controls")
    st.subheader("💾 Download / 📁 Load")

    # A lógica de download só deve aparecer se a big board não estiver vazia
    if 'big_board' in st.session_state and not st.session_state.big_board.empty:

        json_bytes = (
            st.session_state.big_board
            .loc[:, ~st.session_state.big_board.columns.duplicated()]  # garante colunas únicas
            .to_json(orient="records", indent=2)  # DataFrame -> str (JSON)
            .encode("utf-8")  # str -> bytes
        )

        # OPÇÃO 1: DOWNLOAD DO BACKUP COMPLETO (JSON)
        # Este ficheiro serve para poder carregar o estado da board mais tarde.
        st.download_button(
            label="💾 Download Full Backup (JSON)",
            data=json_bytes,
            file_name="big_board_backup_2025.json",
            mime="application/json",
            help="Saves the entire Big Board as a JSON file, that can be reuploaded.",
            use_container_width=True
        )

        big_board_text = save_big_board_to_txt(st.session_state.big_board)
        st.download_button(
            label="📄 Download as Simple Text (.txt)",
            data=big_board_text.encode('utf-8'),
            file_name="big_board_rankings_2025.txt",
            mime="text/plain",
            help="Saves a simple text file with player rankings and positions.",
            use_container_width=True
        )

    else:
        st.info("Add your first player to enable download options.")

    # --- UPLOAD / CARREGAR BIG BOARD -------------------------------------------
    uploaded_file = st.file_uploader(
        "📁 Load Backup (JSON apenas)", type=["json"]
    )

    if uploaded_file is not None:
        loaded_board = load_big_board_from_json(fileobj=uploaded_file)

        if loaded_board is not None and not loaded_board.empty:
            st.session_state.big_board = loaded_board
            st.session_state["board_loaded_now"] = True  # flag!
            save_big_board_to_file(st.session_state.big_board)
        else:
            st.error("❌ Could not read the uploaded JSON file.")

    # Mostrar mensagem de sucesso exactamente 1 vez
    if st.session_state.get("board_loaded_now", False):
        st.success("✅ Big Board loaded successfully!")
        st.session_state["board_loaded_now"] = False  # limpa a flag

    st.markdown("---")
    st.subheader("📊 Evaluation Weights")
    # Assumindo que a constante WEIGHTS existe
    if 'WEIGHTS' in locals() or 'WEIGHTS' in globals():
        for cat, weight in WEIGHTS.items():
            st.text(f"• {cat}: {int(weight * 100)}%")


# --- Callbacks ---
def reset_sliders():
    """Callback to reset all evaluation sliders to their default value."""
    for category in EVAL_CATEGORIES:
        st.session_state[f"slider_{category}"] = 5


# Replace the "Add Player Section" with this improved version:

# --- Add/Modify Player Section ---
st.subheader("➕✏️ Add/Edit Player To Board")

# Initialize player tracking if not exists
if 'last_selected_player' not in st.session_state:
    st.session_state['last_selected_player'] = ""

# Check if we need to reset sliders (this happens BEFORE creating the widgets)
if st.session_state.get('reset_sliders_flag', False):
    for category in EVAL_CATEGORIES:
        st.session_state[f"slider_{category}"] = 5
    st.session_state['reset_sliders_flag'] = False

# Handle clear selection trigger
if st.session_state.get('clear_selection', False):
    st.session_state.player_select = ""
    st.session_state['last_selected_player'] = ""
    for category in EVAL_CATEGORIES:
        st.session_state[f"slider_{category}"] = 5
    st.session_state.clear_selection = False

# Player selection
if df is not None:
    available_players = sorted(df['name'].dropna().unique())
    selected_player = st.selectbox(
        "Select Player to Evaluate",
        [""] + available_players,
        key="player_select"
    )
else:
    selected_player = st.text_input("Player Name", key="player_select")

nome = selected_player

# Check if player is already in big board and show appropriate message
is_existing_player = False
existing_player = None

if nome and not st.session_state.big_board.empty:
    if nome in st.session_state.big_board['Name'].values:
        is_existing_player = True
        # Get existing scores from big board
        existing_player = st.session_state.big_board[st.session_state.big_board['Name'] == nome].iloc[0]

        # Load existing scores into sliders ONLY when player changes (not on every rerun)
        if st.session_state.get('last_selected_player', '') != nome:
            for category in EVAL_CATEGORIES:
                if category in existing_player:
                    st.session_state[f"slider_{category}"] = existing_player[category]
            st.session_state['last_selected_player'] = nome

        # Show modification warning/info
        st.warning(
            f"⚠️ **{nome}** is already in your Big Board. Modify the scores below and click 'Update Player' to save changes.")

        # Show current ranking
        current_board = st.session_state.big_board.sort_values(by="Média Ponderada", ascending=False).reset_index(
            drop=True)
        current_rank = current_board[current_board['Name'] == nome].index[0] + 1
        current_score = existing_player['Média Ponderada']
        st.info(f"📊 Current Ranking: **#{current_rank}** | Current Score: **{current_score:.2f}/10**")
else:
    # If no player selected or player not in board, reset the tracking
    if st.session_state.get('last_selected_player', '') != nome:
        st.session_state['last_selected_player'] = nome

if nome:
    player_info = get_player_info(df, nome)
    st.caption(
        f"Age: {player_info['idade']} | Ht & Ws: {player_info['alt_ws']} | Position: {player_info['posicao']} | Team: {player_info['equipa']}"
    )

# Create the sliders OUTSIDE the form for real-time updates
st.markdown("**Evaluation Categories:**")
scores = {}
cols = st.columns(3)

if 'WEIGHTS' in locals() or 'WEIGHTS' in globals():
    for i, category in enumerate(EVAL_CATEGORIES):
        with cols[i % 3]:
            scores[category] = st.slider(
                f"{category} ({WEIGHTS[category] * 100:.0f}%)",
                0, 10,
                value=st.session_state.get(f"slider_{category}", 5),
                key=f"slider_{category}"
            )

# Real-time preview calculation (now updates automatically with slider changes)
media_preview = calculate_weighted_average(scores)
tier_preview = get_tier(media_preview)

# Display real-time preview with comparison if it's an existing player
col1, col2 = st.columns(2)
with col1:
    st.metric("Score Preview", f"{media_preview}/10", delta=tier_preview)

if is_existing_player and existing_player is not None:
    with col1:
        st.metric("New Score Preview", f"{media_preview}/10", delta=tier_preview)
    with col2:
        current_score = existing_player['Média Ponderada']
        score_change = media_preview - current_score
        if score_change > 0:
            delta_text = f"+{score_change:.2f}"
        elif score_change < 0:
            delta_text = f"{score_change:.2f}"
        else:
            delta_text = "No change"

        st.metric("Current Score", f"{current_score:.2f}/10", delta=delta_text)

# Form with dynamic button text
with st.form("form_jogador"):
    if is_existing_player:
        button_text = f"✏️ Update {nome}"
        button_help = f"Update {nome}'s evaluation scores in the Big Board"
    else:
        button_text = "➕ Add to Big Board"
        button_help = "Add this player to your Big Board"

    # Additional options for existing players
    if is_existing_player:
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button(button_text, type="primary", help=button_help)
        with col2:
            remove_player = st.form_submit_button("🗑️ Remove", type="secondary", help=f"Remove {nome} from Big Board")

    else:
        submitted = st.form_submit_button(button_text, type="primary", use_container_width=True, help=button_help)
        remove_player = False

# Handle form submission
if submitted and nome:
    player_info = get_player_info(df, nome)
    player_data = {
        "Name": nome,
        "Age": player_info['idade'],
        "Measurements": player_info['alt_ws'],
        "Position": player_info['posicao'],
        "College/Team": player_info['equipa'],
        "Média Ponderada": media_preview,
        "Tier": tier_preview,
        **scores
    }

    if is_existing_player:
        # Update existing player
        st.session_state.big_board.loc[
            st.session_state.big_board['Name'] == nome, player_data.keys()
        ] = list(player_data.values())
        save_big_board_to_file(st.session_state.big_board)
        st.success(f"✅ {nome}'s evaluation has been updated!")
    else:
        # Add new player
        new_player_df = pd.DataFrame([player_data])
        st.session_state.big_board = pd.concat([st.session_state.big_board, new_player_df], ignore_index=True)
        save_big_board_to_file(st.session_state.big_board)
        st.success(f"✅ {nome} has been added to the Big Board!")

    # Clear player selection and reset sliders using the flag
    st.session_state['last_selected_player'] = ""
    st.session_state['reset_sliders_flag'] = True
    st.session_state.clear_selection = True
    st.rerun()

# Handle player removal
if remove_player and nome:
    st.session_state.big_board = st.session_state.big_board[st.session_state.big_board['Name'] != nome]
    save_big_board_to_file(st.session_state.big_board)
    st.success(f"✅ {nome} has been removed from the Big Board!")

    # Set flag to reset sliders on next run
    st.session_state['reset_sliders_flag'] = True
    st.rerun()

# --- Big Board Rankings Display ---
st.markdown("---")
st.subheader("📋 Big Board Rankings")

if not st.session_state.big_board.empty:
    display_board = st.session_state.big_board.sort_values(by="Média Ponderada", ascending=False).reset_index(drop=True)

    # Defina o índice para começar em 1
    display_board.index = display_board.index + 1
    display_board.index.name = "Rank"

    display_cols = ["Name", "Tier", "Média Ponderada", "Age", "Measurements", "Position", "College/Team"] + EVAL_CATEGORIES
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
    big_board = save_big_board_to_txt(st.session_state.big_board)
    if big_board:
        st.download_button(
            label="💾 Download Big Board as Text File",
            data=big_board.encode('utf-8'),  # Importante: converter string para bytes
            file_name="big_board.txt",
            mime="text/plain"
        )
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