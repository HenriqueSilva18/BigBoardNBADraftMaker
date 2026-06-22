import streamlit as st


APP_CSS = """
<style>
:root {
    color-scheme: dark;
    --court-bg: #070a08;
    --court-panel: #101610;
    --court-panel-2: #151d17;
    --court-line: #2c3a31;
    --court-line-strong: #496253;
    --court-text: #f4f1e8;
    --court-muted: #b6c2b8;
    --court-dim: #839187;
    --court-green: #36c782;
    --court-gold: #f2c66d;
}

header,
[data-testid="stHeader"],
[data-testid="stDecoration"] {
    display: none !important;
    height: 0px !important;
}

#MainMenu {
    display: none !important;
}

.stApp {
    background:
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,0.028) 1px, transparent 1px),
        radial-gradient(circle at 18% 0%, rgba(54, 199, 130, 0.16), transparent 26rem),
        radial-gradient(circle at 88% 12%, rgba(242, 198, 109, 0.10), transparent 22rem),
        var(--court-bg);
    background-size: 72px 72px, 72px 72px, auto, auto, auto;
    color: var(--court-text);
    cursor: default;
}

.stApp *:not(input):not(textarea):not(select):not([contenteditable="true"]):not(button):not(a):not([role="slider"]):not([role="textbox"]):not([role="combobox"]):not([role="listbox"]):not([role="option"]):not([role="spinbutton"]) {
    cursor: default;
}

.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] * {
    user-select: text;
    -webkit-user-select: text;
    cursor: default !important;
    caret-color: transparent !important;
}

.block-container {
    max-width: 1360px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6,
.stMarkdown,
[data-testid="stMarkdownContainer"] {
    color: var(--court-text);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(54, 199, 130, 0.08), transparent 18rem),
        #0b100d;
    border-right: 1px solid var(--court-line);
}

[data-testid="stSidebar"] * {
    color: var(--court-text);
}

[data-testid="stSidebar"] .stAlert * {
    color: var(--court-text);
}

h1, h2, h3 {
    letter-spacing: 0;
}

div[data-testid="stMetric"] {
    background: var(--court-panel-2);
    border: 1px solid var(--court-line);
    border-radius: 8px;
    padding: 0.85rem 1rem;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--court-muted) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--court-text) !important;
}

.draft-hero {
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    padding: 1.35rem 1.45rem;
    margin-bottom: 1.25rem;
    color: var(--court-text);
    background:
        linear-gradient(135deg, rgba(12, 17, 13, 0.98), rgba(28, 90, 62, 0.94)),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.12) 0 1px, transparent 1px 86px);
    border: 1px solid rgba(148, 255, 193, 0.20);
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.34);
}

.stApp .draft-hero p {
    margin: 0;
    color: var(--court-muted) !important;
}

.draft-hero h1 {
    margin: 0.2rem 0 0.35rem;
    font-size: clamp(2rem, 5vw, 3.2rem);
    line-height: 1;
}

.stApp .hero-kicker {
    text-transform: uppercase;
    font-size: 0.74rem;
    letter-spacing: 0.12em;
    color: #a8ffc8 !important;
    font-weight: 700;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 1rem 0 1.2rem;
}

.kpi-card,
.board-card,
.empty-state {
    background: linear-gradient(180deg, rgba(21, 29, 23, 0.96), rgba(13, 18, 15, 0.96));
    border: 1px solid var(--court-line);
    border-radius: 8px;
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
}

.kpi-card {
    padding: 0.9rem 1rem;
}

.stApp .kpi-label {
    color: var(--court-dim) !important;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.stApp .kpi-value {
    color: var(--court-text) !important;
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.18;
    margin-top: 0.35rem;
}

.board-card {
    padding: 1rem;
    margin-top: 0.1rem;
    max-height: 560px;
    overflow-y: auto;
    overscroll-behavior: contain;
    scrollbar-color: var(--court-green) #0b100d;
    scrollbar-width: thin;
}

.board-card::-webkit-scrollbar {
    width: 0.55rem;
}

.board-card::-webkit-scrollbar-track {
    background: #0b100d;
    border-radius: 999px;
}

.board-card::-webkit-scrollbar-thumb {
    background: var(--court-green-strong);
    border-radius: 999px;
}

.board-row {
    display: grid;
    grid-template-columns: 2.3rem minmax(0, 1fr) 4.1rem;
    gap: 0.75rem;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.7rem 0;
}

.board-row:last-child {
    border-bottom: 0;
}

.stApp .board-rank {
    color: var(--court-gold) !important;
    font-size: 0.95rem;
    font-weight: 900;
}

.stApp .board-name {
    color: var(--court-text) !important;
    font-weight: 800;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.stApp .board-meta {
    color: var(--court-dim) !important;
    font-size: 0.78rem;
    margin-top: 0.15rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.stApp .board-score {
    color: #0b100d !important;
    background: var(--court-green);
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 900;
    padding: 0.24rem 0.45rem;
    text-align: center;
}

.compare-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem;
    margin: 0.85rem 0 1.1rem;
}

.compare-card {
    background:
        linear-gradient(180deg, rgba(21, 29, 23, 0.96), rgba(13, 18, 15, 0.96));
    border: 1px solid var(--court-line);
    border-radius: 8px;
    min-height: 150px;
    padding: 0.9rem 0.95rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 14px 30px rgba(0, 0, 0, 0.24);
}

.compare-card::after {
    content: "";
    position: absolute;
    inset: auto -3rem -4rem auto;
    width: 8rem;
    height: 8rem;
    border-radius: 999px;
    background: rgba(54, 199, 130, 0.10);
}

.stApp .compare-rank {
    color: var(--court-gold) !important;
    font-size: 0.9rem;
    font-weight: 900;
}

.stApp .compare-name {
    color: var(--court-text) !important;
    font-size: 1.18rem;
    font-weight: 900;
    line-height: 1.15;
    margin-top: 0.35rem;
}

.stApp .compare-meta {
    color: var(--court-muted) !important;
    font-size: 0.84rem;
    margin-top: 0.3rem;
}

.stApp .compare-score {
    color: #06100a !important;
    background: var(--court-green);
    border-radius: 999px;
    display: inline-block;
    font-size: 0.96rem;
    font-weight: 950;
    margin-top: 0.65rem;
    padding: 0.22rem 0.55rem;
}

.compare-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.7rem;
    position: relative;
    z-index: 1;
}

.stApp .compare-chip-row span {
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    color: var(--court-muted) !important;
    font-size: 0.76rem;
    padding: 0.18rem 0.48rem;
}

.stApp .compare-subtitle {
    color: var(--court-green) !important;
    font-size: 0.82rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    margin: 0.9rem 0 0.5rem;
    text-transform: uppercase;
}

.comparison-table-wrap {
    border: 1px solid var(--court-line);
    border-radius: 8px;
    max-width: 100%;
    overflow-x: hidden;
    overflow-y: hidden;
    background: #0d120f;
    box-shadow: 0 14px 30px rgba(0, 0, 0, 0.22);
}

.comparison-table-wrap table {
    border-collapse: collapse;
    table-layout: fixed;
    width: 100% !important;
}

.comparison-table-wrap th,
.comparison-table-wrap td {
    border: 1px solid var(--court-line) !important;
    font-size: clamp(0.72rem, 0.82vw, 0.92rem);
    line-height: 1.25;
    overflow: hidden;
    padding: 0.55rem 0.45rem !important;
    text-align: center;
    text-overflow: ellipsis;
    white-space: normal;
    word-break: normal;
}

.comparison-table-wrap th {
    background: #151d17 !important;
    color: var(--court-muted) !important;
    font-weight: 900 !important;
}

.comparison-table-wrap td {
    color: var(--court-text);
}

.comparison-table-wrap th:first-child,
.comparison-table-wrap td:first-child {
    text-align: left;
    width: 12%;
}

.player-chip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.35rem 0 0.9rem;
}

.stApp .player-chip span {
    background: rgba(54, 199, 130, 0.12);
    border: 1px solid rgba(54, 199, 130, 0.28);
    border-radius: 999px;
    color: var(--court-text) !important;
    font-size: 0.85rem;
    padding: 0.25rem 0.65rem;
}

.stApp .empty-state {
    padding: 1.3rem;
    color: var(--court-muted) !important;
}

.stApp .section-label {
    color: var(--court-green) !important;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: -0.2rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    border-bottom: 1px solid var(--court-line);
}

.stTabs [data-baseweb="tab"] {
    background: var(--court-panel);
    border: 1px solid var(--court-line);
    border-bottom: 0;
    border-radius: 8px 8px 0 0;
    color: var(--court-muted);
    padding: 0.45rem 0.85rem;
}

.stTabs [aria-selected="true"] {
    background: var(--court-green-soft);
    color: var(--court-text) !important;
}

/* =========================================================
   BUTTONS — outlined ghost style, fills on hover
   ========================================================= */

div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
div[data-testid="stFormSubmitButton"] button,
.stButton button,
.stDownloadButton button,
.stFormSubmitButton button {
    border-radius: 6px !important;
    border: 1.5px solid var(--court-green) !important;
    background: transparent !important;
    font-family: Inter, Segoe UI, Arial, sans-serif !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    font-size: 0.78rem !important;
    padding: 0.55rem 1.2rem !important;
    transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease !important;
}

div[data-testid="stButton"] button *,
div[data-testid="stDownloadButton"] button *,
div[data-testid="stFormSubmitButton"] button *,
.stButton button *,
.stDownloadButton button *,
.stFormSubmitButton button * {
    color: var(--court-green) !important;
    font-weight: 700 !important;
}

div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover,
.stButton button:hover,
.stDownloadButton button:hover,
.stFormSubmitButton button:hover {
    background: var(--court-green) !important;
    border-color: var(--court-green) !important;
    box-shadow: 0 0 14px rgba(54, 199, 130, 0.35) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stButton"] button:hover *,
div[data-testid="stDownloadButton"] button:hover *,
div[data-testid="stFormSubmitButton"] button:hover *,
.stButton button:hover *,
.stDownloadButton button:hover *,
.stFormSubmitButton button:hover * {
    color: #0b100d !important;
}

div[data-testid="stButton"] button:active,
div[data-testid="stDownloadButton"] button:active,
div[data-testid="stFormSubmitButton"] button:active,
.stButton button:active,
.stDownloadButton button:active,
.stFormSubmitButton button:active {
    transform: translateY(0) !important;
    box-shadow: 0 0 6px rgba(54, 199, 130, 0.2) !important;
}
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    border-radius: 8px;
    background: #0f1511 !important;
    border-color: var(--court-line-strong) !important;
    color: var(--court-text) !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="input"] *,
div[data-baseweb="textarea"] *,
input,
textarea {
    color: var(--court-text) !important;
}

ul[role="listbox"],
div[role="listbox"] {
    background: #0f1511 !important;
    border: 1px solid var(--court-line-strong) !important;
}

li[role="option"],
div[role="option"] {
    color: var(--court-text) !important;
}

label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] * {
    color: var(--court-muted) !important;
    opacity: 1 !important;
}

[data-testid="stSlider"] * {
    color: var(--court-muted) !important;
}

[data-testid="stSlider"] [role="slider"] {
    background: var(--court-green) !important;
    border-color: var(--court-green) !important;
}

[data-testid="stSlider"] div[data-testid="stTickBar"] {
    color: var(--court-dim) !important;
}

[data-testid="stSlider"] input,
[data-testid="stSlider"] [contenteditable="true"],
[data-testid="stSlider"] [role="spinbutton"] {
    caret-color: transparent !important;
    cursor: default !important;
}

[data-testid="stSlider"] input:focus,
[data-testid="stSlider"] [contenteditable="true"]:focus,
[data-testid="stSlider"] [role="spinbutton"]:focus {
    caret-color: transparent !important;
    outline: none !important;
}

[data-testid="stFileUploader"] section {
    background: #0f1511 !important;
    border: 1px dashed var(--court-line-strong) !important;
    border-radius: 8px;
}

[data-testid="stFileUploader"] section * {
    color: var(--court-text) !important;
}

[data-testid="stAlert"] {
    background: #111812 !important;
    border: 1px solid var(--court-line-strong) !important;
    color: var(--court-text) !important;
}

[data-testid="stAlert"] * {
    color: var(--court-text) !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--court-line);
    border-radius: 8px;
    overflow: hidden;
}

[data-testid="stDataFrame"] * {
    color: inherit;
}

hr {
    border-color: var(--court-line);
}

@media (max-width: 900px) {
    .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 560px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def inject_theme():
    st.markdown(APP_CSS, unsafe_allow_html=True)
