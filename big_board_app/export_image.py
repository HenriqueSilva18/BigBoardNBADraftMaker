from streamlit.proto import Heading_pb2
from io import BytesIO
from math import ceil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import RANK_COLUMN
from .storage import order_big_board

# ── Colours ────────────────────────────────────────────────────────────────
BG         = (6,  13,  9)
PANEL      = (14, 22, 17)
PANEL_ALT  = (11, 18, 14)
BORDER     = (38, 58, 46)
GREEN_FULL = (54, 199, 130)
GOLD_FULL  = (242, 198, 109)
TEXT_FULL  = (244, 241, 232)
MUTED_FULL = (160, 175, 163)
DIM_FULL   = (90, 110, 98)
HEADER_BOX = (13, 38, 25)
HEADER_BDR = (50, 120, 82)

# ── Layout (all values at 2× for high-definition output) ──────────────────
CANVAS_W = {1: 2000, 2: 2400, 3: 3200}
PAD      = 80
COL_GAP  = 32
ROW_H    = 130    # generous row height
ROW_GAP  = 12     # visible gap between rows
HEADER_H = 176    # header card
HEAD_PAD = 36     # space below header
FOOT_H   = 84     # footer

# ── Row internal columns ───────────────────────────────────────────────────
RANK_X_REL       = 36
NAME_X_REL       = 140
POS_RIGHT_PAD    = 32
POS_BOX_W        = 130
NAME_INFO_GAP    = 30
NAME_SHARE       = 0.64
MIN_INFO_W       = 240
INFO_LINE_OFFSET = 24

ACCENT_SOFT = (37, 140, 92)

def pick_columns(n: int) -> int:
    """Fewer columns → taller image → more square."""
    if n <= 10:
        return 1
    if n <= 30:
        return 2
    return 3


def load_font(size: int, bold: bool = False):
    candidates = ["seguisb.ttf", "segoeuib.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in candidates:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
            
    # Fallback for stlite / Pyodide deployment
    fallback_font = "Roboto-Bold.ttf" if bold else "Roboto-Regular.ttf"
    if Path(fallback_font).exists():
        return ImageFont.truetype(fallback_font, size=size)

    return ImageFont.load_default()


def clean(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    return "" if s in {"N/A", "nan", "-"} else s


def detail_line(row) -> str:
    """Second line: team, age, measurements (position goes on the name line)."""
    team = clean(row.get("College/Team"))
    age  = clean(row.get("Age"))
    ht   = clean(row.get("Measurements"))
    return "  ·  ".join(p for p in [team, age, ht] if p)


def truncate(draw, text: str, font, max_w: int) -> str:
    if not text:
        return ""
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return (text + "…") if text else ""


def board_to_png_bytes(big_board) -> bytes:
    board = order_big_board(big_board)
    n = len(board)

    num_cols     = pick_columns(n)
    rows_per_col = ceil(n / num_cols) if n > 0 else 1

    # ── Canvas: fixed width, height from content ──────────────────────────
    W      = CANVAS_W[num_cols]
    col_w  = (W - PAD * 2 - COL_GAP * max(num_cols - 1, 0)) // max(num_cols, 1)
    body_h = rows_per_col * (ROW_H + ROW_GAP) - ROW_GAP
    H      = PAD + HEADER_H + HEAD_PAD + body_h + PAD + FOOT_H

    # ── Background ─────────────────────────────────────────────────────────
    image = Image.new("RGB", (W, H), BG)

    # Subtle top-left green glow
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    for r in range(200, 0, -4):
        a = int(7 * (1 - r / 200))
        gd.ellipse((-r, -r, r, r), fill=(*GREEN_FULL, a))
    image.paste(
        Image.alpha_composite(Image.new("RGBA", (W, H)), glow).convert("RGB"),
        mask=glow.split()[3],
    )

    draw = ImageDraw.Draw(image)

    # ── Fonts (2× sizes) ────────────────────────────────────────────────────
    f_kicker = load_font(28, bold=True)
    f_title  = load_font(64, bold=True)
    f_count  = load_font(28)
    f_rank   = load_font(40, bold=True)
    f_name   = load_font(42, bold=True)
    f_pos    = load_font(34)
    f_meta   = load_font(28)
    f_foot   = load_font(24)

    # ── Header card ────────────────────────────────────────────────────────
    hx = PAD
    hy = PAD
    hw = W - PAD * 2

    draw.rounded_rectangle(
        (hx, hy, hx + hw, hy + HEADER_H),
        radius=24,
        fill=HEADER_BOX,
        outline=HEADER_BDR,
        width=2,
    )
    # Left accent bar
    draw.rounded_rectangle(
        (hx + 28, hy + 28, hx + 36, hy + HEADER_H - 28),
        radius=6,
        fill=GREEN_FULL,
    )

    tx = hx + 60
    draw.text((tx, hy + 28), "DRAFT ROOM 2026",     font=f_kicker, fill=GREEN_FULL)
    draw.text((tx, hy + 68), "NBA Draft Big Board", font=f_title,  fill=TEXT_FULL)

    count_text = f"{n} ranked prospect{'s' if n != 1 else ''}"
    cw = int(draw.textlength(count_text, font=f_count))
    draw.text(
        (hx + hw - cw - 40, hy + HEADER_H // 2 - 14),
        count_text,
        font=f_count,
        fill=MUTED_FULL,
    )

    # ── Player rows ────────────────────────────────────────────────────────
    body_top = PAD + HEADER_H + HEAD_PAD

    if board.empty:
        draw.text((PAD + 20, body_top + 32), "No players yet.", font=f_name, fill=MUTED_FULL)
    else:
        for gi, (_, row) in enumerate(board.reset_index(drop=True).iterrows()):
            ci = gi // rows_per_col
            ri = gi % rows_per_col

            rx = PAD + ci * (col_w + COL_GAP)
            ry = body_top + ri * (ROW_H + ROW_GAP)

            rank  = int(row.get(RANK_COLUMN, gi + 1))
            name  = clean(row.get("Name")) or "N/A"
            pos   = clean(row.get("Position"))
            meta  = detail_line(row)
            panel = PANEL if gi % 2 == 0 else PANEL_ALT

            # Row card
            draw.rounded_rectangle(
                (rx, ry, rx + col_w, ry + ROW_H),
                radius=16,
                fill=panel,
                outline=BORDER,
                width=2,
            )

            # Green left accent bar
            draw.rounded_rectangle(
                (rx + 8, ry + 18, rx + 14, ry + ROW_H - 18),
                radius=6,
                fill=(*GREEN_FULL, 90),
            )

            # Fixed internal columns
            mid_y = ry + ROW_H / 2

            rank_x = rx + RANK_X_REL
            name_x = rx + NAME_X_REL

            pos_right = rx + col_w - POS_RIGHT_PAD
            pos_left_bound = pos_right - POS_BOX_W

            usable_w = pos_left_bound - name_x

            name_w = int(usable_w * NAME_SHARE)
            info_x = name_x + name_w + NAME_INFO_GAP
            info_w = pos_left_bound - info_x

            # Guarantee a minimum info column width
            if info_w < MIN_INFO_W:
                info_w = MIN_INFO_W
                info_x = pos_left_bound - info_w
                name_w = info_x - name_x - NAME_INFO_GAP

            # Rank, vertically centered
            rank_str = f"#{rank}"
            draw.text(
                (rank_x, mid_y),
                rank_str,
                font=f_rank,
                fill=GOLD_FULL,
                anchor="lm",
            )

            # Name, vertically centered
            trunc_name = truncate(draw, name, f_name, name_w)
            draw.text(
                (name_x, mid_y),
                trunc_name,
                font=f_name,
                fill=TEXT_FULL,
                anchor="lm",
            )

            # Position, right-aligned inside a fixed position column
            if pos:
                draw.text(
                    (pos_right, mid_y),
                    pos,
                    font=f_pos,
                    fill=MUTED_FULL,
                    anchor="rm",
                )

            # Info block, always starting at the same x position
            team = clean(row.get("College/Team"))
            age  = clean(row.get("Age"))
            ht   = clean(row.get("Measurements"))

            age_ht_parts = []
            if age:
                age_ht_parts.append(str(age))
            if ht:
                age_ht_parts.append(str(ht))

            age_ht = "  ·  ".join(age_ht_parts)

            trunc_team = truncate(draw, team, f_meta, info_w) if team else ""
            trunc_age_ht = truncate(draw, age_ht, f_meta, info_w) if age_ht else ""

            if trunc_team and trunc_age_ht:
                draw.text(
                    (info_x, mid_y - INFO_LINE_OFFSET),
                    trunc_team,
                    font=f_meta,
                    fill=TEXT_FULL,
                    anchor="lm",
                )
                draw.text(
                    (info_x, mid_y + INFO_LINE_OFFSET),
                    trunc_age_ht,
                    font=f_meta,
                    fill=MUTED_FULL,
                    anchor="lm",
                )
            elif trunc_team:
                draw.text(
                    (info_x, mid_y),
                    trunc_team,
                    font=f_meta,
                    fill=TEXT_FULL,
                    anchor="lm",
                )
            elif trunc_age_ht:
                draw.text(
                    (info_x, mid_y),
                    trunc_age_ht,
                    font=f_meta,
                    fill=MUTED_FULL,
                    anchor="lm",
                )

    # ── Footer ─────────────────────────────────────────────────────────────
    fy = H - FOOT_H + 16
    draw.line([(PAD, fy), (W - PAD, fy)], fill=(*BORDER, 140), width=2)
    draw.text((PAD + 8, fy + 24), "Draft Room 2026", font=f_foot, fill=DIM_FULL)

    # ── Export ─────────────────────────────────────────────────────────────
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# Backward-compat alias
def board_to_jpg_bytes(big_board):
    return board_to_png_bytes(big_board)
