from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import RANK_COLUMN, SCORE_COLUMN
from .storage import order_big_board


WIDTH = 1400
MARGIN = 72
HEADER_HEIGHT = 210
ROW_HEIGHT = 86
FOOTER_HEIGHT = 56
BG = "#07100b"
PANEL = "#101a13"
PANEL_ALT = "#132017"
LINE = "#2d4436"
TEXT = "#f4f1e8"
MUTED = "#aeb9af"
GREEN = "#36c782"
GOLD = "#f2c66d"


def load_font(size, bold=False):
    font_names = ["seguisb.ttf", "segoeuib.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for font_name in font_names:
        font_path = Path("C:/Windows/Fonts") / font_name
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def fit_text(draw, text, font, max_width):
    text = str(text)
    if draw.textlength(text, font=font) <= max_width:
        return text

    while text and draw.textlength(f"{text}...", font=font) > max_width:
        text = text[:-1]
    return f"{text}..." if text else ""


def draw_pill(draw, xy, text, font, fill, text_fill):
    left, top, right, bottom = xy
    draw.rounded_rectangle(xy, radius=(bottom - top) // 2, fill=fill)
    text_width = draw.textlength(text, font=font)
    text_height = font.getbbox(text)[3] - font.getbbox(text)[1]
    draw.text(
        ((left + right - text_width) / 2, (top + bottom - text_height) / 2 - 2),
        text,
        fill=text_fill,
        font=font,
    )


def board_to_jpg_bytes(big_board):
    board = order_big_board(big_board)
    row_count = max(len(board), 1)
    height = HEADER_HEIGHT + row_count * ROW_HEIGHT + FOOTER_HEIGHT + MARGIN

    image = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(image)

    title_font = load_font(58, bold=True)
    subtitle_font = load_font(24)
    label_font = load_font(20, bold=True)
    rank_font = load_font(28, bold=True)
    name_font = load_font(30, bold=True)
    meta_font = load_font(19)
    score_font = load_font(22, bold=True)

    draw.rectangle((0, 0, WIDTH, height), fill=BG)
    draw.rounded_rectangle(
        (MARGIN, 44, WIDTH - MARGIN, HEADER_HEIGHT - 18),
        radius=18,
        fill="#0f2a1c",
        outline="#2d6c4b",
        width=2,
    )
    draw.text((MARGIN + 34, 76), "DRAFT ROOM 2025", fill=GREEN, font=label_font)
    draw.text((MARGIN + 34, 104), "NBA Draft Big Board", fill=TEXT, font=title_font)
    draw.text(
        (MARGIN + 38, 172),
        f"{len(board)} prospects ranked by your board order",
        fill=MUTED,
        font=subtitle_font,
    )

    y = HEADER_HEIGHT
    if board.empty:
        draw.text((MARGIN, y), "No players on the board yet.", fill=MUTED, font=subtitle_font)
    else:
        for index, row in board.reset_index(drop=True).iterrows():
            rank = int(row.get(RANK_COLUMN, index + 1))
            name = fit_text(draw, row.get("Name", "N/A"), name_font, 580)
            position = row.get("Position", "N/A")
            team = row.get("College/Team", "N/A")
            tier = row.get("Tier", "N/A")
            score = float(row.get(SCORE_COLUMN, 0))
            meta = fit_text(draw, f"{position} - {team} - {tier}", meta_font, 760)
            fill = PANEL if index % 2 == 0 else PANEL_ALT

            draw.rounded_rectangle(
                (MARGIN, y, WIDTH - MARGIN, y + ROW_HEIGHT - 12),
                radius=14,
                fill=fill,
                outline=LINE,
                width=1,
            )
            draw.text((MARGIN + 26, y + 24), f"#{rank}", fill=GOLD, font=rank_font)
            draw.text((MARGIN + 112, y + 16), name, fill=TEXT, font=name_font)
            draw.text((MARGIN + 114, y + 52), meta, fill=MUTED, font=meta_font)

            draw_pill(
                draw,
                (WIDTH - MARGIN - 124, y + 23, WIDTH - MARGIN - 28, y + 58),
                f"{score:.2f}",
                score_font,
                GREEN,
                "#06100a",
            )
            y += ROW_HEIGHT

    draw.text(
        (MARGIN, height - 46),
        "Generated from Draft Room 2025",
        fill="#6f8174",
        font=meta_font,
    )

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=94, optimize=True)
    return buffer.getvalue()
