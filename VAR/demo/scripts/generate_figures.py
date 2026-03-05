"""
Generate qualitative example figures for the StreamMind paper.
Each figure simulates a demo screenshot showing the video frame,
the SKM filmstrip, and the chat QA interaction.

Run:  python generate_figures.py
Output: ../../paper/figures/qual_recent.pdf
        ../../paper/figures/qual_instant.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DARK_BG = "#1e1e2e"
PANEL_BG = "#2a2a3c"
ACCENT = "#7c8aff"
USER_BG = "#3a3a5c"
BOT_BG = "#2e4a3e"
SCOPE_COLORS = {"recent": "#f59e0b", "instant": "#22c55e", "historical": "#6366f1"}
TEXT_COLOR = "#e0e0e0"


def draw_video_frame(ax, scene_lines, timestamp="00:01:42"):
    """Draw a simulated video frame with scene description."""
    ax.set_facecolor("#1a1a1a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#555")
        spine.set_linewidth(1.5)

    y = 0.6
    for line in scene_lines:
        ax.text(0.5, y, line, ha="center", va="center",
                fontsize=9, color="#aaa", style="italic")
        y -= 0.12

    ax.text(0.95, 0.05, timestamp, ha="right", va="bottom",
            fontsize=7, color="#ff4444", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
    ax.text(0.05, 0.95, "LIVE", ha="left", va="top",
            fontsize=7, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#cc0000", alpha=0.9))


def draw_filmstrip(ax, n_frames, highlight_idx=None):
    """Draw the SKM filmstrip with thumbnail placeholders."""
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(-0.5, n_frames + 0.5)
    ax.set_ylim(-0.3, 1.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(-0.3, 1.35, "Semantic Keyframe Memory", fontsize=7,
            color=TEXT_COLOR, fontweight="bold", va="top")

    rng = np.random.RandomState(42)
    for i in range(n_frames):
        ec = ACCENT if i == highlight_idx else "#555"
        lw = 2.5 if i == highlight_idx else 0.8
        rect = mpatches.FancyBboxPatch(
            (i - 0.35, 0.1), 0.7, 0.9,
            boxstyle="round,pad=0.04", fc="#333", ec=ec, lw=lw)
        ax.add_patch(rect)
        score = rng.uniform(0.3, 0.95)
        ax.text(i, -0.05, f"{score:.2f}", ha="center", fontsize=5, color="#888")


def draw_chat(ax, messages):
    """Draw the chat panel with question and answer."""
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.5, 0.97, "Interactive QA", ha="center", va="top",
            fontsize=9, color=TEXT_COLOR, fontweight="bold")

    y = 0.88
    for msg in messages:
        role = msg["role"]
        text = msg["text"]
        meta = msg.get("meta")

        if role == "user":
            bg = USER_BG
            align = "right"
            x = 0.95
        else:
            bg = BOT_BG
            align = "left"
            x = 0.05

        wrapped = _wrap(text, 38)
        block_h = len(wrapped) * 0.065 + 0.03
        if meta:
            block_h += 0.055

        rect_x = 0.03 if role != "user" else 0.35
        rect_w = 0.62

        rect = mpatches.FancyBboxPatch(
            (rect_x, y - block_h), rect_w, block_h,
            boxstyle="round,pad=0.02", fc=bg, ec="none", alpha=0.9)
        ax.add_patch(rect)

        text_x = rect_x + 0.03
        text_y = y - 0.04
        for line in wrapped:
            ax.text(text_x, text_y, line, fontsize=6.5, color=TEXT_COLOR, va="top")
            text_y -= 0.065

        if meta:
            scope = meta.get("scope", "")
            sc = SCOPE_COLORS.get(scope, "#888")
            ax.text(text_x, text_y, scope.upper(), fontsize=5.5,
                    color="white", fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc=sc, ec="none", alpha=0.85))
            if "frames" in meta:
                ax.text(text_x + 0.18, text_y, f"{meta['frames']} frames",
                        fontsize=5, color="#999", va="top")
            if "latency" in meta:
                ax.text(text_x + 0.38, text_y, f"{meta['latency']}ms",
                        fontsize=5, color="#999", va="top")

        y -= block_h + 0.04


def _wrap(text, width):
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return lines


def make_figure(scene_lines, timestamp, n_memory, highlight,
                messages, filename):
    fig = plt.figure(figsize=(6.5, 3.2), facecolor=DARK_BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.1, 0.9],
                 height_ratios=[3, 1], hspace=0.15, wspace=0.08,
                 left=0.02, right=0.98, top=0.96, bottom=0.04)

    ax_video = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0])
    ax_chat = fig.add_subplot(gs[:, 1])

    draw_video_frame(ax_video, scene_lines, timestamp)
    draw_filmstrip(ax_strip, n_memory, highlight)
    draw_chat(ax_chat, messages)

    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200, facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved {path}")


# --- Figure (a): Recent-scope cooking query ---
make_figure(
    scene_lines=[
        "[Egocentric cooking stream]",
        "Kitchen counter with cutting board",
        "Person chopping vegetables",
    ],
    timestamp="00:03:24",
    n_memory=8,
    highlight=3,
    messages=[
        {"role": "user",
         "text": "What did the person take out of the fridge?"},
        {"role": "assistant",
         "text": "The person took out a carton of milk and a block of cheese from the refrigerator.",
         "meta": {"scope": "recent", "frames": 4, "latency": 172}},
    ],
    filename="qual_recent.pdf",
)

# --- Figure (b): Instant-scope surveillance query ---
make_figure(
    scene_lines=[
        "[Surveillance camera feed]",
        "Empty office room with desks",
        "Overhead fluorescent lighting",
    ],
    timestamp="00:12:07",
    n_memory=6,
    highlight=5,
    messages=[
        {"role": "user",
         "text": "Is anyone in the room right now?"},
        {"role": "assistant",
         "text": "No, the room is currently empty.",
         "meta": {"scope": "instant", "frames": 1, "latency": 89}},
    ],
    filename="qual_instant.pdf",
)

print("Done. Figures saved to", OUT_DIR)
