"""Draw the README figures as SVG, each in a light and a dark version.

    python experiments/make_readme_figures.py      # writes docs/img/*.svg

The numbers are copied from the result files named next to each figure (data/results/ is
not in the repository); change them here if a result is re-run.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
WIDTH = 720
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "border": "#e1e0d9",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "neutral": "#b5b3ab",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "good": "#0ca30c",
        "critical": "#d03b3b",
        "wash": "#f3f2ee",
    },
    "dark": {
        "surface": "#1a1a19",
        "border": "#2c2c2a",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "neutral": "#6b6a64",
        "blue": "#3987e5",
        "orange": "#d95926",
        "good": "#0ca30c",
        "critical": "#d03b3b",
        "wash": "#232322",
    },
}


def text(x, y, s, size=13, fill="ink", weight=400, anchor="start", t=None):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{t[fill]}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def frame(height, body, t, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(title)}" '
        f"font-family='{FONT}'>"
        f"<title>{escape(title)}</title>"
        f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="12" '
        f'fill="{t["surface"]}" stroke="{t["border"]}"/>' + "".join(body) + "</svg>"
    )


def hbar(x, y, length, height, fill):
    """Horizontal bar: square at the baseline (left), 4px rounded data end."""
    r = min(4, length / 2)
    return (
        f'<path d="M{x},{y} H{x + length - r} Q{x + length},{y} {x + length},{y + r} '
        f"V{y + height - r} Q{x + length},{y + height} {x + length - r},{y + height} "
        f'H{x} Z" fill="{fill}"/>'
    )


def vbar(x, base, height, width, fill):
    """Column: square at the baseline (bottom), 4px rounded data end."""
    r = min(4, height / 2)
    top = base - height
    return (
        f'<path d="M{x},{base} V{top + r} Q{x},{top} {x + r},{top} H{x + width - r} '
        f'Q{x + width},{top} {x + width},{top + r} V{base} Z" fill="{fill}"/>'
    )


def leak_chart(t):
    """Experiment 1: data/results/experiment1_scores.json (+ _oldsplit for the live run)."""
    title = "Hidden reasoning that reached the attacker"
    b = [
        text(28, 40, title, 18, weight=600, t=t),
        text(
            28,
            63,
            "267 extraction attacks on DeepSeek-R1-0528-Qwen3-8B · share of complete replies",
            13,
            "ink2",
            t=t,
        ),
    ]
    x0, full = 220, 400
    rows = [
        ("Model served directly", 1.0, "100%", "neutral"),
        ("Through Unmimic", 0.016, "1.6%", "blue"),
    ]
    for i, (label, share, value, color) in enumerate(rows):
        y = 100 + i * 52
        b.append(text(x0 - 16, y + 17, label, 14, "ink", 500, "end", t=t))
        length = max(full * share, 4)
        b.append(hbar(x0, y, length, 24, t[color]))
        b.append(text(x0 + length + 10, y + 17, value, 14, "ink", 600, t=t))
    b.append(f'<line x1="{x0}" y1="92" x2="{x0}" y2="184" stroke="{t["axis"]}"/>')
    b.append(
        text(
            x0 + 52,
            169,
            "all from one bug (a second </think>), fixed in 0.1.0",
            12,
            "muted",
            t=t,
        )
    )
    b.append(
        text(
            28,
            214,
            "Without a guard, a server that passes <think> through sends the "
            "full reasoning with every reply.",
            12,
            "muted",
            t=t,
        )
    )
    return frame(236, b, t, title)


def guard_tiles(t):
    """Input guard: data/results/input_guard_heldout.json (flag 0.5, memory 0.70)."""
    title = "Input guard on attack types it has never seen"
    b = [text(28, 40, title, 18, weight=600, t=t)]
    tiles = [
        (
            "92%",
            "of extraction attempts caught",
            "blocked or flagged · 95% CI 88–95%",
            "267 attacks of 11 types, each type held out of training",
        ),
        (
            "0.04%",
            "of normal prompts blocked",
            "at most 0.09% (95% upper bound)",
            "10,000 real user prompts from WildChat-1M",
        ),
    ]
    for i, (value, label, ci, source) in enumerate(tiles):
        x = 28 + i * 340
        b.append(f'<rect x="{x}" y="62" width="324" height="138" rx="10" fill="{t["wash"]}"/>')
        b.append(text(x + 20, 120, value, 46, "ink", 600, t=t))
        b.append(text(x + 20, 148, label, 15, "ink", 500, t=t))
        b.append(text(x + 20, 170, ci, 12, "ink2", t=t))
        b.append(text(x + 20, 188, source, 12, "muted", t=t))
    return frame(222, b, t, title)


def copycat_chart(t):
    """Students: data/results/student_gains*.json and students_*.json (temperature 0.6)."""
    title = "Does hiding the reasoning stop a copycat?"
    b = [
        text(28, 40, title, 18, weight=600, t=t),
        text(
            28,
            63,
            "GSM8K accuracy (1,319 test problems) after training on ~2,600 replies "
            "from the protected model",
            13,
            "ink2",
            t=t,
        ),
    ]
    series = [
        ("Untrained", "neutral"),
        ("Trained on full reasoning", "blue"),
        ("Trained through Unmimic", "orange"),
    ]
    lx = 28
    for name, color in series:
        b.append(f'<rect x="{lx}" y="84" width="12" height="12" rx="3" fill="{t[color]}"/>')
        b.append(text(lx + 18, 95, name, 13, "ink2", t=t))
        lx += 34 + len(name) * 6.4  # ~6.4 px per character at 13 px
    top, base, left, right = 128, 330, 70, 690
    scale = (base - top) / 100
    for tick in (0, 25, 50, 75, 100):
        y = base - tick * scale
        stroke = t["axis"] if tick == 0 else t["grid"]
        b.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="{stroke}"/>')
        b.append(text(left - 10, y + 4, f"{tick}%", 11, "muted", anchor="end", t=t))
    teacher = base - 86.1 * scale
    b.append(
        f'<line x1="{left}" y1="{teacher}" x2="{right}" y2="{teacher}" '
        f'stroke="{t["ink2"]}" stroke-width="1"/>'
    )
    b.append(text(right, teacher - 6, "Protected model 86%", 11, "ink2", anchor="end", t=t))
    groups = [
        ("Qwen2.5-0.5B copycat", (28.6, 29.6, 40.2)),
        ("Qwen2.5-1.5B copycat", (58.7, 63.2, 64.3)),
    ]
    bar, gap = 28, 10
    for g, (name, values) in enumerate(groups):
        center = left + (right - left) * (0.27 + 0.46 * g)
        x = center - (3 * bar + 2 * gap) / 2
        for (_, color), value in zip(series, values, strict=True):
            b.append(vbar(x, base, value * scale, bar, t[color]))
            b.append(
                text(
                    x + bar / 2,
                    base - value * scale - 7,
                    f"{value:.0f}%",
                    12,
                    "ink",
                    600,
                    "middle",
                    t=t,
                )
            )
            x += bar + gap
        b.append(text(center, base + 22, name, 13, "ink2", 500, "middle", t=t))
    b.append(
        text(
            28,
            388,
            "Small copycats learn from the short visible answer, not the long "
            "hidden reasoning, so hiding it",
            12,
            "muted",
            t=t,
        )
    )
    b.append(
        text(
            28,
            405,
            "doesn't slow them down. Larger copycats (7B+) are not tested yet.",
            12,
            "muted",
            t=t,
        )
    )
    return frame(426, b, t, title)


def flow_chart(t):
    title = "How a request flows through Unmimic"
    b = [text(28, 40, title, 18, weight=600, t=t)]
    b.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{t["muted"]}"/></marker></defs>'
    )
    cx, w = 250, 330

    def box(y, h, head, sub, accent=None):
        stroke = t[accent] if accent else t["border"]
        width = 2 if accent else 1
        parts = [
            f'<rect x="{cx - w / 2}" y="{y}" width="{w}" height="{h}" rx="10" '
            f'fill="{t["wash"]}" stroke="{stroke}" stroke-width="{width}"/>',
            text(cx, y + (h / 2 if not sub else 26), head, 14, "ink", 600, "middle", t=t),
        ]
        if sub:
            parts.append(text(cx, y + 45, sub, 12, "ink2", anchor="middle", t=t))
        return parts

    def side(y, head, sub, accent):
        x = 500
        return [
            f'<line x1="{cx + w / 2}" y1="{y + 30}" x2="{x - 2}" y2="{y + 30}" '
            f'stroke="{t["muted"]}" stroke-width="1.5" marker-end="url(#arrow)"/>',
            text((cx + w / 2 + x) / 2, y + 22, "yes", 12, "muted", anchor="middle", t=t),
            f'<rect x="{x}" y="{y + 6}" width="192" height="48" rx="10" fill="{t["surface"]}" '
            f'stroke="{t[accent]}" stroke-width="2"/>',
            text(x + 96, y + 27, head, 13, "ink", 600, "middle", t=t),
            text(x + 96, y + 44, sub, 11, "ink2", anchor="middle", t=t),
        ]

    def down(y1, y2, label=""):
        parts = [
            f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2 - 2}" stroke="{t["muted"]}" '
            f'stroke-width="1.5" marker-end="url(#arrow)"/>'
        ]
        if label:
            parts.append(text(cx + 10, (y1 + y2) / 2 + 4, label, 12, "muted", t=t))
        return parts

    b += box(64, 44, "Request from a user", "")
    b += down(108, 138)
    b += box(138, 60, "Input guard", "Is this an attempt to extract the reasoning?")
    b += side(138, "✕  Blocked (403)", "never reaches the model", "critical")
    b += down(198, 228, "no")
    b += box(228, 60, "Your model", "vLLM, Ollama, any OpenAI-compatible server")
    b += down(288, 318, "reply = hidden reasoning + answer")
    b += box(318, 60, "Split reasoning from answer", "format learned once by  unmimic init")
    b += down(378, 408)
    b += box(
        408, 60, "Leak detector", "Is the reasoning in the answer? Copied, encoded, translated"
    )
    b += side(408, "✕  Withheld", "answer replaced with a notice", "critical")
    b += down(468, 498, "no")
    b += box(498, 60, "Strip the reasoning", "remove <think> blocks and reasoning fields")
    b += down(558, 588)
    b += box(588, 44, "✓  User gets the answer only", "", accent="good")
    return frame(656, b, t, title)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "leak": leak_chart,
        "input-guard": guard_tiles,
        "copycat": copycat_chart,
        "flow": flow_chart,
    }
    for name, draw in figures.items():
        for mode, theme in THEMES.items():
            path = OUT / f"{name}-{mode}.svg"
            path.write_text(draw(theme) + "\n", encoding="utf-8")
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
