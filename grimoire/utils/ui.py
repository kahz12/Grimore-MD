"""
User Interface (UI) and Theming for Project Grimoire.
This module centralizes the Rich theme and reusable UI components such as
banners, panels, tables, and progress bars to ensure a consistent look and feel
across all CLI commands.
"""
from contextlib import contextmanager
from typing import Iterator

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich import box

# Custom Rich theme with Grimoire's signature "mystic" color palette
THEME = Theme({
    "grimoire.primary":   "bold medium_purple3",
    "grimoire.secondary": "medium_orchid",
    "grimoire.accent":    "cyan",
    "grimoire.success":   "bold green",
    "grimoire.warning":   "bold yellow",
    "grimoire.danger":    "bold red",
    "grimoire.muted":     "grey50",
    "grimoire.mystic":    "italic medium_orchid",
    "grimoire.rune":      "bold cyan",
})

console = Console(theme=THEME, highlight=False)


# ── Brand / Headers ─────────────────────────────────────────────────────────

def render_banner(version: str = "v2.0") -> Panel:
    """Returns the large centered brand panel for the dashboard."""
    title = Text.assemble(
        ("🔮  ", "grimoire.accent"),
        ("GRIMOIRE", "grimoire.primary"),
        ("  ·  ", "grimoire.muted"),
        (version, "grimoire.accent"),
    )
    subtitle = Text("Automated Knowledge Engine", style="grimoire.mystic")
    return Panel(
        Align.center(Group(title, subtitle)),
        border_style="grimoire.primary",
        padding=(1, 2),
    )


def command_header(title: str, subtitle: str = "") -> None:
    """Prints a consistent branded header at the start of a command execution."""
    bar = Text()
    bar.append("✦ ", style="grimoire.accent")
    bar.append(title, style="grimoire.primary")
    if subtitle:
        bar.append("  · ", style="grimoire.muted")
        bar.append(subtitle, style="grimoire.mystic")
    console.print()
    console.print(bar)
    console.print(Text("─" * min(console.width - 2, 60), style="grimoire.muted"))


def section(title: str) -> None:
    """Prints a section divider with a rune bullet point."""
    console.print()
    console.print(Text(f"  ⟡ {title}", style="grimoire.rune"))


def tip(message: str) -> None:
    """Prints a light bulb hint message to the console."""
    console.print(
        Text.assemble(
            Text("  💡 ", style="grimoire.muted"),
            Text.from_markup(message),
        )
    )


# ── Panels ──────────────────────────────────────────────────────────────────

def info_panel(message, title: str = "Info") -> Panel:
    return Panel(message, title=title, border_style="grimoire.accent", padding=(0, 1))


def success_panel(message, title: str = "Done") -> Panel:
    return Panel(message, title=title, border_style="grimoire.success", padding=(0, 1))


def warn_panel(message, title: str = "Warning") -> Panel:
    return Panel(message, title=title, border_style="grimoire.warning", padding=(0, 1))


def error_panel(message, title: str = "Error") -> Panel:
    return Panel(message, title=title, border_style="grimoire.danger", padding=(0, 1))


def oracle_panel(answer: str) -> Panel:
    """Custom panel for displaying the Oracle's answers with specific styling."""
    return Panel(
        Text(answer, style="white"),
        title=Text("🔮  Oracle", style="grimoire.primary"),
        border_style="grimoire.secondary",
        padding=(1, 2),
    )


# ── Badges ──────────────────────────────────────────────────────────────────

def dry_run_badge() -> Text:
    return Text(" DRY-RUN ", style="black on bright_yellow")


def live_mode_badge() -> Text:
    return Text(" LIVE ", style="black on bright_green")


def daemon_badge(active: bool) -> Text:
    if active:
        return Text(" ● ACTIVE ", style="white on green")
    return Text(" ○ INACTIVE ", style="white on grey35")


# ── Tables ──────────────────────────────────────────────────────────────────

def kv_table(rows: list[tuple[str, object]]) -> Table:
    """Creates a clean two-column table for key-value dashboard metrics."""
    t = Table(
        box=box.SIMPLE,
        show_header=False,
        padding=(0, 2),
        pad_edge=False,
    )
    t.add_column(style="grimoire.muted", no_wrap=True)
    t.add_column()
    for key, value in rows:
        if isinstance(value, (str, int, float)):
            value = Text(str(value), style="grimoire.accent")
        t.add_row(key, value)
    return t


def tag_frequency_table(rows: list[tuple[str, int]], width: int = 20) -> Table:
    """Creates a table showing tag frequency with relative visual bars."""
    t = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="grimoire.muted",
        padding=(0, 2),
        pad_edge=False,
    )
    t.add_column("Tag", style="grimoire.primary", no_wrap=True)
    t.add_column("Uses", justify="right", style="grimoire.accent", no_wrap=True)
    t.add_column("Frequency")

    if not rows:
        return t

    peak = max((count for _, count in rows), default=1) or 1
    for name, count in rows:
        ratio = count / peak if peak else 0.0
        filled = max(1, int(round(ratio * width))) if count else 0
        bar = Text()
        bar.append("█" * filled, style="grimoire.secondary")
        bar.append("░" * (width - filled), style="grimoire.muted")
        t.add_row(name, str(count), bar)
    return t


def coverage_bar(done: int, total: int, width: int = 20) -> Text:
    """Visual bar showing progress or coverage ratio."""
    if total <= 0:
        return Text("—", style="grimoire.muted")
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(ratio * width))
    bar = Text()
    bar.append("█" * filled, style="grimoire.success")
    bar.append("░" * (width - filled), style="grimoire.muted")
    bar.append(f"  {done}/{total} ({ratio * 100:.0f}%)", style="grimoire.accent")
    return bar


# ── Progress ────────────────────────────────────────────────────────────────

@contextmanager
def progress_bar() -> Iterator[Progress]:
    """Provides a branded context manager for displaying task progress."""
    progress = Progress(
        SpinnerColumn(style="grimoire.accent"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None, complete_style="grimoire.secondary"),
        MofNCompleteColumn(),
        TextColumn("[grimoire.muted]·[/]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    with progress:
        yield progress
