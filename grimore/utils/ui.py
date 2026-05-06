"""
User Interface (UI) and Theming for Project Grimore.
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

# Custom Rich theme with Grimore's signature "mystic" color palette
THEME = Theme({
    "grimore.primary":   "bold medium_purple3",
    "grimore.secondary": "medium_orchid",
    "grimore.accent":    "cyan",
    "grimore.success":   "bold green",
    "grimore.warning":   "bold yellow",
    "grimore.danger":    "bold red",
    "grimore.muted":     "grey50",
    "grimore.mystic":    "italic medium_orchid",
    "grimore.rune":      "bold cyan",
})

console = Console(theme=THEME, highlight=False)


# ── Brand / Headers ─────────────────────────────────────────────────────────

def render_banner(version: str = "v2.0") -> Panel:
    """Returns the large centered brand panel for the dashboard."""
    title = Text.assemble(
        ("🔮  ", "grimore.accent"),
        ("GRIMORE", "grimore.primary"),
        ("  ·  ", "grimore.muted"),
        (version, "grimore.accent"),
    )
    subtitle = Text("Automated Knowledge Engine", style="grimore.mystic")
    return Panel(
        Align.center(Group(title, subtitle)),
        border_style="grimore.primary",
        padding=(1, 2),
    )


def command_header(title: str, subtitle: str = "") -> None:
    """Prints a consistent branded header at the start of a command execution."""
    bar = Text()
    bar.append("✦ ", style="grimore.accent")
    bar.append(title, style="grimore.primary")
    if subtitle:
        bar.append("  · ", style="grimore.muted")
        bar.append(subtitle, style="grimore.mystic")
    console.print()
    console.print(bar)
    console.print(Text("─" * min(console.width - 2, 60), style="grimore.muted"))


def section(title: str) -> None:
    """Prints a section divider with a rune bullet point."""
    console.print()
    console.print(Text(f"  ⟡ {title}", style="grimore.rune"))


def tip(message: str) -> None:
    """Prints a light bulb hint message to the console."""
    console.print(
        Text.assemble(
            Text("  💡 ", style="grimore.muted"),
            Text.from_markup(message),
        )
    )


# ── Panels ──────────────────────────────────────────────────────────────────

def info_panel(message, title: str = "Info") -> Panel:
    return Panel(message, title=title, border_style="grimore.accent", padding=(0, 1))


def success_panel(message, title: str = "Done") -> Panel:
    return Panel(message, title=title, border_style="grimore.success", padding=(0, 1))


def warn_panel(message, title: str = "Warning") -> Panel:
    return Panel(message, title=title, border_style="grimore.warning", padding=(0, 1))


def error_panel(message, title: str = "Error") -> Panel:
    return Panel(message, title=title, border_style="grimore.danger", padding=(0, 1))


def oracle_panel(answer: str) -> Panel:
    """Custom panel for displaying the Oracle's answers with specific styling."""
    return Panel(
        Text(answer, style="white"),
        title=Text("🔮  Oracle", style="grimore.primary"),
        border_style="grimore.secondary",
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
    t.add_column(style="grimore.muted", no_wrap=True)
    t.add_column()
    for key, value in rows:
        if isinstance(value, (str, int, float)):
            value = Text(str(value), style="grimore.accent")
        t.add_row(key, value)
    return t


def tag_frequency_table(rows: list[tuple[str, int]], width: int = 20) -> Table:
    """Creates a table showing tag frequency with relative visual bars."""
    t = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="grimore.muted",
        padding=(0, 2),
        pad_edge=False,
    )
    t.add_column("Tag", style="grimore.primary", no_wrap=True)
    t.add_column("Uses", justify="right", style="grimore.accent", no_wrap=True)
    t.add_column("Frequency")

    if not rows:
        return t

    peak = max((count for _, count in rows), default=1) or 1
    for name, count in rows:
        ratio = count / peak if peak else 0.0
        filled = max(1, int(round(ratio * width))) if count else 0
        bar = Text()
        bar.append("█" * filled, style="grimore.secondary")
        bar.append("░" * (width - filled), style="grimore.muted")
        t.add_row(name, str(count), bar)
    return t


def coverage_bar(done: int, total: int, width: int = 20) -> Text:
    """Visual bar showing progress or coverage ratio."""
    if total <= 0:
        return Text("—", style="grimore.muted")
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(ratio * width))
    bar = Text()
    bar.append("█" * filled, style="grimore.success")
    bar.append("░" * (width - filled), style="grimore.muted")
    bar.append(f"  {done}/{total} ({ratio * 100:.0f}%)", style="grimore.accent")
    return bar


# ── Progress ────────────────────────────────────────────────────────────────

@contextmanager
def progress_bar() -> Iterator[Progress]:
    """Provides a branded context manager for displaying task progress."""
    progress = Progress(
        SpinnerColumn(style="grimore.accent"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None, complete_style="grimore.secondary"),
        MofNCompleteColumn(),
        TextColumn("[grimore.muted]·[/]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    with progress:
        yield progress
