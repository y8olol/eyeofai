from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


EYE_ART = [
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣶⠟⠃⠉⠀⠀⠀⠈⠙⢶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⢀⢤⣺⣿⣿⣿⠇⠀⠀⠀⠀⠀⢀⣠⣤⣬⣿⣿⣷⣄⣂⠀⠀⠀⠀⠀",
    "⠀⠀⢖⢼⡺⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠈⠹⠿⠿⣿⣿⣿⣿⣚⠸⣣⠆⠀⠀",
    "⠀⠀⠀⠀⠑⠨⢟⣿⣿⣿⣆⠠⡀⠀⠀⠀⠀⠀⠀⢠⣿⣿⠿⠣⠙⠁⠁⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠦⣈⡱⡲⢖⣌⠅⠴⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
]

RED_WHITE = ["#ff2d2d", "#ff5b5b", "#ff8a8a", "#ffb8b8", "#ffe6e6", "#ffffff"]


def _gradient_text(lines: list[str], palette: list[str]) -> Text:
    text = Text()
    total = max(1, len(lines) - 1)
    for idx, line in enumerate(lines):
        color = palette[int((idx / total) * (len(palette) - 1))]
        text.append(line + "\n", style=f"bold {color}")
    return text


def print_eye_header(console: Console) -> None:
    eye = _gradient_text(EYE_ART, RED_WHITE)
    subtitle = Text("EyeOfAI - all-seeing coordinate hunter", style="bold white")
    group = Group(eye, subtitle)
    panel = Panel(group, border_style="#ff5b5b", title="[bold white]EyeOfAI[/bold white]")
    console.print(panel)


def print_run_info(
    console: Console,
    *,
    query: str,
    source: str,
    selected_models: list[str],
    free_scope: str,
    billing_mode: str,
    strategy: str,
    attempted: int,
    skipped_non_vision: int,
    concurrency: int,
    max_retries: int,
) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold #ff8a8a", width=20)
    table.add_column(style="white")
    table.add_row("Query", query)
    table.add_row("Source", source)
    table.add_row("Model Scope", free_scope)
    table.add_row("Billing Mode", billing_mode)
    table.add_row("Strategy", strategy)
    table.add_row("Attempted Models", str(attempted))
    table.add_row("Skipped Non-Vision", str(skipped_non_vision))
    table.add_row("Concurrency", str(concurrency))
    table.add_row("Max Retries", str(max_retries))
    table.add_row("Models", ", ".join(selected_models) if selected_models else "-")
    panel = Panel(table, border_style="#ff2d2d", title="[bold white]Run Info[/bold white]")
    console.print(panel)


def print_result_summary(
    console: Console,
    *,
    winner: dict | None,
    confidence: float,
    agreement_attempted: float,
    agreement_successful: float,
    chosen_models: list[str],
    uncertain_reason: str | None,
    failed_count: int,
) -> None:
    status = "UNCERTAIN" if uncertain_reason else "OK"
    status_color = "#ff4d4d" if uncertain_reason else "#ffffff"
    lines = [
        f"[bold #ff8a8a]Status:[/bold #ff8a8a] [{status_color}]{status}[/{status_color}]",
        f"[bold #ff8a8a]Confidence:[/bold #ff8a8a] [white]{confidence:.3f}[/white]",
        f"[bold #ff8a8a]Agreement (attempted):[/bold #ff8a8a] [white]{agreement_attempted:.3f}[/white]",
        f"[bold #ff8a8a]Agreement (successful):[/bold #ff8a8a] [white]{agreement_successful:.3f}[/white]",
        f"[bold #ff8a8a]Failed Models:[/bold #ff8a8a] [white]{failed_count}[/white]",
        f"[bold #ff8a8a]Chosen Models:[/bold #ff8a8a] [white]{', '.join(chosen_models) if chosen_models else '-'}[/white]",
        f"[bold #ff8a8a]Winner BBox:[/bold #ff8a8a] [white]{winner if winner else '-'}[/white]",
    ]
    if uncertain_reason:
        lines.append(f"[bold #ff8a8a]Reason:[/bold #ff8a8a] [white]{uncertain_reason}[/white]")
    panel = Panel("\n".join(lines), border_style="#ff2d2d", title="[bold white]Result Summary[/bold white]")
    console.print(panel)
