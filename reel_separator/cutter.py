"""Cut video into individual reels based on matched titles and silence detection."""

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .ffmpeg_utils import cut_reel, get_duration, get_video_extension
from .models import ReelCut, TitleMatch, TranscriptionResult
from .silence import compute_cut_points
from .utils import sanitize_filename

console = Console()


def plan_cuts(
    matches: list[TitleMatch],
    transcription: TranscriptionResult,
    video_path: Path,
    output_dir: Path,
    min_silence: float = 0.5,
) -> list[ReelCut]:
    """Compute cut plan from matched titles.

    Args:
        matches: Title matches sorted by start_time.
        transcription: Full transcription.
        video_path: Source video path.
        output_dir: Output directory for reels.
        min_silence: Minimum silence gap for cut detection.

    Returns:
        List of ReelCut with computed timestamps and filenames.
    """
    duration = get_duration(video_path)
    ext = get_video_extension(video_path)

    cut_data = compute_cut_points(matches, transcription, duration, min_silence)

    cuts: list[ReelCut] = []
    for match, start, end in cut_data:
        filename = sanitize_filename(match.title, match.reel_index, ext)
        cuts.append(
            ReelCut(
                reel_index=match.reel_index,
                title=match.title,
                start=start,
                end=end,
                duration=round(end - start, 3),
                output_filename=filename,
            )
        )

    return cuts


def print_cut_plan(cuts: list[ReelCut]) -> None:
    """Print a Rich table showing the cut plan."""
    table = Table(title="Cut Plan", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Filename", style="cyan")

    for cut in cuts:
        table.add_row(
            str(cut.reel_index + 1),
            cut.title,
            _format_time(cut.start),
            _format_time(cut.end),
            _format_time(cut.duration),
            cut.output_filename,
        )

    console.print(table)


def execute_cuts(
    cuts: list[ReelCut],
    video_path: Path,
    output_dir: Path,
    fast: bool = False,
) -> list[Path]:
    """Execute all cuts and produce individual reel files.

    Args:
        cuts: Cut plan.
        video_path: Source video.
        output_dir: Output directory.
        fast: Use stream copy instead of re-encoding (faster but may glitch).

    Returns:
        List of output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    failed: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Cutting reels...", total=len(cuts))

        for cut in cuts:
            output_path = output_dir / cut.output_filename
            progress.update(task, description=f"Cutting: {cut.title[:40]}")

            try:
                cut_reel(video_path, output_path, cut.start, cut.end, fast)
                output_paths.append(output_path)
            except Exception as e:
                failed.append(f"{cut.title}: {e}")
                console.print(
                    f"[red]Error cutting '{cut.title}':[/red] {e}"
                )

            progress.advance(task)

    if failed:
        console.print(f"\n[red]{len(failed)} cut(s) failed:[/red]")
        for f in failed:
            console.print(f"  - {f}")

    return output_paths


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS.ms."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"
