"""CLI entry point for reel-separator."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .cutter import execute_cuts, plan_cuts, print_cut_plan
from .markdown_parser import parse_scripts
from .matcher import find_title_matches, print_matches_table
from .transcriber import transcribe_video

app = typer.Typer(
    name="reel-separator",
    help="Split raw video(s) into individual reels based on a markdown script.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _process_single_video(
    video: Path,
    reels: list,
    titles_prompt: str,
    output_dir: Path,
    model: str,
    language: str,
    device: Optional[str],
    threshold: float,
    min_silence: float,
    precise: bool,
    dry_run: bool,
    save_transcript: bool,
    use_api: bool = False,
) -> list[Path]:
    """Process a single video through the full pipeline.

    Returns list of output file paths.
    """
    # --- Transcribe ---
    console.print(Panel(f"[bold]Transcribing:[/bold] {video.name}"))

    transcription = transcribe_video(
        video_path=video,
        model_size=model,
        language=language,
        initial_prompt=titles_prompt,
        device=device,
        use_api=use_api,
    )

    if save_transcript:
        transcript_path = output_dir / f"{video.stem}_transcript.json"
        transcript_path.mkdir(parents=True, exist_ok=True) if not transcript_path.parent.exists() else None
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            json.dumps(transcription.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"Transcript saved: [cyan]{transcript_path}[/cyan]\n")

    # --- Match titles ---
    console.print(Panel(f"[bold]Matching titles:[/bold] {video.name}"))
    matches = find_title_matches(reels, transcription, threshold)

    if not matches:
        console.print(
            f"[red]Error:[/red] No titles matched in {video.name}. "
            "Try lowering --threshold or using a larger Whisper model."
        )
        console.print(f"\n[dim]Full transcription:[/dim]\n{transcription.full_text}")
        return []

    print_matches_table(matches)

    if len(matches) < len(reels):
        console.print(
            f"\n[yellow]Warning:[/yellow] Only {len(matches)}/{len(reels)} titles matched."
        )
    console.print()

    # --- Compute cut plan ---
    console.print(Panel(f"[bold]Computing cut points:[/bold] {video.name}"))
    cuts = plan_cuts(matches, transcription, video, output_dir, min_silence)
    print_cut_plan(cuts)
    console.print()

    # --- Execute cuts ---
    if dry_run:
        console.print("[yellow]Dry run — skipping cuts.[/yellow]")
        return []

    console.print(Panel(f"[bold]Cutting reels:[/bold] {video.name}"))
    return execute_cuts(cuts, video, output_dir, precise)


@app.command()
def separate(
    videos: list[Path] = typer.Argument(
        ...,
        help="Path(s) to raw video file(s)",
        exists=True,
        dir_okay=False,
    ),
    script: Path = typer.Option(
        ...,
        "--script",
        "-s",
        help="Path to the markdown script file",
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: reels/ next to first video)",
    ),
    model: str = typer.Option(
        "medium",
        "--model",
        "-m",
        help="Whisper model: tiny/base/small/medium/large",
    ),
    language: str = typer.Option(
        "es",
        "--language",
        "-l",
        help="Language code for transcription",
    ),
    device: str = typer.Option(
        None,
        "--device",
        "-d",
        help="Compute device: cpu/cuda/mps (auto-detected if omitted)",
    ),
    threshold: float = typer.Option(
        70.0,
        "--threshold",
        "-t",
        help="Fuzzy match threshold (0-100)",
    ),
    min_silence: float = typer.Option(
        0.5,
        "--min-silence",
        help="Minimum silence gap in seconds for cut detection",
    ),
    precise: bool = typer.Option(
        False,
        "--precise",
        help="Frame-accurate cuts (re-encodes, slower)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Transcribe and match but don't cut",
    ),
    save_transcript: bool = typer.Option(
        False,
        "--save-transcript",
        help="Save transcription as JSON file",
    ),
    use_api: bool = typer.Option(
        False,
        "--api",
        help="Use OpenAI API instead of local Whisper (requires OPENAI_API_KEY)",
    ),
) -> None:
    """Split raw video(s) into individual reels.

    Accepts one or multiple video files. Each video is transcribed with
    Whisper, titles from the markdown script are matched, and the video
    is split at silence gaps between reels.

    Examples:

        reel-separator video.mp4 -s script.md

        reel-separator bruto1.mp4 bruto2.mp4 bruto3.mp4 -s script.md
    """
    # Resolve base output directory
    if output_dir is None:
        output_dir = videos[0].parent / "reels"

    multi_video = len(videos) > 1

    # --- Step 1: Parse markdown ---
    console.print(Panel("[bold]Step 1:[/bold] Parsing markdown script"))
    reels = parse_scripts(script)

    if not reels:
        console.print("[red]Error:[/red] No reel titles found in markdown.")
        raise typer.Exit(1)

    console.print(f"Found [bold]{len(reels)}[/bold] reel(s):")
    for r in reels:
        console.print(f"  {r.index + 1}. {r.title}")
    console.print()

    if multi_video:
        console.print(f"Processing [bold]{len(videos)}[/bold] video(s):")
        for v in videos:
            console.print(f"  - {v.name}")
        console.print()

    # Build initial prompt with all titles
    titles_prompt = ". ".join(r.title for r in reels) + "."

    # --- Process each video ---
    all_output_paths: list[Path] = []
    video_results: dict[str, list[Path]] = {}

    for i, video in enumerate(videos):
        if multi_video:
            console.print(Rule(f"Video {i + 1}/{len(videos)}: {video.name}"))

        # Output subfolder per video when multiple videos
        if multi_video:
            video_output = output_dir / video.stem
        else:
            video_output = output_dir

        paths = _process_single_video(
            video=video,
            reels=reels,
            titles_prompt=titles_prompt,
            output_dir=video_output,
            model=model,
            language=language,
            device=device,
            threshold=threshold,
            min_silence=min_silence,
            precise=precise,
            dry_run=dry_run,
            save_transcript=save_transcript,
            use_api=use_api,
        )

        all_output_paths.extend(paths)
        video_results[video.name] = paths

    # --- Summary ---
    if dry_run:
        console.print("\n[yellow]Dry run complete — no files created.[/yellow]")
        raise typer.Exit(0)

    console.print()
    summary_lines = [
        f"[bold green]Done![/bold green] "
        f"{len(all_output_paths)} reel(s) from {len(videos)} video(s).\n"
    ]
    summary_lines.append(f"Output: [cyan]{output_dir}[/cyan]\n")

    for video_name, paths in video_results.items():
        if multi_video:
            summary_lines.append(f"\n[bold]{video_name}[/bold] ({len(paths)} reels):")
        for p in paths:
            summary_lines.append(f"  - {p.name}")

    console.print(Panel("\n".join(summary_lines), title="Summary"))


if __name__ == "__main__":
    app()
