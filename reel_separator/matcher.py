"""Fuzzy match reel titles against Whisper transcription."""

from rapidfuzz import fuzz
from rich.console import Console
from rich.table import Table

from .models import ReelScript, TitleMatch, TranscribedWord, TranscriptionResult
from .utils import normalize_text

console = Console()


def _flatten_words(transcription: TranscriptionResult) -> list[TranscribedWord]:
    """Flatten all words from all segments into a single list."""
    words: list[TranscribedWord] = []
    for seg in transcription.segments:
        words.extend(seg.words)
    return words


def _score_match(title_normalized: str, window_normalized: str) -> float:
    """Score a title against a window using combined fuzzy metrics.

    Uses weighted combination of ratio (order-sensitive, length-sensitive)
    and token_sort_ratio (handles reordering but respects length).
    Avoids token_set_ratio which is too permissive for short strings.
    """
    # ratio: penalizes length differences and respects order
    r = fuzz.ratio(title_normalized, window_normalized)
    # token_sort_ratio: sorts tokens first, then does ratio — handles reordering
    tsr = fuzz.token_sort_ratio(title_normalized, window_normalized)
    # Weight: favor ratio (stricter) but allow some flexibility from token_sort
    return 0.6 * r + 0.4 * tsr


def find_title_matches(
    reels: list[ReelScript],
    transcription: TranscriptionResult,
    threshold: float = 70.0,
) -> list[TitleMatch]:
    """Find where each reel title is spoken in the transcription.

    Uses sequential sliding window fuzzy matching: processes titles in
    markdown order and restricts each search to AFTER the previous match.
    This prevents duplicate position matches and ensures correct ordering.

    Args:
        reels: Parsed reel scripts with titles.
        transcription: Whisper transcription with word timestamps.
        threshold: Minimum fuzzy match score (0-100) to accept a match.

    Returns:
        List of TitleMatch sorted by start_time.
    """
    all_words = _flatten_words(transcription)
    if not all_words:
        console.print("[red]Error:[/red] No words in transcription.")
        return []

    matches: list[TitleMatch] = []
    search_start_idx = 0  # Restrict search to after previous match

    for reel in reels:
        title_normalized = normalize_text(reel.title)
        title_word_count = len(title_normalized.split())

        best_score = 0.0
        best_start_idx = -1
        best_end_idx = -1

        # Try window sizes around title word count
        min_window = max(2, title_word_count - 1)
        max_window = title_word_count + 3

        for ws in range(min_window, max_window + 1):
            for i in range(search_start_idx, len(all_words) - ws + 1):
                window_text = " ".join(w.word for w in all_words[i : i + ws])
                window_normalized = normalize_text(window_text)

                score = _score_match(title_normalized, window_normalized)

                if score > best_score:
                    best_score = score
                    best_start_idx = i
                    best_end_idx = i + ws - 1

        if best_score >= threshold and best_start_idx >= 0:
            matched_text = " ".join(
                w.word for w in all_words[best_start_idx : best_end_idx + 1]
            )
            matches.append(
                TitleMatch(
                    reel_index=reel.index,
                    title=reel.title,
                    matched_text=matched_text,
                    score=round(best_score, 1),
                    start_time=all_words[best_start_idx].start,
                    end_time=all_words[best_end_idx].end,
                )
            )
            # Move search window past this match for next title
            search_start_idx = best_end_idx + 1
        else:
            console.print(
                f"[yellow]Warning:[/yellow] Title not found: "
                f"'{reel.title}' (best score: {best_score:.0f})"
            )

    return matches


def print_matches_table(matches: list[TitleMatch]) -> None:
    """Print a Rich table showing match results."""
    table = Table(title="Title Matches", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Matched Text", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")

    for m in matches:
        score_style = "green" if m.score >= 85 else "yellow" if m.score >= 70 else "red"
        table.add_row(
            str(m.reel_index + 1),
            m.title,
            m.matched_text,
            f"[{score_style}]{m.score:.0f}[/{score_style}]",
            _format_time(m.start_time),
            _format_time(m.end_time),
        )

    console.print(table)


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS.ms."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"
