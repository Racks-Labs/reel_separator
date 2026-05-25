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


def _find_best_match(
    title_normalized: str,
    title_word_count: int,
    all_words: list[TranscribedWord],
    start_idx: int,
    end_idx: int | None = None,
    excluded_ranges: list[tuple[int, int]] | None = None,
) -> tuple[float, int, int]:
    """Find best matching window for a title in a word range.

    Returns (best_score, best_start_idx, best_end_idx).
    """
    if end_idx is None:
        end_idx = len(all_words)
    if excluded_ranges is None:
        excluded_ranges = []

    best_score = 0.0
    best_start_idx = -1
    best_end_idx = -1

    min_window = max(2, title_word_count - 1)
    max_window = title_word_count + 3

    for ws in range(min_window, max_window + 1):
        for i in range(start_idx, min(end_idx, len(all_words) - ws + 1)):
            # Skip excluded ranges
            if any(ex_start <= i <= ex_end for ex_start, ex_end in excluded_ranges):
                continue

            window_text = " ".join(w.word for w in all_words[i : i + ws])
            window_normalized = normalize_text(window_text)

            score = _score_match(title_normalized, window_normalized)

            if score > best_score:
                best_score = score
                best_start_idx = i
                best_end_idx = i + ws - 1

    return best_score, best_start_idx, best_end_idx


def find_title_matches(
    reels: list[ReelScript],
    transcription: TranscriptionResult,
    threshold: float = 70.0,
) -> list[TitleMatch]:
    """Find where each reel title is spoken in the transcription.

    Two-pass approach:
      Pass 1: Sequential matching (titles in markdown order, each search
              starts after previous match). Handles the common case.
      Pass 2: For titles missed in pass 1, search the entire transcription
              excluding already-matched positions. Handles out-of-order or
              garbled transcriptions.

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
    matched_indices: set[int] = set()  # reel indices already matched
    matched_ranges: list[tuple[int, int]] = []  # word ranges already used

    # --- Pass 1: Sequential matching ---
    search_start_idx = 0

    for reel in reels:
        title_normalized = normalize_text(reel.title)
        title_word_count = len(title_normalized.split())

        best_score, best_start_idx, best_end_idx = _find_best_match(
            title_normalized, title_word_count, all_words, search_start_idx
        )

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
            matched_indices.add(reel.index)
            matched_ranges.append((best_start_idx, best_end_idx))
            # Move search window past this match
            search_start_idx = best_end_idx + 1

    # --- Pass 2: Retry missed titles (non-sequential) ---
    missed_reels = [r for r in reels if r.index not in matched_indices]

    if missed_reels:
        console.print(
            f"[dim]Pass 2: retrying {len(missed_reels)} missed title(s) "
            f"without sequential constraint...[/dim]"
        )

        for reel in missed_reels:
            title_normalized = normalize_text(reel.title)
            title_word_count = len(title_normalized.split())

            # Search entire transcription, excluding already-matched ranges
            best_score, best_start_idx, best_end_idx = _find_best_match(
                title_normalized,
                title_word_count,
                all_words,
                start_idx=0,
                end_idx=None,
                excluded_ranges=matched_ranges,
            )

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
                matched_ranges.append((best_start_idx, best_end_idx))
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] Title not found: "
                    f"'{reel.title}' (best score: {best_score:.0f})"
                )

    # Sort all matches by start_time
    matches.sort(key=lambda m: m.start_time)

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
