"""Silence detection for finding clean cut points between reels."""

from .models import TitleMatch, TranscribedWord, TranscriptionResult


def _flatten_words(transcription: TranscriptionResult) -> list[TranscribedWord]:
    """Flatten all words from all segments."""
    words: list[TranscribedWord] = []
    for seg in transcription.segments:
        words.extend(seg.words)
    return words


def find_silence_cut_point(
    title_match: TitleMatch,
    all_words: list[TranscribedWord],
    min_silence: float = 0.5,
) -> float:
    """Find the silence gap before a title and return the cut point.

    Looks backward from title start to find previous word's end time.
    Cuts at midpoint of the silence gap.

    Args:
        title_match: The matched title with start_time.
        all_words: All transcribed words sorted by time.
        min_silence: Minimum gap in seconds to count as silence.

    Returns:
        Cut point timestamp in seconds.
    """
    title_start = title_match.start_time

    # Find the last word that ends before this title starts
    prev_word_end = 0.0
    for word in all_words:
        if word.end <= title_start:
            prev_word_end = word.end
        else:
            break

    gap = title_start - prev_word_end

    if gap >= min_silence:
        # Cut at midpoint of silence
        return prev_word_end + gap / 2
    else:
        # No clear silence — cut slightly before title
        return max(0.0, title_start - 0.2)


def compute_cut_points(
    matches: list[TitleMatch],
    transcription: TranscriptionResult,
    video_duration: float,
    min_silence: float = 0.5,
) -> list[tuple[TitleMatch, float, float]]:
    """Compute start/end cut points for each reel using silence detection.

    Args:
        matches: Title matches sorted by start_time.
        transcription: Full transcription with word timestamps.
        video_duration: Total video duration in seconds.
        min_silence: Minimum gap to count as silence.

    Returns:
        List of (TitleMatch, cut_start, cut_end) tuples.
    """
    all_words = _flatten_words(transcription)
    if not matches:
        return []

    # Find cut point for each title
    cut_points: list[float] = []
    for match in matches:
        cut = find_silence_cut_point(match, all_words, min_silence)
        cut_points.append(cut)

    # Build (match, start, end) tuples
    results: list[tuple[TitleMatch, float, float]] = []
    for i, match in enumerate(matches):
        start = cut_points[i]
        end = cut_points[i + 1] if i + 1 < len(cut_points) else video_duration
        results.append((match, round(start, 3), round(end, 3)))

    return results
