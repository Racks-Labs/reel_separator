"""Whisper transcription with word-level timestamps."""

import os
import warnings
from pathlib import Path

# Suppress noisy warnings before torch/whisper imports
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning)

from rich.console import Console

from .models import TranscribedSegment, TranscribedWord, TranscriptionResult

console = Console()


def _detect_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "cpu"  # Whisper has issues with MPS, use CPU for now
    except ImportError:
        pass
    return "cpu"


def transcribe_video(
    video_path: Path,
    model_size: str = "medium",
    language: str = "es",
    initial_prompt: str | None = None,
    device: str | None = None,
) -> TranscriptionResult:
    """Transcribe video audio using Whisper with word-level timestamps.

    Args:
        video_path: Path to video file.
        model_size: Whisper model (tiny/base/small/medium/large).
        language: Language code.
        initial_prompt: Hint prompt containing reel titles to bias recognition.
        device: Compute device (cpu/cuda/mps). Auto-detected if None.

    Returns:
        TranscriptionResult with segments and word timestamps.
    """
    import whisper

    if device is None:
        device = _detect_device()

    console.print(f"[bold]Loading Whisper model:[/bold] {model_size} on {device}")
    model = whisper.load_model(model_size, device=device)

    console.print(f"[bold]Transcribing:[/bold] {video_path.name}")
    with console.status("[bold cyan]Transcribing audio...", spinner="dots"):
        result = model.transcribe(
            str(video_path),
            language=language,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            temperature=0.0,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )

    segments: list[TranscribedSegment] = []
    full_text_parts: list[str] = []

    for seg in result.get("segments", []):
        words: list[TranscribedWord] = []
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            words.append(
                TranscribedWord(
                    word=w["word"].strip(),
                    start=round(w["start"], 3),
                    end=round(w["end"], 3),
                    probability=round(w.get("probability", 0.0), 3),
                )
            )

        text = seg.get("text", "").strip()
        if words:
            segments.append(
                TranscribedSegment(
                    text=text,
                    start=words[0].start,
                    end=words[-1].end,
                    words=words,
                )
            )
            full_text_parts.append(text)

    total_words = sum(len(s.words) for s in segments)
    console.print(
        f"[green]Transcription done.[/green] "
        f"{total_words} words in {len(segments)} segments."
    )

    return TranscriptionResult(
        segments=segments,
        language=language,
        full_text=" ".join(full_text_parts),
    )
