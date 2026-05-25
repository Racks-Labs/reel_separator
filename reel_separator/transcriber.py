"""Whisper transcription with word-level timestamps (local or API)."""

import os
import subprocess
import tempfile
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

# OpenAI API max file size: 25MB
API_MAX_SIZE_BYTES = 25 * 1024 * 1024


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


def _extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from video as mp3 for API upload."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "libmp3lame",
            "-b:a", "64k",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    return output_path


def _split_audio(audio_path: Path, max_size: int = API_MAX_SIZE_BYTES) -> list[tuple[Path, float]]:
    """Split audio into chunks under max_size. Returns [(chunk_path, offset_seconds)]."""
    file_size = audio_path.stat().st_size
    if file_size <= max_size:
        return [(audio_path, 0.0)]

    # Get duration
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    import json
    duration = float(json.loads(result.stdout)["format"]["duration"])

    # Calculate chunk duration proportional to size limit
    num_chunks = (file_size // max_size) + 1
    chunk_duration = duration / num_chunks

    chunks: list[tuple[Path, float]] = []
    for i in range(num_chunks):
        offset = i * chunk_duration
        chunk_path = audio_path.parent / f"chunk_{i:03d}{audio_path.suffix}"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(offset),
                "-t", str(chunk_duration + 1),  # +1s overlap for continuity
                "-i", str(audio_path),
                "-c", "copy",
                str(chunk_path),
            ],
            capture_output=True, check=True,
        )
        chunks.append((chunk_path, offset))

    return chunks


def _transcribe_api(
    video_path: Path,
    language: str = "es",
    initial_prompt: str | None = None,
) -> TranscriptionResult:
    """Transcribe using OpenAI Whisper API with word timestamps."""
    try:
        from openai import OpenAI
    except ImportError:
        console.print("[red]Error:[/red] openai package not installed. Run: pip install openai")
        raise SystemExit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] OPENAI_API_KEY not set. "
            "Export it: [cyan]export OPENAI_API_KEY=sk-...[/cyan]"
        )
        raise SystemExit(1)

    client = OpenAI(api_key=api_key)

    console.print("[bold]Backend:[/bold] OpenAI API (whisper-1)")
    console.print(f"[bold]Transcribing:[/bold] {video_path.name}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Extract audio as lightweight mp3
        audio_path = tmp_dir / "audio.mp3"
        with console.status("[bold cyan]Extracting audio...", spinner="dots"):
            _extract_audio(video_path, audio_path)

        audio_size = audio_path.stat().st_size / (1024 * 1024)
        console.print(f"Audio extracted: {audio_size:.1f}MB")

        # Split if needed
        chunks = _split_audio(audio_path)
        if len(chunks) > 1:
            console.print(f"Split into {len(chunks)} chunks (API limit: 25MB)")

        all_segments: list[TranscribedSegment] = []
        full_text_parts: list[str] = []

        for chunk_path, offset in chunks:
            with console.status(
                f"[bold cyan]Transcribing chunk (offset {offset:.0f}s)..."
                if len(chunks) > 1 else "[bold cyan]Transcribing via API...",
                spinner="dots",
            ):
                with open(chunk_path, "rb") as f:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                        prompt=initial_prompt,
                    )

            # Parse words
            for word_data in getattr(response, "words", []) or []:
                # API returns word objects with start/end/word
                pass  # We'll build segments from them below

            # Parse segments and build word-level data
            for seg in getattr(response, "segments", []) or []:
                text = seg.get("text", "").strip() if isinstance(seg, dict) else getattr(seg, "text", "").strip()
                seg_start = (seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start", 0)) + offset
                seg_end = (seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end", 0)) + offset

                if text:
                    all_segments.append(
                        TranscribedSegment(
                            text=text,
                            start=round(seg_start, 3),
                            end=round(seg_end, 3),
                            words=[],  # Will fill from word-level data
                        )
                    )
                    full_text_parts.append(text)

            # Build words from API word-level timestamps
            api_words = getattr(response, "words", []) or []
            if api_words:
                # Replace segments with word-enriched versions
                word_idx = 0
                enriched_segments: list[TranscribedSegment] = []

                for seg in all_segments:
                    seg_words: list[TranscribedWord] = []
                    while word_idx < len(api_words):
                        w = api_words[word_idx]
                        w_start = (w.get("start", 0) if isinstance(w, dict) else getattr(w, "start", 0)) + offset
                        w_end = (w.get("end", 0) if isinstance(w, dict) else getattr(w, "end", 0)) + offset
                        w_text = (w.get("word", "") if isinstance(w, dict) else getattr(w, "word", "")).strip()

                        if w_start > seg.end + 0.5:
                            break

                        seg_words.append(
                            TranscribedWord(
                                word=w_text,
                                start=round(w_start, 3),
                                end=round(w_end, 3),
                                probability=1.0,
                            )
                        )
                        word_idx += 1

                    if seg_words:
                        enriched_segments.append(
                            TranscribedSegment(
                                text=seg.text,
                                start=seg_words[0].start,
                                end=seg_words[-1].end,
                                words=seg_words,
                            )
                        )
                    else:
                        enriched_segments.append(seg)

                all_segments = enriched_segments

            # Clean up chunk files (not the main audio)
            if chunk_path != audio_path:
                chunk_path.unlink(missing_ok=True)

    total_words = sum(len(s.words) for s in all_segments)
    console.print(
        f"[green]Transcription done.[/green] "
        f"{total_words} words in {len(all_segments)} segments."
    )

    return TranscriptionResult(
        segments=all_segments,
        language=language,
        full_text=" ".join(full_text_parts),
    )


def _transcribe_local(
    video_path: Path,
    model_size: str = "medium",
    language: str = "es",
    initial_prompt: str | None = None,
    device: str | None = None,
) -> TranscriptionResult:
    """Transcribe using local Whisper model with word timestamps."""
    import whisper

    if device is None:
        device = _detect_device()

    console.print(f"[bold]Backend:[/bold] Local Whisper ({model_size} on {device})")
    console.print(f"[bold]Transcribing:[/bold] {video_path.name}")

    with console.status("[bold cyan]Loading model...", spinner="dots"):
        model = whisper.load_model(model_size, device=device)

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


def transcribe_video(
    video_path: Path,
    model_size: str = "medium",
    language: str = "es",
    initial_prompt: str | None = None,
    device: str | None = None,
    use_api: bool = False,
) -> TranscriptionResult:
    """Transcribe video audio using Whisper (local or API).

    Args:
        video_path: Path to video file.
        model_size: Whisper model for local (tiny/base/small/medium/large).
        language: Language code.
        initial_prompt: Hint prompt containing reel titles.
        device: Compute device for local (cpu/cuda/mps).
        use_api: Use OpenAI API instead of local model.

    Returns:
        TranscriptionResult with segments and word timestamps.
    """
    if use_api:
        return _transcribe_api(video_path, language, initial_prompt)
    return _transcribe_local(video_path, model_size, language, initial_prompt, device)
