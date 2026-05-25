"""Data models for reel-separator pipeline."""

from pydantic import BaseModel


class ReelScript(BaseModel):
    """A single reel script parsed from the markdown file."""

    index: int
    title: str


class TranscribedWord(BaseModel):
    """A single word with timing from Whisper."""

    word: str
    start: float
    end: float
    probability: float


class TranscribedSegment(BaseModel):
    """A transcription segment containing words."""

    text: str
    start: float
    end: float
    words: list[TranscribedWord]


class TranscriptionResult(BaseModel):
    """Full transcription output from Whisper."""

    segments: list[TranscribedSegment]
    language: str
    full_text: str


class TitleMatch(BaseModel):
    """A matched title found in the transcription."""

    reel_index: int
    title: str
    matched_text: str
    score: float
    start_time: float
    end_time: float


class ReelCut(BaseModel):
    """A computed cut point for extracting a reel."""

    reel_index: int
    title: str
    start: float
    end: float
    duration: float
    output_filename: str
