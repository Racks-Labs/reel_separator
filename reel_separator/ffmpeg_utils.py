"""Low-level ffmpeg/ffprobe subprocess wrappers."""

import json
import subprocess
from pathlib import Path


def get_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def get_video_extension(video_path: Path) -> str:
    """Get video file extension (including dot)."""
    return video_path.suffix.lower()


def cut_reel(
    video_path: Path,
    output_path: Path,
    start: float,
    end: float,
    precise: bool = False,
) -> Path:
    """Cut a segment from the video.

    Args:
        video_path: Source video.
        output_path: Output file path.
        start: Start time in seconds.
        end: End time in seconds.
        precise: If True, re-encode for frame-accurate cuts (slower).

    Returns:
        Path to output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if precise:
        # Re-encode for frame-accurate cuts
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            str(start),
            "-to",
            str(end),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    else:
        # Stream copy — fast but cuts on keyframes
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
