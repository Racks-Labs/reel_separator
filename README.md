# reel-separator

Split a raw video into individual reels based on a markdown script file using Whisper speech-to-text.

You record one long video reading all your reel scripts (including titles). This tool detects when each title is spoken, finds the silence gaps between reels, and cuts the video into individual files named after each reel.

## Install

```bash
uv tool install git+https://github.com/Racks-Labs/reel_separator
```

## Update

```bash
uv tool upgrade reel-separator
```

## Requirements

- `ffmpeg` and `ffprobe` on PATH (`brew install ffmpeg` on Mac)
- For `--api`: an OpenAI API key

## Usage

```bash
# Basic (local Whisper, medium model)
reel-separator video.mp4 -s script.md

# Using OpenAI API (faster, no GPU needed)
reel-separator video.mp4 -s script.md --api sk-your-key-here

# Or with environment variable
export OPENAI_API_KEY=sk-...
reel-separator video.mp4 -s script.md

# Multiple videos, same script
reel-separator bruto1.mp4 bruto2.mp4 bruto3.mp4 -s script.md

# Dry run (transcribe + match, don't cut)
reel-separator video.mp4 -s script.md --dry-run

# Frame-accurate cuts (slower, re-encodes)
reel-separator video.mp4 -s script.md --precise

# Save transcription as JSON
reel-separator video.mp4 -s script.md --save-transcript
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-s, --script` | Path to markdown script file | **required** |
| `-o, --output` | Output directory | `reels/` next to video |
| `-m, --model` | Whisper model (tiny/base/small/medium/large) | `medium` |
| `-l, --language` | Language code | `es` |
| `-d, --device` | Compute device (cpu/cuda/mps) | auto |
| `-t, --threshold` | Fuzzy match threshold (0-100) | `70` |
| `--min-silence` | Min silence gap for cut detection (seconds) | `0.5` |
| `--api` | Use OpenAI API (pass key or set OPENAI_API_KEY) | off |
| `--precise` | Frame-accurate cuts (re-encodes) | off |
| `--dry-run` | Show matches and cut plan without cutting | off |
| `--save-transcript` | Save transcription JSON | off |

## Markdown script format

```markdown
# Title of first reel

Hooks
...

Body
...

CTA
...

---

# Title of second reel

Hooks
...
```

Each reel section separated by `---`. Title = first `# Heading` in each section.

## Output

Single video:
```
reels/
  01_Title_of_first_reel.mp4
  02_Title_of_second_reel.mp4
```

Multiple videos:
```
reels/
  bruto1/
    01_Title_of_first_reel.mp4
    02_Title_of_second_reel.mp4
  bruto2/
    03_Title_of_third_reel.mp4
```

## How it works

1. Parses markdown to extract reel titles
2. Transcribes video audio with Whisper (local or API)
3. Fuzzy-matches each title in the transcription (sequential order)
4. Detects silence gaps before each title for clean cut points
5. Splits video with ffmpeg at those points
