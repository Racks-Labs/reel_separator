"""Parse a markdown script file into a list of reel titles."""

import re
from pathlib import Path

from rich.console import Console

from .models import ReelScript

console = Console()


def parse_scripts(script_path: Path) -> list[ReelScript]:
    """Parse markdown file and extract reel titles.

    Expects sections separated by --- with # Title as first heading.
    """
    text = script_path.read_text(encoding="utf-8")

    # Split on horizontal rules (---)
    blocks = re.split(r"^\s*---\s*$", text, flags=re.MULTILINE)

    reels: list[ReelScript] = []

    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue

        # Find first H1 heading
        match = re.search(r"^#\s+(.+)$", block, flags=re.MULTILINE)
        if not match:
            console.print(
                f"[yellow]Warning:[/yellow] Block {i + 1} has no # title, skipping"
            )
            continue

        title = match.group(1).strip()
        reels.append(ReelScript(index=len(reels), title=title))

    return reels
