"""Training data for the input guard: attack families and normal prompts.

Attack families are YAML files (``attacks/public``, ``attacks/private``); see
``attacks/public/README.md``. Normal prompts are JSONL (``{"prompt": ...}`` per line) or
YAML (``prompts: [...]``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Family:
    name: str
    description: str
    examples: tuple[str, ...]


def load_families(*dirs: Path) -> list[Family]:
    """Every family in the given folders; missing folders are skipped."""
    families = []
    for folder in dirs:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "family" not in data or not data.get("examples"):
                raise ValueError(f"{path}: empty, or missing 'family' / 'examples'")
            examples = tuple(str(e) for e in data["examples"])
            families.append(Family(data["family"], data.get("description", ""), examples))
    names = [f.name for f in families]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"attack families defined twice: {', '.join(duplicates)}")
    return families


def load_prompts(*paths: Path) -> list[str]:
    """Normal prompts from JSONL or YAML files; missing files are skipped."""
    prompts: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            # split("\n"), not splitlines(): prompts may contain U+2028 and similar, which
            # splitlines() treats as line ends but JSON keeps inside strings.
            lines = text.split("\n")
            prompts += [json.loads(line)["prompt"] for line in lines if line.strip()]
        else:
            prompts += [str(p) for p in yaml.safe_load(text)["prompts"]]
    return prompts
