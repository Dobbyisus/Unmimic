"""Attack memory: nearest neighbours over embeddings of known attacks and normal prompts.

A chunk is "near a known attack" when its closest attack is similar enough and closer than
its closest normal prompt. Learning a new attack is instant: store one more vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Match:
    attack_similarity: float
    attack_family: str
    benign_similarity: float


class AttackMemory:
    def __init__(self, attack_vectors: Any, attack_families: list[str], benign_vectors: Any):
        self.attacks = attack_vectors
        self.families = list(attack_families)
        self.benign = benign_vectors

    def add(self, vector: Any, family: str) -> None:
        import numpy as np

        self.attacks = np.vstack([self.attacks, vector[None, :]])
        self.families.append(family)

    def __len__(self) -> int:
        return len(self.families)

    def nearest(self, vectors: Any) -> list[Match]:
        """Closest attack and closest normal prompt for each row (unit vectors)."""
        attack_sims = vectors @ self.attacks.T
        benign_sims = vectors @ self.benign.T if len(self.benign) else None
        matches = []
        for row, sims in enumerate(attack_sims):
            best = int(sims.argmax())
            benign = float(benign_sims[row].max()) if benign_sims is not None else -1.0
            matches.append(Match(float(sims[best]), self.families[best], benign))
        return matches
