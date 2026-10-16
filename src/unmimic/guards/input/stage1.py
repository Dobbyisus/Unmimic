"""Stage 1: fast filter run on every request.

Logistic regression on multilingual sentence embeddings, so a translated attack scores
like its English original. Returns P(attack) per embedded chunk.
"""

from __future__ import annotations

from typing import Any


class Stage1:
    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        # Attacks are rare next to normal prompts; weight the classes evenly.
        self.model = LogisticRegression(class_weight="balanced", max_iter=2000)

    def fit(self, attack_vectors: Any, benign_vectors: Any) -> Stage1:
        import numpy as np

        x = np.vstack([attack_vectors, benign_vectors])
        y = np.r_[np.ones(len(attack_vectors)), np.zeros(len(benign_vectors))]
        self.model.fit(x, y)
        return self

    def score(self, vectors: Any) -> Any:
        return self.model.predict_proba(vectors)[:, 1]
