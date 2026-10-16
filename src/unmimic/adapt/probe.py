"""Probe: find where a model puts its reasoning and write its format config.

Sends ordinary questions (never attacks), looks at where each reply's reasoning appears,
and keeps the format most replies agree on.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx

from unmimic.reasoning import KNOWN_FIELDS, ModelFormat

PROMPTS = [
    "What is 17 * 23?",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
    "Is 221 a prime number?",
    "A shirt costs $40 after a 20% discount. What was the original price?",
    "What is the next number in the sequence 2, 6, 12, 20, 30?",
    "How many minutes are there in 3.5 hours?",
    "All bloops are razzies and all razzies are lazzies. Are all bloops lazzies?",
    "What is the sum of the interior angles of a hexagon?",
    "Solve for x: 3x + 7 = 25.",
    "A rectangle has a perimeter of 30 and a width of 5. What is its area?",
    "Which is larger, 2^10 or 10^3?",
    "Tom is older than Ann. Ann is older than Ben. Who is the youngest?",
    "What is 15% of 240?",
    "How many days are there in four non-leap years?",
    "Write a Python expression that reverses the string s.",
    "What is the greatest common divisor of 84 and 126?",
    "If today is Wednesday, what day will it be in 10 days?",
    "A recipe needs 3 eggs for 12 cookies. How many eggs are needed for 30 cookies?",
    "Convert 0.375 to a fraction in lowest terms.",
    "What is the remainder when 1000 is divided by 7?",
]

TAG_PAIRS = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
]

# Longest first, so "**Final Answer:**" wins over "Final Answer:".
MARKERS = [
    "**Final Answer:**",
    "**Final Answer**",
    "### Final Answer",
    "Final Answer:",
    "Final answer:",
]


class ProbeError(Exception):
    pass


def observe(message: dict[str, Any]) -> ModelFormat:
    """The format one reply shows."""
    for name in KNOWN_FIELDS:
        if message.get(name):
            return ModelFormat(kind="field", field=name)
    content = message.get("content") or ""
    for open_tag, close_tag in TAG_PAIRS:
        opened = open_tag in content
        if opened or close_tag in content:
            return ModelFormat(
                kind="tag", open_tag=open_tag, close_tag=close_tag, implicit_open=not opened
            )
    for marker in MARKERS:
        if marker in content:
            return ModelFormat(kind="marker", marker=marker)
    return ModelFormat(kind="none")


def vote(model: str, observations: list[ModelFormat]) -> ModelFormat:
    """The format most replies agree on.

    ``implicit_open`` is left out of the vote: if any reply lacked the opening tag, the
    result assumes it can be missing (replies that do include it still split correctly).
    """
    if not observations:
        raise ProbeError("no replies to inspect")

    def key(fmt: ModelFormat) -> str:
        return fmt.model_copy(update={"implicit_open": False}).model_dump_json()

    counts = Counter(key(obs) for obs in observations)
    winner, matches = counts.most_common(1)[0]
    implicit = any(obs.implicit_open for obs in observations if key(obs) == winner)
    return ModelFormat.model_validate_json(winner).model_copy(
        update={
            "model": model,
            "implicit_open": implicit,
            "samples": len(observations),
            "matches": matches,
        }
    )


def pick_model(client: httpx.Client) -> str:
    """The only model the server offers; otherwise ask the user to choose."""
    response = client.get("/models")
    response.raise_for_status()
    ids = [m["id"] for m in response.json().get("data", [])]
    if len(ids) == 1:
        return ids[0]
    if not ids:
        raise ProbeError("the server lists no models; pass --model")
    raise ProbeError(f"the server offers several models; pass --model (one of: {', '.join(ids)})")


def probe(
    client: httpx.Client,
    model: str,
    samples: int = len(PROMPTS),
    max_tokens: int = 1024,
    on_reply: Callable[[int, int], None] | None = None,
) -> ModelFormat:
    """Ask ``samples`` ordinary questions and return the format the replies agree on."""
    prompts = PROMPTS[: max(1, min(samples, len(PROMPTS)))]
    observations = []
    for done, prompt in enumerate(prompts, start=1):
        response = client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        observations.append(observe(response.json()["choices"][0]["message"]))
        if on_reply:
            on_reply(done, len(prompts))
    return vote(model, observations)


def describe(fmt: ModelFormat) -> str:
    if fmt.kind == "field":
        return f"separate API field '{fmt.field}'"
    if fmt.kind == "tag":
        added = " (opening tag added by the chat template)" if fmt.implicit_open else ""
        return f"inside {fmt.open_tag}...{fmt.close_tag} tags{added}"
    if fmt.kind == "marker":
        return f"text before '{fmt.marker}'"
    return "no visible reasoning"
