"""How much each copy gained over the untrained base model (Student 0).

A vs B alone can't tell "Unmimic removed nothing" from "neither copy learned anything".
This reads the benchmark files from eval_student.py and reports, against each baseline:

    gain without Unmimic = A - 0        gain through Unmimic = B - 0
    share let through    = (B - 0) / (A - 0)

with 95% intervals from a paired bootstrap over test problems (every model answered the
same problems, so their scores are paired). The share is only printed when A's gain is
clearly above zero; dividing two gains near zero is meaningless.

    python experiments/compare_students.py \\
        --students data/results/students_sampled.json \\
        --baselines data/results/student0_zeroshot.json data/results/student0_fewshot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["students"]


def correct_by_id(result: dict) -> dict[str, bool]:
    return {r["id"]: bool(r["correct"]) for r in result["records"]}


def bootstrap(columns: dict[str, np.ndarray], resamples: int, seed: int) -> dict[str, np.ndarray]:
    """Accuracy of each column over the same resampled problem sets."""
    n = len(next(iter(columns.values())))
    picks = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    return {name: col[picks].mean(axis=1) for name, col in columns.items()}


def ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--students", type=Path, required=True, help="A and B from eval_student")
    parser.add_argument("--baselines", type=Path, nargs="+", required=True, help="Student 0 runs")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/results/student_gains.json"))
    args = parser.parse_args()

    students = load(args.students)
    full = next(s for s in students if "full_reasoning" in s["adapter"])
    guarded = next(s for s in students if "unmimic" in s["adapter"])

    report = []
    for path in args.baselines:
        for base in load(path):
            maps = {"0": correct_by_id(base), "A": correct_by_id(full), "B": correct_by_id(guarded)}
            ids = sorted(set.intersection(*(set(m) for m in maps.values())))
            columns = {k: np.array([m[i] for i in ids], dtype=float) for k, m in maps.items()}
            boot = bootstrap(columns, args.resamples, args.seed)
            acc = {k: float(c.mean()) for k, c in columns.items()}
            gain_a, gain_b = boot["A"] - boot["0"], boot["B"] - boot["0"]
            row = {
                "baseline": f"{base['adapter']} ({path.stem})",
                "n": len(ids),
                "accuracy": acc,
                "gain_without_unmimic": acc["A"] - acc["0"],
                "gain_without_unmimic_ci95": ci(gain_a),
                "gain_through_unmimic": acc["B"] - acc["0"],
                "gain_through_unmimic_ci95": ci(gain_b),
                "share_let_through": None,
                "share_let_through_ci95": None,
            }
            if row["gain_without_unmimic_ci95"][0] > 0:
                share = gain_b / np.where(gain_a > 0, gain_a, np.nan)
                row["share_let_through"] = (acc["B"] - acc["0"]) / (acc["A"] - acc["0"])
                row["share_let_through_ci95"] = ci(share[~np.isnan(share)])
            report.append(row)

            print(f"\nbaseline: {row['baseline']}, {row['n']} problems")
            print(f"  Student 0 {acc['0']:.1%}   A {acc['A']:.1%}   B {acc['B']:.1%}")
            for label, key in (
                ("without Unmimic (A-0)", "without"),
                ("through Unmimic (B-0)", "through"),
            ):
                lo, hi = row[f"gain_{key}_unmimic_ci95"]
                print(f"  gain {label}: {row[f'gain_{key}_unmimic']:+.1%} [{lo:+.1%} to {hi:+.1%}]")
            if row["share_let_through"] is None:
                print("  share let through: not meaningful (A gained nothing measurable)")
            else:
                lo, hi = row["share_let_through_ci95"]
                print(f"  share let through: {row['share_let_through']:.0%} [{lo:.0%}-{hi:.0%}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"rows": report, "config": {k: str(v) for k, v in vars(args).items()}}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
