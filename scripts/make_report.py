"""Write report.md, the comment CML posts on each pull request.

Run from the repository root: python -m scripts.make_report
"""

from __future__ import annotations

from src.config import METRICS_JSON, PLOT_PNG, PROMOTION_JSON, ROOT, load_params, read_json


def main() -> None:
    params = load_params()
    metrics = read_json(METRICS_JSON)
    promotion = read_json(PROMOTION_JSON)
    gain = 100 * (1 - metrics["rmse"] / metrics["naive_rmse"])

    lines = [
        "## Model report",
        "",
        f"Model: `{params['train']['model']}` on {metrics['n_test']} test rows",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| RMSE | {metrics['rmse']} |",
        f"| MAE | {metrics['mae']} |",
        f"| WAPE | {metrics['wape']:.1%} |",
        f"| Naive baseline RMSE | {metrics['naive_rmse']} |",
        f"| Gain over baseline | {gain:.1f}% |",
        f"| Promotion threshold (RMSE) | {params['promote']['max_rmse']} |",
        "",
        f"Promotion in this CI run: **{'yes' if promotion['promoted'] else 'no'}**, "
        f"{promotion['reason']}.",
        "",
        f"![predicted vs actual](./{PLOT_PNG.relative_to(ROOT).as_posix()})",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
