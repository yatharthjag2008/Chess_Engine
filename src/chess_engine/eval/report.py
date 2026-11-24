"""Render evaluation results (the ``results.json`` dictionary) as a Markdown report.

The report is plain Markdown with relative links to the charts saved next to it,
so it reads well on GitHub and in an editor.
"""

from __future__ import annotations

from typing import Any

CHARTS = {
    "accuracy": "accuracy_by_rating.png",
    "calibration": "calibration.png",
    "cross_rating": "cross_rating.png",
    "breakdowns": "accuracy_by_phase_and_clock.png",
}


def _pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def _ci(ci: list[float] | tuple[float, float] | None) -> str:
    if not ci or ci[0] is None or ci[0] != ci[0]:  # missing or NaN
        return ""
    return f" ({ci[0] * 100:.1f}-{ci[1] * 100:.1f})"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header)))
             + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _baseline_value(results: dict[str, Any], name: str, label: str | None) -> str:
    baseline = results.get("baselines", {}).get(name)
    if not baseline or baseline.get("skipped"):
        return "-"
    stats = baseline["overall"] if label is None else baseline["buckets"].get(label, {})
    return _pct(stats.get("top1"))


def render_report(results: dict[str, Any]) -> str:
    """The full Markdown report."""
    labels = results["labels"]
    matching = results["move_matching"]
    overall = matching["overall"]
    setup = results["setup"]
    baselines = results.get("baselines", {})
    out: list[str] = [f"# Evaluation: {setup['model_name']}", ""]

    out += ["## Summary", ""]
    out.append(f"- **Top-1 move match: {_pct(overall['top1'])}**{_ci(matching.get('ci', {}).get('overall'))} "
               f"of {overall['positions']:,} held-out positions ({setup['split']} split, "
               f"{setup['games']:,} games); top-3 {_pct(overall['top3'])}, top-5 {_pct(overall['top5'])}.")
    out.append(f"- Mean log-likelihood of the human move {overall['log_likelihood']:.3f} "
               f"(perplexity {overall['perplexity']:.2f}, out of {overall['mean_legal_moves']:.1f} legal moves on average).")
    for key in ("most_common_move", "most_common_move_same_rating", "stockfish", "random"):
        baseline = baselines.get(key)
        if not baseline:
            continue
        if baseline.get("skipped"):
            out.append(f"- {baseline['name']}: skipped ({baseline['skipped']}).")
            continue
        s = baseline["overall"]
        text = f"- {baseline['name']}: {_pct(s['top1'])}"
        if key.startswith("most_common") and "coverage" in s:
            text += (f" (it knows {_pct(s['coverage'])} of the positions; on those it scores "
                     f"{_pct(s.get('top1_covered'))} vs. the model's {_pct(s.get('model_top1_covered'))})")
        if key == "stockfish":
            text += (f" on {s['positions']:,} sampled positions (the model scores "
                     f"{_pct(s.get('model_top1_covered'))} on the same ones)")
        out.append(text + ".")
    diff = results.get("model_vs_most_common")
    if diff:
        out.append(f"- Model minus most-common-move baseline: {diff['difference'] * 100:+.1f} points"
                   f"{_ci(diff.get('ci'))}.")
    cross = results.get("cross_rating")
    if cross:
        out.append(f"- Telling the model each player's real rating scores {_pct(cross['matched_top1'])}; "
                   f"the best single rating for everyone ({cross['fixed_rating']['bucket']}) scores "
                   f"{_pct(cross['fixed_rating']['top1'])} on the same {sum(cross['positions']):,} positions.")
    out.append("")

    out += ["## Accuracy by rating", "", f"![Top-1 accuracy by rating]({CHARTS['accuracy']})", ""]
    header = ["rating", "positions", "model top-1", "top-3", "top-5", "log-lik", "most common", "same-rating common",
              "Stockfish", "random"]
    rows = []
    cis = matching.get("ci", {}).get("buckets", [None] * len(labels))
    for label, ci in zip(labels, cis):
        s = matching["buckets"][label]
        if not s.get("positions"):
            continue
        rows.append([label, f"{s['positions']:,}", _pct(s["top1"]) + _ci(ci), _pct(s["top3"]), _pct(s["top5"]),
                     f"{s['log_likelihood']:.3f}", _baseline_value(results, "most_common_move", label),
                     _baseline_value(results, "most_common_move_same_rating", label),
                     _baseline_value(results, "stockfish", label), _baseline_value(results, "random", label)])
    rows.append(["**all**", f"{overall['positions']:,}", f"**{_pct(overall['top1'])}**", _pct(overall["top3"]),
                 _pct(overall["top5"]), f"{overall['log_likelihood']:.3f}",
                 _baseline_value(results, "most_common_move", None),
                 _baseline_value(results, "most_common_move_same_rating", None),
                 _baseline_value(results, "stockfish", None), _baseline_value(results, "random", None)])
    out += [_table(header, rows), "",
            f"Brackets: {int(results['setup']['confidence'] * 100)}% confidence intervals from "
            f"{results['setup']['bootstrap_samples']:,} bootstrap resamples of whole games. Stockfish is scored "
            f"on a random sample of positions; every other column on all of them.", ""]

    if cross:
        out += ["## Does the model play like each rating?", "", f"![Cross-rating accuracy]({CHARTS['cross_rating']})",
                "", "Each column is one group of players; each row tells the model a different rating. "
                "Outlined: the rating that predicts that group best.", ""]
        rows = [[label, _pct(cross["diagonal_top1"][p]), str(cross["best_model_bucket"][p] or "-")]
                for p, label in enumerate(labels) if cross["positions"][p]]
        out += [_table(["players rated", "model told their rating", "best rating to tell the model"], rows), ""]

    breakdowns = results.get("breakdowns")
    if breakdowns:
        out += ["## When is the model right?", "", f"![Accuracy by phase and clock]({CHARTS['breakdowns']})", ""]
        for key, title in (("phase", "Game phase"), ("clock", "Mover's clock"), ("legal_moves", "Legal moves")):
            rows = [[band, f"{s['positions']:,}", _pct(s.get("top1")), _pct(s.get("top5")),
                     _pct(s.get("random_top1"))] for band, s in breakdowns[key].items() if s.get("positions")]
            out += [f"**{title}**", "", _table([title.lower(), "positions", "top-1", "top-5", "random"], rows), ""]

    calibration = results.get("calibration")
    if calibration:
        out += ["## Calibration", "", f"![Calibration]({CHARTS['calibration']})", "",
                f"- Every legal move: expected calibration error {calibration['all_moves']['ece']:.4f} over "
                f"{calibration['all_moves']['predictions']:,} predictions.",
                f"- The model's top move: expected calibration error {calibration['top1']['ece']:.4f}."]
        value = results.get("value_head")
        if value:
            shares = value["result_shares"]
            out.append(f"- Value head (game result from this position): {_pct(value['accuracy'])} of results "
                       f"guessed right (wins {_pct(shares['win'], 0)}, draws {_pct(shares['draw'], 0)}, "
                       f"losses {_pct(shares['loss'], 0)}), log loss {value['log_loss']:.3f}, Brier "
                       f"{value['brier']:.3f}, score calibration error {value['score']['ece']:.4f}.")
        out.append("")

    out += ["## Setup", "", _table(["", ""], [
        ["model file", f"`{setup['model_path']}`"],
        ["trained steps", str(setup.get("model_step", "-"))],
        ["dataset", f"`{setup['dataset']}`"],
        ["split", f"{setup['split']} ({setup['positions']:,} positions, {setup['games']:,} games)"],
        ["Stockfish", setup.get("stockfish", "-")],
        ["evaluation time", f"{setup['seconds']:.0f} s"],
    ]), ""]
    return "\n".join(out)
