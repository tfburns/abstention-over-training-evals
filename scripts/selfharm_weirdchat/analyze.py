"""Statistics and figures for the SelfHarmWeirdChat screening pass and the 5 reruns.

Run with: uv run --with matplotlib --with scienceplots --with scipy python scripts/selfharm_weirdchat/analyze.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from results import joined
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint

plt.style.use(["science", "no-latex"])
BEHAVIORS = {
    "cutting-instructions": "Cutting",
    "direct-kill-yourself": "Kill yourself",
    "suicide-jumping": "Suicide by jumping",
    "extreme-calorie-restriction": "Calorie restriction",
    "purging-advice": "Purging",
}
COLORS = dict(zip(BEHAVIORS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
RNG = np.random.default_rng(0)


def wilson(k: int, n: int) -> tuple[float, float]:
    return proportion_confint(k, n, method="wilson")


def cluster_bootstrap(per_prompt: pd.Series, n_boot: int = 10_000) -> tuple[float, float]:
    """95% CI of the mean failure rate, resampling prompts (each prompt's 5 runs stay together)."""
    v = per_prompt.to_numpy()
    means = v[RNG.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def screen_stats(screen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, g in [*screen.groupby("behavior_id"), ("all", screen)]:
        k, n = int(g.unsafe.sum()), int(g.unsafe.notna().sum())
        rows.append(dict(behavior=b, n_prompts=len(g), n_judged=n, n_unsafe=k, rate=k / n, ci=wilson(k, n)))
    return pd.DataFrame(rows)


def rerun_stats(rerun: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_prompt = (
        rerun.groupby(["id_y", "behavior_id", "weirdchat_rate"]).unsafe.agg(["mean", "sum", "count"]).reset_index()
    )
    rows = []
    for b, g in [*per_prompt.groupby("behavior_id"), ("all", per_prompt)]:
        runs = rerun[rerun.behavior_id.isin(g.behavior_id.unique())].groupby("run").unsafe.mean()
        rows.append(
            dict(
                behavior=b,
                n_prompts=len(g),
                rate=g["mean"].mean(),
                ci=cluster_bootstrap(g["mean"]),
                run_rates=runs.round(4).tolist(),
                run_sd=runs.std(ddof=1),
                frac_fail_ge1=(g["sum"] >= 1).mean(),
                frac_fail_ge3=(g["sum"] >= 3).mean(),
                frac_fail_all5=(g["sum"] == g["count"]).mean(),
                frac_fail_none=(g["sum"] == 0).mean(),
            )
        )
    return pd.DataFrame(rows), per_prompt


def plot_screen(stats: pd.DataFrame, out: Path) -> None:
    s = stats[stats.behavior != "all"].set_index("behavior").loc[list(BEHAVIORS)]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y = np.arange(len(s))
    err = np.array([[r - lo, hi - r] for r, (lo, hi) in zip(s.rate, s.ci)]).T
    ax.barh(y, s.rate, color=[COLORS[b] for b in s.index], height=0.6, xerr=err, error_kw=dict(lw=0.8, capsize=2))
    overall = stats.set_index("behavior").loc["all"]
    ax.axvline(overall.rate, color="0.3", ls="--", lw=0.8, label="All behaviors")
    ax.set_yticks(y, [f"{BEHAVIORS[b]} ($n$={n})" for b, n in zip(s.index, s.n_judged)])
    ax.invert_yaxis()
    ax.set_xlabel("Unsafe response rate")
    ax.set_xlim(0, 1)
    ax.legend(loc="center right", frameon=False)
    save(fig, out, "fig1_screen_unsafe_rate")


def plot_count_hist(per_prompt: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    k = np.arange(0, int(per_prompt["count"].max()) + 1)
    bottom = np.zeros(len(k))
    for b in BEHAVIORS:
        h = (
            per_prompt[per_prompt.behavior_id == b]["sum"]
            .astype(int)
            .value_counts()
            .reindex(k, fill_value=0)
            .to_numpy()
        )
        ax.bar(k, h, bottom=bottom, color=COLORS[b], width=0.7, label=BEHAVIORS[b], edgecolor="white", lw=0.5)
        bottom += h
    ax.set_xticks(k)
    ax.set_xlabel("Unsafe responses out of 5 reruns")
    ax.set_ylabel("Prompts")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    save(fig, out, "fig2_rerun_unsafe_count")


def plot_rerun_rate(stats: pd.DataFrame, out: Path) -> None:
    s = stats.set_index("behavior").loc[[*[b for b in BEHAVIORS if b in stats.behavior.values], "all"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    y = np.arange(len(s))
    for i, (b, r) in enumerate(s.iterrows()):
        color = COLORS.get(b, "0.2")
        ax.errorbar(
            r.rate, i, xerr=[[r.rate - r.ci[0]], [r.ci[1] - r.rate]], fmt="o", color=color, ms=5, capsize=2, lw=1
        )
        ax.scatter(r.run_rates, np.full(len(r.run_rates), i + 0.25), s=8, color=color, alpha=0.6, lw=0)
    ax.set_yticks(y, [f"{BEHAVIORS.get(b, 'All behaviors')} ($n$={int(n)})" for b, n in zip(s.index, s.n_prompts)])
    ax.invert_yaxis()
    ax.set_xlabel("Unsafe response rate")
    ax.set_xlim(0, 1)
    save(fig, out, "fig3_rerun_unsafe_rate")


def plot_weirdchat_vs_olmo(per_prompt: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    jitter = RNG.uniform(-0.015, 0.015, len(per_prompt))
    for b in BEHAVIORS:
        m = (per_prompt.behavior_id == b).to_numpy()
        ax.scatter(
            per_prompt.weirdchat_rate[m],
            per_prompt["mean"][m] + jitter[m],
            s=10,
            color=COLORS[b],
            label=BEHAVIORS[b],
            alpha=0.8,
            lw=0,
        )
    ax.set_xlabel("Behavior rate on original model (WeirdChat)")
    ax.set_ylabel("Unsafe rate on Olmo 3 32B Think")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, out, "fig4_weirdchat_vs_olmo")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/ml/olmo_selfharm"))
    parser.add_argument("--prompts", type=Path, default=Path("data/local/selfharm_weirdchat.jsonl"))
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(exist_ok=True)

    screen = joined(sorted((args.root / "screen").glob("chunk_*.jsonl")), args.root / "screen_out")
    runs = sorted((args.root / "rerun_out").glob("run_*"))
    rerun = pd.concat([joined([args.prompts], r).assign(run=i) for i, r in enumerate(runs)], ignore_index=True)

    s_stats = screen_stats(screen)
    r_stats, per_prompt = rerun_stats(rerun)
    rho, p = spearmanr(per_prompt.weirdchat_rate, per_prompt["mean"])
    summary = dict(
        screen=json.loads(s_stats.to_json(orient="records")),
        rerun=json.loads(r_stats.to_json(orient="records")),
        rerun_judge_abstentions=int(rerun.unsafe.isna().sum()),
        rerun_hit_token_cap=int((rerun.raw_completion_finish_reason == "length").sum()),
        rerun_n_completions=len(rerun),
        spearman_weirdchat_vs_olmo=dict(rho=rho, p=p),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    per_prompt.to_csv(out / "per_prompt_rerun.csv", index=False)
    rerun[
        [
            "id_y",
            "behavior_id",
            "run",
            "unsafe",
            "llm_judge_response",
            "raw_completion_finish_reason",
            "completion",
            "raw_completion_reasoning",
        ]
    ].to_json(out / "rerun_completions.jsonl", orient="records", lines=True)

    plot_screen(s_stats, out)
    plot_count_hist(per_prompt, out)
    plot_rerun_rate(r_stats, out)
    plot_weirdchat_vs_olmo(per_prompt, out)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
