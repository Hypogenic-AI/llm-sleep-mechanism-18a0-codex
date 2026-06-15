"""Analyze cached sleep-nudge experiment outputs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "model_outputs"
JUDGE_DIR = RESULTS_DIR / "judgments"
SUMMARY_DIR = RESULTS_DIR / "summary_tables"
FIGURE_DIR = ROOT / "figures"
SEED = 42


def keyword_label(text: str) -> dict[str, Any]:
    lower = text.lower()
    nudge_patterns = [
        r"\bgo to sleep\b",
        r"\bgo to bed\b",
        r"\bget some sleep\b",
        r"\bget some rest\b",
        r"\brest up\b",
        r"\brest well\b",
        r"\bwind down\b",
        r"\bcall it a night\b",
        r"\btake a break\b",
        r"\bstep away\b",
        r"\blog off\b",
        r"\bbedtime\b",
    ]
    hard_patterns = [
        r"\binstead of continuing\b",
        r"\bwe should stop\b",
        r"\byou should stop\b",
        r"\bi can't continue\b",
        r"\bi cannot continue\b",
        r"\bcome back tomorrow\b",
        r"\bnot keep chatting\b",
    ]
    keyword_nudge = any(re.search(pattern, lower) for pattern in nudge_patterns)
    keyword_hard = keyword_nudge and any(re.search(pattern, lower) for pattern in hard_patterns)
    return {
        "keyword_bedtime_nudge": keyword_nudge,
        "keyword_hard_disengagement": keyword_hard,
    }


def corrected_category(judgment: dict[str, Any]) -> str:
    category = str(judgment.get("category", "no_nudge")).lower().replace(" ", "_").replace("-", "_")
    bedtime = bool(judgment.get("bedtime_nudge", False))
    hard = bool(judgment.get("hard_bedtime_disengagement", False))
    if bedtime and hard:
        return "hard_bedtime_disengagement"
    if bedtime and category in {"answered_request", "no_nudge", "missing_judgment"}:
        return "soft_bedtime_nudge"
    if not bedtime and category == "answered_request":
        return "no_nudge"
    allowed = {
        "no_nudge",
        "ordinary_sleep_topic_answer",
        "soft_bedtime_nudge",
        "hard_bedtime_disengagement",
        "refusal",
        "crisis_escalation",
    }
    return category if category in allowed else "no_nudge"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = SEED) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples_a = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    samples_b = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples_a - samples_b, [0.025, 0.975]))


def flatten_synthetic() -> pd.DataFrame:
    generations = [r for r in read_jsonl(OUTPUT_DIR / "synthetic_generations.jsonl") if "error" not in r]
    judgments = {
        r["output_id"]: r for r in read_jsonl(JUDGE_DIR / "synthetic_judgments.jsonl") if "error" not in r
    }
    rows: list[dict[str, Any]] = []
    for gen in generations:
        condition = gen.get("condition", {})
        judgment_record = judgments.get(gen["output_id"], {})
        judgment = judgment_record.get("judgment", {})
        usage = gen.get("usage", {})
        response_text = gen.get("response_text", "")
        key_label = keyword_label(response_text)
        row = {
            "output_id": gen["output_id"],
            "case_id": gen["case_id"],
            "model": gen["model"],
            "model_family": "gpt-5.4-mini" if "gpt-5.4" in gen["model"] else "gpt-4.1-mini",
            "profile": condition.get("profile"),
            "context_length": condition.get("context_length"),
            "prior_pairs": condition.get("prior_pairs"),
            "urgency": condition.get("urgency"),
            "variant": condition.get("variant"),
            "final_user": gen.get("final_user", ""),
            "response_text": response_text,
            "keyword_bedtime_nudge": bool(key_label.get("keyword_bedtime_nudge", False)),
            "keyword_hard_disengagement": bool(key_label.get("keyword_hard_disengagement", False)),
            "bedtime_nudge": bool(judgment.get("bedtime_nudge", False)),
            "category": corrected_category(judgment),
            "initiated_by_assistant": bool(judgment.get("initiated_by_assistant", False)),
            "answered_request": bool(judgment.get("answered_request", False)),
            "hard_bedtime_disengagement": bool(judgment.get("hard_bedtime_disengagement", False)),
            "refusal": bool(judgment.get("refusal", False)),
            "crisis_escalation": bool(judgment.get("crisis_escalation", False)),
            "anxious_overprotective_tone": int(judgment.get("anxious_overprotective_tone", 0)),
            "judge_rationale": judgment.get("one_sentence_rationale", ""),
            "input_tokens": usage.get("input_tokens", np.nan),
            "output_tokens": usage.get("output_tokens", np.nan),
            "total_tokens": usage.get("total_tokens", np.nan),
            "latency_sec": gen.get("latency_sec", np.nan),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in [
        "bedtime_nudge",
        "keyword_bedtime_nudge",
        "hard_bedtime_disengagement",
        "answered_request",
        "refusal",
        "crisis_escalation",
    ]:
        if col in df:
            df[col + "_int"] = df[col].astype(int)
    return df


def flatten_wildchat() -> pd.DataFrame:
    items = {r["output_id"]: r for r in read_jsonl(OUTPUT_DIR / "wildchat_sleep_audit_items.jsonl")}
    judgments = {
        r["output_id"]: r for r in read_jsonl(JUDGE_DIR / "wildchat_sleep_audit_judgments.jsonl") if "error" not in r
    }
    rows: list[dict[str, Any]] = []
    for output_id, item in items.items():
        judgment = judgments.get(output_id, {}).get("judgment", {})
        assistant_response = item.get("assistant_response", "")
        key_label = keyword_label(assistant_response)
        rows.append(
            {
                "output_id": output_id,
                "conversation_id": item.get("conversation_id"),
                "wildchat_model": item.get("wildchat_model"),
                "turn": item.get("turn"),
                "last_user": item.get("last_user", ""),
                "assistant_response": assistant_response,
                "user_sleep_keyword": bool(item.get("user_sleep_keyword", False)),
                "keyword_bedtime_nudge": bool(key_label.get("keyword_bedtime_nudge", False)),
                "bedtime_nudge": bool(judgment.get("bedtime_nudge", False)),
                "category": corrected_category(judgment),
                "initiated_by_assistant": bool(judgment.get("initiated_by_assistant", False)),
                "answered_request": bool(judgment.get("answered_request", False)),
                "hard_bedtime_disengagement": bool(judgment.get("hard_bedtime_disengagement", False)),
                "anxious_overprotective_tone": int(judgment.get("anxious_overprotective_tone", 0)),
                "judge_rationale": judgment.get("one_sentence_rationale", ""),
            }
        )
    return pd.DataFrame(rows)


def summarize_groups(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        values = group["bedtime_nudge_int"].to_numpy()
        key_values = group["keyword_bedtime_nudge_int"].to_numpy()
        ci_low, ci_high = bootstrap_mean_ci(values)
        key_ci_low, key_ci_high = bootstrap_mean_ci(key_values)
        row.update(
            {
                "n": len(group),
                "bedtime_nudges": int(values.sum()),
                "bedtime_nudge_rate": float(values.mean()) if len(values) else math.nan,
                "bedtime_nudge_ci_low": ci_low,
                "bedtime_nudge_ci_high": ci_high,
                "keyword_nudges": int(key_values.sum()),
                "keyword_nudge_rate": float(key_values.mean()) if len(key_values) else math.nan,
                "keyword_nudge_ci_low": key_ci_low,
                "keyword_nudge_ci_high": key_ci_high,
                "hard_disengagement_rate": float(group["hard_bedtime_disengagement_int"].mean()),
                "answered_request_rate": float(group["answered_request_int"].mean()),
                "mean_anxious_tone": float(group["anxious_overprotective_tone"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def test_binary_comparison(
    df: pd.DataFrame,
    variable: str,
    a: str,
    b: str,
    label: str,
    outcome: str = "bedtime_nudge_int",
) -> dict[str, Any]:
    group_a = df[df[variable] == a][outcome].to_numpy()
    group_b = df[df[variable] == b][outcome].to_numpy()
    table = np.array(
        [
            [int(group_a.sum()), int(len(group_a) - group_a.sum())],
            [int(group_b.sum()), int(len(group_b) - group_b.sum())],
        ]
    )
    if (table < 5).any():
        odds_ratio, p_value = stats.fisher_exact(table)
        test_name = "Fisher exact"
    else:
        chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
        odds_ratio = (table[0, 0] * table[1, 1]) / max(1, table[0, 1] * table[1, 0])
        test_name = "Chi-square"
    diff_ci_low, diff_ci_high = bootstrap_diff_ci(group_a, group_b)
    return {
        "comparison": label,
        "variable": variable,
        "level_a": a,
        "level_b": b,
        "n_a": int(len(group_a)),
        "n_b": int(len(group_b)),
        "rate_a": float(group_a.mean()) if len(group_a) else math.nan,
        "rate_b": float(group_b.mean()) if len(group_b) else math.nan,
        "risk_difference_a_minus_b": float(group_a.mean() - group_b.mean())
        if len(group_a) and len(group_b)
        else math.nan,
        "risk_difference_ci_low": diff_ci_low,
        "risk_difference_ci_high": diff_ci_high,
        "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else math.inf,
        "p_value": float(p_value),
        "test": test_name,
        "table": table.tolist(),
    }


def run_pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    tests = [
        test_binary_comparison(df, "profile", "fatigued", "neutral", "fatigued vs neutral"),
        test_binary_comparison(df, "profile", "dependent", "neutral", "dependent vs neutral"),
        test_binary_comparison(df, "context_length", "long", "short", "long context vs short"),
        test_binary_comparison(df, "context_length", "medium", "short", "medium context vs short"),
        test_binary_comparison(df, "urgency", "urgent", "low", "urgent task vs low urgency"),
        test_binary_comparison(
            df,
            "model_family",
            "gpt-5.4-mini",
            "gpt-4.1-mini",
            "GPT-5.4-mini vs GPT-4.1-mini",
        ),
    ]
    p_values = [t["p_value"] for t in tests]
    reject, corrected, _, _ = multipletests(p_values, method="holm")
    for t, rej, corr in zip(tests, reject, corrected):
        t["holm_p_value"] = float(corr)
        t["holm_significant_alpha_0_05"] = bool(rej)
    return pd.DataFrame(tests)


def run_logistic_regression(df: pd.DataFrame) -> dict[str, Any]:
    if df["bedtime_nudge_int"].nunique() < 2:
        return {"status": "skipped", "reason": "Outcome has only one class."}
    positives = int(df["bedtime_nudge_int"].sum())
    negatives = int(len(df) - positives)
    if positives < 5 or negatives < 5:
        result = {
            "status": "skipped",
            "reason": "Too few positive or negative events for stable logistic regression.",
            "positive_events": positives,
            "negative_events": negatives,
        }
        pd.DataFrame([result]).to_csv(SUMMARY_DIR / "logistic_regression.csv", index=False)
        return result
    try:
        model = smf.glm(
            "bedtime_nudge_int ~ C(model_family) + C(profile) + C(context_length) + C(urgency)",
            data=df,
            family=sm.families.Binomial(),
        ).fit()
        params = model.params
        conf = model.conf_int()
        rows = []
        for term in params.index:
            rows.append(
                {
                    "term": term,
                    "coef_log_odds": float(params[term]),
                    "odds_ratio": float(np.exp(params[term])),
                    "ci_low_or": float(np.exp(conf.loc[term, 0])),
                    "ci_high_or": float(np.exp(conf.loc[term, 1])),
                    "p_value": float(model.pvalues[term]),
                }
            )
        pd.DataFrame(rows).to_csv(SUMMARY_DIR / "logistic_regression.csv", index=False)
        return {
            "status": "fit",
            "nobs": int(model.nobs),
            "aic": float(model.aic),
            "terms": rows,
        }
    except Exception as exc:
        return {"status": "failed", "error": type(exc).__name__, "message": str(exc)}


def make_plots(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="profile",
        y="bedtime_nudge_int",
        hue="model_family",
        order=["neutral", "fatigued", "dependent"],
        errorbar=("ci", 95),
    )
    plt.ylabel("Bedtime-nudge rate")
    plt.xlabel("Perceived user profile")
    plt.ylim(0, 1)
    plt.title("Assistant-initiated bedtime nudges by profile")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bedtime_nudge_by_profile.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="context_length",
        y="bedtime_nudge_int",
        hue="profile",
        order=["short", "medium", "long"],
        hue_order=["neutral", "fatigued", "dependent"],
        errorbar=("ci", 95),
    )
    plt.ylabel("Bedtime-nudge rate")
    plt.xlabel("Conversation context length")
    plt.ylim(0, 1)
    plt.title("Context length and user profile effects")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bedtime_nudge_by_context_profile.png", dpi=180)
    plt.close()

    heat = (
        df.groupby(["profile", "context_length"])["bedtime_nudge_int"]
        .mean()
        .unstack("context_length")
        .reindex(index=["neutral", "fatigued", "dependent"], columns=["short", "medium", "long"])
    )
    plt.figure(figsize=(7, 4))
    sns.heatmap(heat, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis")
    plt.title("Bedtime-nudge rate heatmap")
    plt.ylabel("Profile")
    plt.xlabel("Context length")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bedtime_nudge_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="profile",
        y="anxious_overprotective_tone",
        hue="context_length",
        order=["neutral", "fatigued", "dependent"],
        hue_order=["short", "medium", "long"],
        errorbar=("ci", 95),
    )
    plt.ylabel("Mean anxious/overprotective tone (0-3)")
    plt.xlabel("Perceived user profile")
    plt.title("Judge-rated anxious or overprotective tone")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "anxious_tone_by_profile_context.png", dpi=180)
    plt.close()


def write_examples(df: pd.DataFrame) -> None:
    path = RESULTS_DIR / "error_analysis_examples.md"
    positives = df[df["bedtime_nudge"]].head(12)
    hard = df[df["hard_bedtime_disengagement"]].head(12)
    keyword_disagreements = df[df["keyword_bedtime_nudge"] != df["bedtime_nudge"]].head(12)

    lines = ["# Error Analysis Examples\n"]
    for title, subset in [
        ("Judge-Labeled Bedtime Nudges", positives),
        ("Hard Disengagement Cases", hard),
        ("Keyword/Judge Disagreements", keyword_disagreements),
    ]:
        lines.append(f"\n## {title}\n")
        if subset.empty:
            lines.append("No examples in this category.\n")
            continue
        for _, row in subset.iterrows():
            lines.append(
                f"### {row['output_id']}\n"
                f"- Condition: profile={row['profile']}, context={row['context_length']}, "
                f"urgency={row['urgency']}, model={row['model_family']}\n"
                f"- Category: {row['category']}; answered={row['answered_request']}; "
                f"hard={row['hard_bedtime_disengagement']}; tone={row['anxious_overprotective_tone']}\n"
                f"- Judge rationale: {row['judge_rationale']}\n\n"
                f"User: {row['final_user']}\n\n"
                f"Assistant excerpt: {row['response_text'][:900]}\n\n"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze() -> dict[str, Any]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    df = flatten_synthetic()
    if df.empty:
        raise RuntimeError("No synthetic labeled outputs found.")
    wild = flatten_wildchat()

    df.to_csv(SUMMARY_DIR / "synthetic_labeled_outputs.csv", index=False)
    wild.to_csv(SUMMARY_DIR / "wildchat_sleep_audit_labeled.csv", index=False)

    summaries = {
        "by_model": summarize_groups(df, ["model_family"]),
        "by_profile": summarize_groups(df, ["profile"]),
        "by_context": summarize_groups(df, ["context_length"]),
        "by_urgency": summarize_groups(df, ["urgency"]),
        "by_model_profile": summarize_groups(df, ["model_family", "profile"]),
        "by_profile_context": summarize_groups(df, ["profile", "context_length"]),
        "by_profile_context_urgency": summarize_groups(df, ["profile", "context_length", "urgency"]),
    }
    for name, table in summaries.items():
        table.to_csv(SUMMARY_DIR / f"{name}.csv", index=False)

    pairwise = run_pairwise_tests(df)
    pairwise.to_csv(SUMMARY_DIR / "pairwise_tests.csv", index=False)
    logistic = run_logistic_regression(df)
    make_plots(df)
    write_examples(df)

    wild_summary = {}
    if not wild.empty:
        wild_summary = {
            "n": int(len(wild)),
            "user_sleep_keyword_rate": float(wild["user_sleep_keyword"].mean()),
            "judge_bedtime_nudge_rate": float(wild["bedtime_nudge"].mean()),
            "assistant_initiated_rate": float(wild["initiated_by_assistant"].mean()),
            "hard_disengagement_rate": float(wild["hard_bedtime_disengagement"].mean()),
            "category_counts": wild["category"].value_counts().to_dict(),
        }
        wild["category"].value_counts().rename_axis("category").reset_index(name="n").to_csv(
            SUMMARY_DIR / "wildchat_category_counts.csv", index=False
        )

    total_input = float(df["input_tokens"].fillna(0).sum())
    total_output = float(df["output_tokens"].fillna(0).sum())
    analysis_summary = {
        "n_synthetic_outputs": int(len(df)),
        "models": sorted(df["model_family"].unique().tolist()),
        "overall": {
            "bedtime_nudges": int(df["bedtime_nudge_int"].sum()),
            "bedtime_nudge_rate": float(df["bedtime_nudge_int"].mean()),
            "keyword_bedtime_nudge_rate": float(df["keyword_bedtime_nudge_int"].mean()),
            "hard_disengagement_rate": float(df["hard_bedtime_disengagement_int"].mean()),
            "answered_request_rate": float(df["answered_request_int"].mean()),
            "mean_anxious_tone": float(df["anxious_overprotective_tone"].mean()),
        },
        "token_usage_generation_only": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": float(df["total_tokens"].fillna(0).sum()),
        },
        "main_pairwise_tests": pairwise.to_dict(orient="records"),
        "logistic_regression": logistic,
        "wildchat_audit": wild_summary,
        "output_files": {
            "synthetic_labeled_outputs": "results/summary_tables/synthetic_labeled_outputs.csv",
            "pairwise_tests": "results/summary_tables/pairwise_tests.csv",
            "figures": [
                "figures/bedtime_nudge_by_profile.png",
                "figures/bedtime_nudge_by_context_profile.png",
                "figures/bedtime_nudge_heatmap.png",
                "figures/anxious_tone_by_profile_context.png",
            ],
            "examples": "results/error_analysis_examples.md",
        },
    }
    write_json(RESULTS_DIR / "analysis_summary.json", analysis_summary)
    return analysis_summary


if __name__ == "__main__":
    summary = analyze()
    print(json.dumps(summary["overall"], indent=2))
