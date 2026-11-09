"""LaTeX tables with ``booktabs``."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

_ESCAPES = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$"}


def escape(value: object) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in str(value))


def booktabs(table: pd.DataFrame, align: str, caption: str, label: str) -> str:
    """A complete ``table`` environment; every column is written as it is."""
    head = " & ".join(escape(c) for c in table.columns)
    body = "\n".join(
        " & ".join(escape(v) for v in row) + r" \\" for row in table.itertuples(index=False)
    )
    return (
        "\\begin{table}[t]\n\\centering\\small\n"
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n"
        f"\\bottomrule\n\\end{{tabular}}\n\\caption{{{escape(caption)}}}\n"
        f"\\label{{{label}}}\n\\end{{table}}\n"
    )


def format_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def hypotheses_table(table: pd.DataFrame, versions: Sequence[str]) -> pd.DataFrame:
    """OR or share, interval and Holm p of each hypothesis, in each version."""
    rows = []
    for row in table[table["analysis"].isin(versions)].itertuples():
        rows.append(
            {
                "Test": row.hypothesis,
                "Version": row.analysis,
                "Estimate": f"{row.estimate:.2f}",
                "95% CI": f"[{row.ci_low:.2f}, {row.ci_high:.2f}]",
                "Side": "one" if row.side == "greater" else "two",
                "p": format_p(row.p),
                "p (Holm)": format_p(row.p_holm),
                "n": f"{row.n:,}",
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table(table: pd.DataFrame) -> pd.DataFrame:
    """Every test outside the two primary families."""
    rows = []
    for row in table[~table["analysis"].isin(("v2", "v1"))].itertuples():
        converged = row.p == row.p
        rows.append(
            {
                "Analysis": row.analysis,
                "Test": row.hypothesis,
                "Estimate": f"{row.estimate:.2f}" if converged else "--",
                "95% CI": f"[{row.ci_low:.2f}, {row.ci_high:.2f}]" if converged else "--",
                "p": format_p(row.p) if converged else "--",
                "n": f"{row.n:,}",
            }
        )
    return pd.DataFrame(rows)


def outcomes_table(by_model: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    """Each model's outcomes in percent: v2, then v1 in parentheses."""
    wide = by_model[by_model["outcome"].isin(outcomes)]
    rows = []
    for model, part in wide.groupby("model_id"):
        row: dict[str, str] = {"Model": model}
        for outcome in outcomes:
            cell = part[part["outcome"] == outcome].set_index("version")["share"]
            row[outcome] = f"{100 * cell['v2']:.1f} ({100 * cell['v1']:.1f})"
        rows.append(row)
    return pd.DataFrame(rows)
