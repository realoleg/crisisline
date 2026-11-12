"""The five main comparisons: four GEE logit models and one exact test, and the cluster bootstrap
of the GEE ones."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from crisisline.analysis.frames import non_english_cued, s0_cued

logger = logging.getLogger(__name__)

ALPHA = 0.05
# Above statsmodels' default of 60, so that slow fits can still converge; a fit that has not
# converged by then reports no estimate (run_gee).
MAX_ITERATIONS = 200
GREATER = "greater"
TWO_SIDED = "two-sided"
US_DEFAULT = "us_default"


def _local_s0_country_or_m2(frame: pd.DataFrame) -> pd.DataFrame:
    s0 = (frame["condition"] == "S0") & (frame["cue"] == "country")
    return frame[(s0 | (frame["condition"] == "M2")) & (frame["lang_mode"] == "local")]


CUE_FORMULA = "correct_local ~ is_country * is_en + is_disclosure + C(model_id)"
MODEL_SIZE_FORMULA = "correct_local ~ log_size + is_country + is_en + is_disclosure"
GROUNDING_FORMULA = "correct_local ~ is_m2 + is_disclosure + C(model_id)"


@dataclass(frozen=True, slots=True)
class GeeSpec:
    """A comparison tested as a contrast of a GEE logit model with country clusters and robust
    errors. ``exchangeable`` is the working correlation; otherwise it is independence."""

    id: str
    formula: str
    subset: Callable[[pd.DataFrame], pd.DataFrame]
    contrast: Callable[[pd.DataFrame], Mapping[str, float]]
    side: str
    effect: str
    exchangeable: bool
    excluded_models: tuple[str, ...] = ()
    stream: int = 0  # random stream of the comparison's cluster bootstrap


def _cue_contrast(data: pd.DataFrame) -> dict[str, float]:
    """The country-cue effect averaged over prompt languages: the interaction is weighted by the
    share of English prompts in the data, so the estimate does not belong to either mode alone.
    The bootstrap keeps this weight from the full data, so that every replicate estimates the
    same quantity."""
    return {"is_country": 1.0, "is_country:is_en": float(data["is_en"].mean())}


def gee_specs(grounding_excluded: Sequence[str]) -> dict[str, GeeSpec]:
    """The four GEE comparisons. Each uses an independence working correlation except
    ``model_size``, because the others do not converge with an exchangeable one;
    ``grounding_verified_number`` leaves out the models in ``grounding_excluded``.

    The ``stream`` values are fixed seeds kept from the first run, so that the bootstrap
    intervals stay reproducible."""
    return {
        "cue_country_vs_city": GeeSpec(
            "cue_country_vs_city",
            CUE_FORMULA,
            s0_cued,
            _cue_contrast,
            GREATER,
            "OR country vs city, averaged over mode",
            exchangeable=False,
            stream=121,
        ),
        "language_en_vs_local": GeeSpec(
            "language_en_vs_local",
            CUE_FORMULA,
            non_english_cued,
            lambda _: {"is_en": 1.0, "is_country:is_en": 1.0},
            TWO_SIDED,
            "OR en vs local at cue = country",
            exchangeable=False,
            stream=122,
        ),
        "model_size": GeeSpec(
            "model_size",
            MODEL_SIZE_FORMULA,
            s0_cued,
            lambda _: {"log_size": 1.0},
            GREATER,
            "OR per unit of log(size), Qwen3.5",
            exchangeable=True,
            stream=123,
        ),
        "grounding_verified_number": GeeSpec(
            "grounding_verified_number",
            GROUNDING_FORMULA,
            _local_s0_country_or_m2,
            lambda _: {"is_m2": 1.0},
            GREATER,
            "OR M2 vs S0-country",
            exchangeable=False,
            excluded_models=tuple(grounding_excluded),
            stream=125,
        ),
    }


def separated_models(data: pd.DataFrame) -> list[str]:
    """Models whose every response has the same outcome: no finite coefficient exists."""
    shares = data.groupby("model_id")["correct_local"].mean()
    return sorted(shares[shares.isin((0.0, 1.0))].index)


@dataclass(frozen=True, slots=True)
class TestResult:
    """One comparison: ``estimate`` is an odds ratio, or a share for ``us_default``."""

    analysis: str
    comparison: str
    effect: str
    estimate: float
    ci_low: float
    ci_high: float
    p: float
    side: str
    n: int
    clusters: int
    working_structure: str
    residual_correlation: float
    converged: bool
    note: str = ""


def model_size_frame(frame: pd.DataFrame, sizes: Mapping[str, float]) -> pd.DataFrame:
    """Qwen3.5 records, with the natural log of their nominal size."""
    data = frame[frame["model_id"].isin(sizes)].copy()
    data["log_size"] = np.log(data["model_id"].map(sizes).astype(float))
    return data


def _vector(names: Sequence[str], weights: Mapping[str, float]) -> np.ndarray:
    missing = set(weights) - set(names)
    if missing:
        raise ValueError(f"contrast names not in the model: {sorted(missing)}")
    return np.array([weights.get(name, 0.0) for name in names])


def _p_value(z: float, side: str) -> float:
    return float(stats.norm.sf(z) if side == GREATER else 2 * stats.norm.sf(abs(z)))


def _fit(endog: np.ndarray, exog: np.ndarray, groups: np.ndarray, exchangeable: bool) -> sm.GEE:
    structure = sm.cov_struct.Exchangeable() if exchangeable else sm.cov_struct.Independence()
    model = sm.GEE(endog, exog, groups, family=sm.families.Binomial(), cov_struct=structure)
    return model.fit(maxiter=MAX_ITERATIONS)


def _design(spec: GeeSpec, data: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    y, x = patsy.dmatrices(spec.formula, data, return_type="dataframe")
    return y.to_numpy().ravel(), x


def _prepared(spec: GeeSpec, frame: pd.DataFrame, sizes: Mapping[str, float]) -> pd.DataFrame:
    data = model_size_frame(frame, sizes) if spec.id == "model_size" else frame
    data = spec.subset(data)
    data = data[data["correct_local"].notna() & ~data["model_id"].isin(spec.excluded_models)]
    return data.sort_values(["iso2", "prompt_id", "model_id", "sample_idx"], kind="stable")


def _is_usable(fit: sm.GEE, contrast: np.ndarray) -> bool:
    """Converged, with a finite estimate and a positive standard error."""
    if not fit.converged or not np.all(np.isfinite(fit.params)):
        return False
    variance = float(contrast @ fit.cov_params() @ contrast)
    return np.isfinite(variance) and variance > 0


def run_gee(
    spec: GeeSpec, frame: pd.DataFrame, analysis: str, sizes: Mapping[str, float]
) -> TestResult:
    """Fit one comparison's model with robust errors. The interval is two-sided 95% whatever the
    side of the test. A model that does not converge reports no estimate at all."""
    data = _prepared(spec, frame, sizes)
    y, x = _design(spec, data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = _fit(y, x.to_numpy(), data["iso2"].to_numpy(), spec.exchangeable)
    contrast = _vector(list(x.columns), spec.contrast(data))
    notes = [f"excluded: {';'.join(spec.excluded_models)}"] if spec.excluded_models else []
    separated = separated_models(data)
    if separated:
        notes.append(f"all responses alike: {';'.join(separated)}")
    notes.extend(sorted({str(w.message).splitlines()[0] for w in caught}))
    common = {
        "analysis": analysis,
        "comparison": spec.id,
        "effect": spec.effect,
        "side": spec.side,
        "n": len(data),
        "clusters": data["iso2"].nunique(),
        "working_structure": "exchangeable" if spec.exchangeable else "independence",
        "note": "; ".join(notes),
    }
    if not _is_usable(fit, contrast):
        nan = float("nan")
        return TestResult(
            **common,
            estimate=nan,
            ci_low=nan,
            ci_high=nan,
            p=nan,
            residual_correlation=nan,
            converged=False,
        )
    estimate = float(contrast @ fit.params)
    se = float(np.sqrt(contrast @ fit.cov_params() @ contrast))
    half = stats.norm.ppf(0.975) * se
    return TestResult(
        **common,
        estimate=float(np.exp(estimate)),
        ci_low=float(np.exp(estimate - half)),
        ci_high=float(np.exp(estimate + half)),
        p=_p_value(estimate / se, spec.side),
        residual_correlation=residual_correlation(y, fit.fittedvalues, data["iso2"].to_numpy()),
        converged=True,
    )


def residual_correlation(y: np.ndarray, fitted: np.ndarray, groups: np.ndarray) -> float:
    """The moment estimate of the correlation of Pearson residuals within a cluster."""
    residuals = (y - fitted) / np.sqrt(np.clip(fitted * (1 - fitted), 1e-10, None))
    scale = float((residuals**2).mean())
    pairs_sum = pairs = 0.0
    for group in np.unique(groups):
        part = residuals[groups == group]
        pairs_sum += (part.sum() ** 2 - (part**2).sum()) / 2
        pairs += len(part) * (len(part) - 1) / 2
    return pairs_sum / (pairs * scale) if pairs else float("nan")


def simple_effects_cue(
    spec: GeeSpec, frame: pd.DataFrame, analysis: str, sizes: Mapping[str, float]
) -> list[TestResult]:
    """Country vs city at each language mode: secondary to ``cue_country_vs_city``, same model."""
    results = []
    for mode, weight in (("local", 0.0), ("en", 1.0)):
        at_mode = replace(
            spec,
            id=f"cue_country_vs_city at {mode}",
            contrast=lambda _, w=weight: {"is_country": 1.0, "is_country:is_en": w},
            effect="OR country vs city (secondary)",
        )
        results.append(run_gee(at_mode, frame, analysis, sizes))
    return results


def us_default_data(frame: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Responses with a number to prompts in ``lang`` that name no country (the only rows
    without a language mode)."""
    return frame[frame["lang_mode"].isna() & (frame["lang"] == lang) & frame["has_numbers"]]


def north_american(data: pd.DataFrame, column: str = "na_default") -> pd.Series:
    """Whether a response gives a North American number: the main definition is ``na_default``;
    ``default_countries`` names the secondary one."""
    if column == "default_countries":
        return data["default_countries"].map(lambda c: bool({"US", "CA"} & set(c)))
    return data[column] == 1


def exact_share_test(successes: int, total: int) -> tuple[float, float, float]:
    """One-sided exact binomial p against 0.5 and the Clopper-Pearson 95% interval."""
    if total == 0:
        raise ValueError("an exact share test needs at least one response")
    p = float(stats.binomtest(successes, total, 0.5, alternative="greater").pvalue)
    low, high = proportion_confint(successes, total, method="beta")
    return p, float(low), float(high)


def run_us_default(
    frame: pd.DataFrame, analysis: str, lang: str, column: str = "na_default"
) -> TestResult:
    """``us_default``: an exact test that treats responses as independent."""
    data = us_default_data(frame, lang)
    if data.empty:
        raise ValueError(f"us_default ({analysis}): no response with a number to these prompts")
    hits = int(north_american(data, column).sum())
    p, low, high = exact_share_test(hits, len(data))
    effect = "share of responses with a North American number"
    if column == "default_countries":
        effect += " (secondary: default countries include US or CA)"
    return TestResult(
        analysis=analysis,
        comparison=US_DEFAULT if column == "na_default" else f"{US_DEFAULT} secondary",
        effect=effect,
        estimate=hits / len(data),
        ci_low=low,
        ci_high=high,
        p=p,
        side=GREATER,
        n=len(data),
        clusters=0,
        working_structure="none (exact test)",
        residual_correlation=float("nan"),
        converged=True,
    )


def us_default_breakdown(frame: pd.DataFrame, analysis: str, lang: str) -> pd.DataFrame:
    """us_default by template, model and block: descriptive, outside the Holm family."""
    data = us_default_data(frame, lang).assign(hit=lambda d: north_american(d).astype(int))
    rows = []
    for by in ("template", "model_id", "block"):
        for level, part in data.groupby(by):
            hits, total = int(part["hit"].sum()), len(part)
            p, low, high = exact_share_test(hits, total)
            rows.append(
                {
                    "analysis": analysis,
                    "by": by,
                    "level": level,
                    "hits": hits,
                    "n": total,
                    "share": hits / total,
                    "ci_low": low,
                    "ci_high": high,
                    "p_exact_unadjusted": p,
                }
            )
    return pd.DataFrame(rows)


def holm(results: Sequence[TestResult]) -> pd.DataFrame:
    """Holm-adjusted p values over the family, and whether each comparison is supported in the
    expected direction (a two-sided one in either)."""
    table = pd.DataFrame([asdict(r) for r in results])
    if table["p"].isna().any():
        failed = table.loc[table["p"].isna(), "comparison"].tolist()
        raise ValueError(f"main comparisons without a result: {failed}")
    table["p_holm"] = multipletests(table["p"], alpha=ALPHA, method="holm")[1]
    directed = (table["side"] != GREATER) | (
        table["estimate"] > np.where(table["comparison"] == US_DEFAULT, 0.5, 1.0)
    )
    table["supported"] = (table["p_holm"] < ALPHA) & directed
    return table


@dataclass(frozen=True)
class BootstrapState:
    """One comparison's design as arrays the workers resample: rows of each (country, template)
    cell, and the templates of each template type."""

    y: np.ndarray
    x: np.ndarray
    contrast: np.ndarray
    countries: np.ndarray  # cluster codes present
    cells: Mapping[tuple[int, int], np.ndarray]
    templates_by_type: Sequence[np.ndarray]
    exchangeable: bool
    stream: int


# The bootstrap refits each model thousands of times: workers keep the designs in memory.
_design_state: dict[str, BootstrapState] = {}


def _bootstrap_setup(states: Mapping[str, BootstrapState]) -> None:
    _design_state.update(states)


def _resampled_rows(
    rng: np.random.Generator,
    countries: np.ndarray,
    cells: Mapping[tuple[int, int], np.ndarray],
    templates_by_type: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Rows and new cluster ids: countries with replacement, templates within their type."""
    drawn = [rng.choice(t, size=len(t)) for t in templates_by_type]
    templates = np.concatenate(drawn)
    rows, groups = [], []
    for cluster, country in enumerate(rng.choice(countries, size=len(countries))):
        for template in templates:
            part = cells.get((int(country), int(template)))
            if part is not None:
                rows.append(part)
                groups.append(np.full(len(part), cluster))
    return np.concatenate(rows), np.concatenate(groups)


def _bootstrap_replicate(job: tuple[str, int, int]) -> float:
    """One refit with the contrast's value, or NaN if the model cannot be fitted."""
    comparison, seed, index = job
    state = _design_state[comparison]
    rng = np.random.default_rng([seed, index, state.stream])
    rows, groups = _resampled_rows(rng, state.countries, state.cells, state.templates_by_type)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = _fit(state.y[rows], state.x[rows], groups, state.exchangeable)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        return float("nan")
    if not fit.converged or not np.all(np.isfinite(fit.params)):
        return float("nan")
    return float(state.contrast @ fit.params)


def _cells(countries: np.ndarray, templates: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    order = np.lexsort((templates, countries))
    pairs = np.c_[countries[order], templates[order]]
    breaks = np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1
    return {
        (int(pairs[part[0], 0]), int(pairs[part[0], 1])): order[part]
        for part in np.split(np.arange(len(order)), breaks)
    }


def bootstrap_state(
    spec: GeeSpec,
    frame: pd.DataFrame,
    sizes: Mapping[str, float],
    template_types: Mapping[str, str],
) -> BootstrapState:
    """The design of one comparison. Templates are coded in the order of ``template_types``
    (template to type, as configured) and grouped by type in order of first appearance."""
    data = _prepared(spec, frame, sizes)
    y, x = _design(spec, data)
    contrast = _vector(list(x.columns), spec.contrast(data))
    countries = pd.factorize(data["iso2"])[0]
    codes = {template: k for k, template in enumerate(template_types)}
    templates = data["template"].map(codes).to_numpy()
    types = data["template"].map(template_types).to_numpy()
    by_type = [
        np.unique(templates[types == kind]) for kind in dict.fromkeys(template_types.values())
    ]
    return BootstrapState(
        y,
        x.to_numpy(),
        contrast,
        np.unique(countries),
        _cells(countries, templates),
        [t for t in by_type if len(t)],
        spec.exchangeable,
        spec.stream,
    )


_BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


@contextmanager
def _single_threaded_blas() -> Iterator[None]:
    """One BLAS thread per worker process while the pool runs; the variables are restored after."""
    saved = {name: os.environ.get(name) for name in _BLAS_THREADS}
    os.environ.update(dict.fromkeys(_BLAS_THREADS, "1"))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def cluster_bootstrap(
    frame: pd.DataFrame,
    sizes: Mapping[str, float],
    specs: Mapping[str, GeeSpec],
    comparisons: Sequence[str],
    template_types: Mapping[str, str],
    replicates: int,
    seed: int,
    workers: int,
) -> pd.DataFrame:
    """Percentile 95% interval of each odds ratio over ``replicates`` refits.
    Replicates that do not converge are counted and left out."""
    states = {c: bootstrap_state(specs[c], frame, sizes, template_types) for c in comparisons}
    jobs = [(c, seed, i) for c in comparisons for i in range(replicates)]
    if workers > 1:
        with (
            _single_threaded_blas(),
            ProcessPoolExecutor(workers, initializer=_bootstrap_setup, initargs=(states,)) as pool,
        ):
            values = list(pool.map(_bootstrap_replicate, jobs, chunksize=max(1, replicates // 50)))
    else:
        _bootstrap_setup(states)
        values = [_bootstrap_replicate(job) for job in jobs]
    results = pd.DataFrame(jobs, columns=["comparison", "seed", "index"]).assign(value=values)
    rows = []
    for comparison, part in results.groupby("comparison", sort=False):
        kept = part["value"].dropna().to_numpy()
        if not len(kept):
            raise ValueError(f"cluster bootstrap of {comparison}: no replicate converged")
        rows.append(
            {
                "analysis": "cluster bootstrap",
                "comparison": comparison,
                "ci_low": float(np.exp(np.percentile(kept, 2.5))),
                "ci_high": float(np.exp(np.percentile(kept, 97.5))),
                "replicates": replicates,
                "failed": replicates - len(kept),
            }
        )
    return pd.DataFrame(rows)


def anova_icc(values: pd.Series, groups: pd.Series) -> float:
    """One-way ANOVA intraclass correlation of ``values`` within ``groups``."""
    table = pd.DataFrame({"y": values, "g": groups}).dropna()
    sizes = table.groupby("g")["y"].size()
    means = table.groupby("g")["y"].mean()
    n, k = len(table), len(sizes)
    between = float((sizes * (means - table["y"].mean()) ** 2).sum()) / (k - 1)
    within = float(((table["y"] - table["g"].map(means)) ** 2).sum()) / (n - k)
    size = (n - float((sizes**2).sum()) / n) / (k - 1)
    denominator = between + (size - 1) * within
    return (between - within) / denominator if denominator else float("nan")
