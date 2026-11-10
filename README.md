# crisisline

Do language models give the correct crisis line and emergency number for the user's country?

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Download the raw generations

The model responses are released on Zenodo:
[10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX). Unpack the archive at the
repository root. It holds `artifacts/generations/` and `artifacts/robustness/generations/`
(responses and run manifests) and `artifacts/scores_prefix.parquet` (the scores of the extraction
before the audit's fixes, which the current code does not produce).

They were generated with `crisisline translate`, `crisisline generate` (API models, an
`OPENROUTER_API_KEY` in `.env`, see `.env.example`) and `crisisline colab` (local models on Google
Colab). The labelled files in `data/annotations/` were drawn with `crisisline audit sample
[--changed]` and `crisisline unknowns [--flagged]`.

## Reproduce

```bash
uv run crisisline reproduce
```

It runs offline from the raw generations, `data/` and `configs/`, in this order: validate, build,
score (ground truth v2, v1, and v1 without labels), audit score, unknowns --estimate, analyze,
paper.

| Output | What it is |
|---|---|
| log: validation | errors and warnings of the ground truth in `data/` |
| `artifacts/prompts.jsonl` | the 1,252 prompts sent to each model |
| `artifacts/m2_resources.csv` | the crisis line and emergency numbers each M2 system prompt names |
| `artifacts/scores.parquet` | every response scored against ground truth v2: numbers found, their class, the outcomes |
| `artifacts/unknowns.csv` | numbers that are neither in the ground truth nor labelled |
| `artifacts/scores_v1.parquet`, `unknowns_v1.csv` | the same against ground truth v1 |
| `artifacts/scores_prelabels.parquet`, `unknowns_prelabels.csv` | the same against ground truth v1 with no labels: the population of the fabrication estimate |
| log: extraction audit | precision and recall of number extraction, overall, per script group and per language, for the audit sample and the re-audit |
| `artifacts/fabrication_estimate.json` | the share of unknown-number mentions labelled fabricated, with intervals, per stratum and overall |
| `artifacts/analysis/hypotheses.csv` | the five confirmatory tests (`cue_country_vs_city`, `language_en_vs_local`, `model_size`, `us_default`, `grounding_verified_number`) with Holm-adjusted p in each ground-truth version, then the sensitivity analyses |
| `artifacts/analysis/bootstrap.csv`, `icc.csv`, `us_default_breakdown.csv`, `grounding_verified_number_excluded.csv`, `extraction_comparison.csv` | cluster bootstrap intervals, intraclass correlations, and the tables behind the tests |
| `artifacts/analysis/outcomes_*.csv` | every outcome by model, by model and condition, and by every cell |
| `artifacts/analysis/*.csv` (others) | descriptive tables: referral outcomes, the Qwen size ladder, foreign numbers, default countries, location questions, Georgia, blocked answers, temperature robustness, loops, fabrication by model |
| `artifacts/analysis/exploratory/` | exploratory analyses: grounding, language gap, fabrication in reader terms, near-misses, templates, countries and WHO regions, directories |
| `artifacts/analysis/models.csv` | each model's backend, identifier, responses per prompt, records and versions seen |
| `artifacts/analysis/figures/` | figures as vector PDF at ACL widths |
| `artifacts/analysis/examples.jsonl` | example candidates chosen by rule, pending human review |
| `artifacts/analysis/notes.md`, `manifest.json` | caveats on the intervals; git commit, package versions and input checksums |
| `artifacts/analysis/scores/` | the robustness block and the official-sources-only variant, scored |
| `artifacts/paper/figures/` | the paper's figures as PDF and PNG at the printed size, each with a CSV of exactly the data it plots and its prompt subset |
| `artifacts/paper/tables/`, `numbers.csv` | the paper's tables, one row per value with its 95% interval and prompt subset, and every number the text cites |

## Evaluate a new model

Add the model under `models:` in `configs/models.yaml`: `backend: openrouter` with the OpenRouter
`model` slug, or `backend: ollama` with an Ollama `tag`. To group it with others in the
exploratory tables, add its model_id to `analysis.exploratory.model_groups` in
`configs/experiment.yaml`; otherwise it forms a group of its own. Then:

```bash
uv run crisisline generate --model M   # API model: OPENROUTER_API_KEY in .env; local model: ollama serve, ollama pull TAG
uv run crisisline reproduce            # scores every model against each ground-truth version, then analyze
```

Local models run against the Ollama server in `configs/models.yaml` (`ollama.host`, by default
this machine). `uv run crisisline colab --model M` is an optional helper that runs the same
generate on a Google Colab GPU; it needs the `colab` CLI (google-colab-cli), signed in. The local
runs in the paper used it, on an NVIDIA L4 (`gpu_name` in the run manifests).

## Rerun from scratch

```bash
cp .env.example .env                    # set OPENROUTER_API_KEY
uv run crisisline translate --all --force   # re-translates qa=auto entries; ~35 min at 15 requests/min, ~$1
uv run crisisline build                 # seconds
for m in gpt-6-luna gemini-3.5-flash-lite claude-haiku-5.5; do uv run crisisline generate --model $m & done; wait   # ~4 h, ~$6
for m in qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b gemma4-e2b gemma4-e4b llama3.1-8b local-8; do uv run crisisline generate --model $m && uv run crisisline generate --robustness --model $m; done   # ~45 h on an L4
uv run crisisline reproduce             # ~4 min, offline
```

The local models need `ollama serve` and `ollama pull` of each tag in `configs/models.yaml`. With
`crisisline colab --model M [M ...]` (and `--robustness`) the same runs go to Colab instead: about
75 Colab compute units for the eight models.

Generation cannot be reproduced byte for byte: provider models change behind the same name, and
API sampling is not deterministic. This is why the raw generations are published. The manual
labels in `data/annotations/` (extraction audit, unknown numbers) are published as data and are
not regenerated; numbers that a new run produces and no label covers count as unknown.
