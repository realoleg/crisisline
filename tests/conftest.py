"""Shared fixtures: the real config and ground truth from the repository."""

from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings

from crisisline.cli import extract_config, load_config, read_csv, score_config
from crisisline.extract import ExtractConfig, NumberIndex, build_index
from crisisline.score import ScoreConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

# Property tests draw the same examples on every run.
settings.register_profile("repo", derandomize=True, deadline=None, max_examples=300)
settings.load_profile("repo")


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    return load_config(REPO_ROOT / "configs" / "experiment.yaml")


@pytest.fixture(scope="session")
def index(config: dict[str, Any]) -> NumberIndex:
    return build_index(read_csv(REPO_ROOT / config["paths"]["resources"]))


@pytest.fixture(scope="session")
def index_v1(config: dict[str, Any]) -> NumberIndex:
    """Ground truth v1, for extraction cases written against it: LK 1990 and RU 8-800-333-44-34
    were not ground-truth numbers there, which is what these cases test."""
    return build_index(read_csv(REPO_ROOT / config["paths"]["resources_v1"]))


@pytest.fixture(scope="session")
def extract_cfg(config: dict[str, Any]) -> ExtractConfig:
    return extract_config(config)


@pytest.fixture(scope="session")
def score_cfg(config: dict[str, Any]) -> ScoreConfig:
    return score_config(config)
