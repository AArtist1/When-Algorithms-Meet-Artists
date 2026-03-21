"""Shared fixtures and configuration for the test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "data: data file and structure tests")
    config.addinivalue_line("markers", "models: Pydantic model validation tests")
    config.addinivalue_line("markers", "embedding: embedding shape and quality tests")
    config.addinivalue_line("markers", "umap: consensus UMAP pipeline tests")
    config.addinivalue_line("markers", "clustering: cluster count and quality tests")
    config.addinivalue_line("markers", "stats: statistical test and salience ratio tests")
    config.addinivalue_line("markers", "e2e: end-to-end pipeline tests")
    config.addinivalue_line("markers", "slow: tests requiring model downloads or heavy computation")


# ---------------------------------------------------------------------------
# Session-scoped fixtures (loaded once per test run, use REAL data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def df_public():
    """Load the full public discourse corpus (real data, 891 rows)."""
    from src.data_loading import load_public_discourse
    return load_public_discourse(DATA_DIR)


@pytest.fixture(scope="session")
def df_artist():
    """Load the full artist perspectives dataset (real data, 1259 rows)."""
    from src.data_loading import load_artist_perspectives
    return load_artist_perspectives(DATA_DIR)


@pytest.fixture(scope="session")
def df_likert():
    """Load the Likert anchor phrases (real data, 250 rows)."""
    from src.data_loading import load_likert_anchors
    return load_likert_anchors(DATA_DIR)


# ---------------------------------------------------------------------------
# Smaller fixtures for unit tests (still real data, just subsets)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def df_public_sample(df_public):
    """First 100 rows of public discourse for faster tests."""
    return df_public.head(100).copy()


@pytest.fixture(scope="session")
def df_artist_sample(df_artist):
    """First 50 rows of artist perspectives for faster tests."""
    return df_artist.head(50).copy()


# ---------------------------------------------------------------------------
# Synthetic fixtures for pure unit tests (no data dependency)
# ---------------------------------------------------------------------------

@pytest.fixture
def random_embeddings():
    """Random 1024-dim embeddings for testing pipeline logic."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 1024)).astype(np.float32)


@pytest.fixture
def random_umap_coords():
    """Random 8-dim coordinates simulating UMAP output."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 8)).astype(np.float32)


@pytest.fixture
def random_labels():
    """Random cluster labels for 100 points across 5 clusters."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 5, size=100)
