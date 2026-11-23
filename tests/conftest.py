import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import synthetic_diagrams  # noqa: E402


@pytest.fixture(scope="session")
def diagrams():
    return synthetic_diagrams.generate(count=4, seed=7)


@pytest.fixture(scope="session")
def dataset_zip(diagrams) -> bytes:
    return synthetic_diagrams.dataset_zip(diagrams)
