import pytest
import os

from pathlib import Path
from scrfd import SCRFD

KiB = 2**10
MiB = (2**10) * KiB


@pytest.fixture
def scrfd_model() -> SCRFD:
    path = Path("./models/scrfd.onnx")
    assert path.exists()
    size = os.path.getsize(path)
    assert 1 * MiB < size < 4 * MiB
    return SCRFD.from_path(path)
