import pytest

from pathlib import Path
from scrfd import SCRFD
from scrfd.base import SCRFDBase


@pytest.fixture(scope="package")
def scrfd_model() -> SCRFD:
    path = Path("./models/scrfd.onnx")
    assert path.exists()
    return SCRFD.from_path(path)


@pytest.fixture(scope="package")
def scrfd_base_model(scrfd_model: SCRFD) -> SCRFDBase:
    return scrfd_model._inner
