import os
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def change_directory(path: os.PathLike) -> Iterator[None]:
    path = Path(path)
    assert path.exists()
    assert path.is_dir()
    go_back = Path.cwd()
    try:
        os.chdir(Path(path))
        yield
    finally:
        os.chdir(go_back)
