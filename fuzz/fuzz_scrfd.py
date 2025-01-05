#!/usr/bin/python3

import atheris
import os
import sys
import typing

from scrfd import SCRFD
from helpers import FuzzHelper


def main() -> None:
    SCRFD_PATH = "../models/scrfd.onnx"
    os.path.exists(SCRFD_PATH)
    atheris.instrument_all()  # type: ignore
    atheris.Setup(sys.argv, fuzz_scrfd(SCRFD_PATH))
    atheris.Fuzz()


def fuzz_scrfd(model_path: str) -> typing.Callable[[bytes], None]:
    scrfd_model = SCRFD.from_path(model_path)

    def fn(data: bytes) -> None:
        helper = FuzzHelper(data)

        img = helper.get_image()
        assert img.mode == "RGB"
        if img.width == 0:
            return

        _ = scrfd_model.detect(img)

    return fn


if __name__ == "__main__":
    main()
