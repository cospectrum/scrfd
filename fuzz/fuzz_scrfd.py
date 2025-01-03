#!/usr/bin/python3

import atheris
import os
import sys
import typing

import numpy as np

from PIL import Image
from PIL.Image import Image as PILImage

from scrfd import SCRFD


def main() -> None:
    SCRFD_PATH = "../models/scrfd.onnx"
    os.path.exists(SCRFD_PATH)
    atheris.instrument_all()  # type: ignore
    atheris.Setup(sys.argv, fuzz_scrfd(SCRFD_PATH))
    atheris.Fuzz()


def fuzz_scrfd(model_path: str) -> typing.Callable[[bytes], None]:
    scrfd_model = SCRFD.from_path(model_path)

    def fn(img: PILImage) -> None:
        _ = scrfd_model.detect(img)

    return lambda data: test_one_input(data, fn)


def test_one_input(data: bytes, fn: typing.Callable[[PILImage], typing.Any]) -> None:
    img = random_rgb_image(data)
    if img is None:
        return
    fn(img)


def random_rgb_image(data: bytes) -> PILImage | None:
    CH = 3
    data = data[: len(data) - len(data) % CH]
    assert len(data) % CH == 0
    N = len(data) // CH
    if N == 0:
        return None
    height, width = split_number(N)
    assert height * width == N
    buf = np.frombuffer(data, dtype=np.uint8).reshape(height, width, CH)
    img = Image.fromarray(buf)
    assert (img.height, img.width) == (height, width)
    assert img.mode == "RGB"
    return img


def split_number(n: int) -> tuple[int, int]:
    assert n > 0
    max_q, max_p = (n, 1)
    for p in range(1, n):
        if n % p != 0:
            continue
        q = n // p
        if min(q, p) > min(max_q, max_p):
            max_q = q
            max_p = p
    return max_q, max_p


if __name__ == "__main__":
    main()
