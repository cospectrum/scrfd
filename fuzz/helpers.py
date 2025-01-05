import atheris
import numpy as np

from PIL.Image import Image as PILImage
from PIL import Image


_MIN_INT = -10000
_MAX_INT = 10000


class FuzzHelper:
    fdp: atheris.FuzzedDataProvider  # type: ignore
    seed: int

    def __init__(self, input_bytes: bytes, seed: int | None = None) -> None:
        self.fdp = atheris.FuzzedDataProvider(input_bytes)  # type: ignore
        if seed is None:
            seed = self.get_uint()
        assert seed >= 0
        self.seed = seed

    def get_int(self, min_int: int = _MIN_INT, max_int: int = _MAX_INT) -> int:
        return self.fdp.ConsumeIntInRange(min_int, max_int)

    def get_uint(self, max_uint: int = 2 * _MAX_INT) -> int:
        return self.get_int(0, max_uint)

    def get_image(self) -> PILImage:
        rng = np.random.default_rng(seed=self.seed)

        CH = 3
        width = self.get_uint()
        height = self.get_uint()
        data = rng.bytes(height * width * CH)

        buf = np.frombuffer(data, dtype=np.uint8).reshape(height, width, CH)
        img = Image.fromarray(buf, mode="RGB")
        assert (img.height, img.width) == (height, width)
        return img
