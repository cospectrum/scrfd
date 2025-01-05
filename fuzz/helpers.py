import atheris
import numpy as np

from PIL.Image import Image as PILImage
from PIL import Image


MIN_INT = -2000
MAX_INT = 2000
MAX_UINT = 2 * MAX_INT


class FuzzHelper:
    fdp: atheris.FuzzedDataProvider  # type: ignore
    rng: np.random.Generator

    def __init__(self, input_bytes: bytes, seed: int | None = None) -> None:
        self.fdp = atheris.FuzzedDataProvider(input_bytes)  # type: ignore
        if seed is None:
            seed = self.get_uint()
        assert seed >= 0
        self.rng = np.random.default_rng(seed=seed)

    def get_int(self, min_int: int = MIN_INT, max_int: int = MAX_INT) -> int:
        assert min_int <= max_int
        return self.fdp.ConsumeIntInRange(min_int, max_int)

    def get_uint(self, max_uint: int = MAX_UINT) -> int:
        return self.get_int(0, max_uint)

    def get_non_zero_uint(self, max_uint: int = MAX_UINT) -> int:
        return self.get_int(1, max_uint)

    def get_bytes(self, size: int) -> bytes | None:
        buf = self.fdp.ConsumeBytes(size)
        if len(buf) != size:
            return None
        return buf

    def get_rgb_image(self) -> PILImage:
        CH = 3
        height = self.get_uint()
        width = self.get_uint()

        buf_size = width * height * CH
        buf = self.rng.bytes(buf_size)
        assert len(buf) == buf_size

        array = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, CH)
        img = Image.fromarray(array, mode="RGB")
        assert (img.height, img.width) == (height, width)
        return img
