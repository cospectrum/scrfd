import numpy as np
from hypothesis import (
    strategies as st,
)
from hypothesis.extra import numpy as npst
from PIL import Image
from PIL.Image import Image as PILImage


@st.composite
def arbitrary_rgb_image(
    draw: st.DrawFn,
    height: tuple[int, int],
    width: tuple[int, int],
) -> PILImage:
    assert height[0] <= height[1]
    assert width[0] <= width[1]
    h = draw(st.integers(height[0], height[1]))
    w = draw(st.integers(width[0], width[1]))
    buf = draw(
        npst.arrays(
            dtype=np.uint8,
            shape=(h, w, 3),
        )
    )
    img = Image.fromarray(buf)
    assert img.mode == "RGB"
    return img
