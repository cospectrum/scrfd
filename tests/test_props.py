import numpy as np

from hypothesis import (
    given,
    settings,
    strategies as st,
)

from PIL.Image import Image as PILImage

from scrfd import SCRFD
from scrfd.schemas import Threshold
from scrfd.base import SCRFDBase

from .utils.proptest import arbitrary_rgb_image as arb_img


@given(
    img=arb_img((1, 20), (1, 20)),
)
def test_blob_from_image(img: PILImage) -> None:
    img_buf = np.array(img)
    assert img_buf.shape == (img.height, img.width, 3)

    blob = SCRFDBase.blob_from_image(img_buf)
    assert blob.shape == (3, img.height, img.width)


@given(
    img=arb_img((1, 200), (1, 200)),
    width=st.integers(1, 200),
    height=st.integers(1, 200),
)
def test_resize(img: PILImage, width: int, height: int) -> None:
    resized = SCRFDBase.resize(img, width=width, height=height)
    assert resized.height == height
    assert resized.width == width


@settings(
    deadline=3 * 1000,
    max_examples=200,
)
@given(
    img=arb_img((1, 1000), (1, 1000)),
    probability=st.none() | st.floats(0.0, 1.0),
    nms=st.none() | st.floats(0.0, 1.0),
)
def test_scrfd(
    img: PILImage,
    probability: float | None,
    nms: float | None,
    scrfd_model: SCRFD,
) -> None:
    print(f"{img.height=}, {img.width=}, {probability=}, {nms=}")
    threshold = Threshold()
    if nms is not None:
        threshold.nms = nms
    if probability is not None:
        threshold.probability = probability

    faces = scrfd_model.detect(img, threshold)
    for face in faces:
        assert face.probability >= threshold.probability
