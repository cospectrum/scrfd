import pytest

from pathlib import Path
from scrfd import SCRFD, Face
from PIL import Image

from .utils import round_face
from .truth_faces import TRUTH_FACES


DATA_ROOT = Path("./images/")


@pytest.mark.parametrize(
    ["sample_filename", "num_faces"],
    [
        ("solvay_conference_1927.jpg", 29),
    ],
)
def test_num_faces(
    sample_filename: str,
    num_faces: int,
    scrfd_model: SCRFD,
) -> None:
    img_path = DATA_ROOT / sample_filename
    assert img_path.exists()
    img = Image.open(img_path).convert("RGB")

    faces = scrfd_model.detect(img)
    assert len(faces) == num_faces


@pytest.mark.parametrize(
    ["sample_filename", "expected_faces"],
    [
        ("newton.png", TRUTH_FACES["newton.png"]),
    ],
)
def test_truth(
    sample_filename: str,
    expected_faces: list[Face],
    scrfd_model: SCRFD,
) -> None:
    img_path = DATA_ROOT / sample_filename
    assert img_path.exists()
    img = Image.open(img_path).convert("RGB")

    faces = scrfd_model.detect(img)
    assert len(faces) == len(expected_faces)
    for face, expected_face in zip(faces, expected_faces):
        face = round_face(face)
        assert face == expected_face
