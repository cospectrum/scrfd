import pytest

from pathlib import Path
from PIL import Image

from scrfd import SCRFD, Face
from scrfd.common import draw_faces

from .utils import keypoints_within_box, round_face
from .truth_faces import TRUTH_FACES


DATA_ROOT = Path("../images/")
TRUTH_ROOT = Path("./tests/truth")
assert DATA_ROOT.exists()
assert TRUTH_ROOT.exists()


@pytest.mark.parametrize(
    ["sample_filename", "num_faces"],
    [
        ("solvay_conference_1927.png", 29),
        ("newton.png", 1),
        ("gauss.png", 1),
    ],
)
def test_num_faces(
    sample_filename: str,
    num_faces: int,
    scrfd_model: SCRFD,
) -> None:
    img_path = DATA_ROOT / sample_filename
    truth_path = TRUTH_ROOT / sample_filename
    assert img_path.exists()
    assert truth_path.exists()
    img = Image.open(img_path).convert("RGB")
    truth = Image.open(truth_path).convert("RGB")

    faces = scrfd_model.detect(img)
    assert len(faces) == num_faces
    for face in faces:
        assert keypoints_within_box(face.keypoints, face.bbox)

    img = draw_faces(img, faces)
    assert img == truth


@pytest.mark.parametrize(
    ["sample_filename", "expected_faces"],
    [
        ("newton.png", TRUTH_FACES["newton.png"]),
        ("gauss.png", TRUTH_FACES["gauss.png"]),
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
        assert face.probability == expected_face.probability
        assert face.bbox == expected_face.bbox
        assert face.keypoints == expected_face.keypoints
