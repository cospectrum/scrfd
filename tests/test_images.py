import pytest

from pathlib import Path
from scrfd import SCRFD, Face
from PIL import Image

from .utils import round_face, point_within_box
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
    for face in faces:
        kps = face.keypoints
        points = [
            kps.nose,
            kps.left_eye,
            kps.right_eye,
            kps.left_mouth,
            kps.right_mouth,
        ]
        for point in points:
            assert point_within_box(point, face.bbox)


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
