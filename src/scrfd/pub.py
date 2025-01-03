from __future__ import annotations
import os

from dataclasses import dataclass
from onnxruntime import InferenceSession  # type: ignore
from PIL.Image import Image as PILImage
from typing import Sequence

from .base import SCRFDBase, Detections
from .schemas import Face, Point, Bbox, FaceKeypoints, Threshold


@dataclass
class SCRFD:
    _inner: SCRFDBase

    @staticmethod
    def from_session(session: InferenceSession) -> SCRFD:
        return SCRFD(SCRFDBase(session))

    @staticmethod
    def from_path(
        path: str | os.PathLike, providers: Sequence[str] | None = None
    ) -> SCRFD:
        session = InferenceSession(path, providers=providers)
        return SCRFD.from_session(session)

    def detect(self, image: PILImage, threshold: Threshold | None = None) -> list[Face]:
        if threshold is None:
            threshold = Threshold()
        assert image.mode == "RGB"
        detections = self._inner.detect(image, threshold=threshold)
        return _parse_detections(detections)


def _parse_detections(detections: Detections) -> list[Face]:
    bboxes = detections.bboxes
    keypoints = detections.keypoints

    assert len(bboxes) == len(keypoints)
    N = len(bboxes)
    if N == 0:
        return []
    assert bboxes.shape == (N, 5)
    assert keypoints.shape == (N, 5, 2)

    faces = []
    for bbox, kps in zip(bboxes, keypoints):
        bbox = [float(scalar) for scalar in bbox]
        x1, y1, x2, y2, score = bbox
        upper_left = Point(x=x1, y=y1)
        lower_right = Point(x=x2, y=y2)
        bbox = Bbox(upper_left=upper_left, lower_right=lower_right)

        kps = [Point(x=float(x), y=float(y)) for x, y in kps]
        kps = FaceKeypoints(
            left_eye=kps[0],
            right_eye=kps[1],
            nose=kps[2],
            left_mouth=kps[3],
            right_mouth=kps[4],
        )
        face = Face(
            bbox=bbox,
            keypoints=kps,
            probability=score,
        )
        faces.append(face)

    return faces
