from __future__ import annotations

from dataclasses import dataclass
from PIL.Image import Image as PILImage

from .base import SCRFDBase, Detections
from .schemas import Face, Point, Bbox, FaceKeypoints, Threshold


@dataclass
class SCRFD:
    _inner: SCRFDBase

    @staticmethod
    def from_path(path: str, providers: list[str] | None = None) -> SCRFD:
        return SCRFD(SCRFDBase(path, providers))

    def detect(
        self,
        image: PILImage,
        threshold: Threshold | None = None,
        max_faces: int | None = None,
    ) -> list[Face]:
        if threshold is None:
            threshold = Threshold()
        image = image if image.mode == "RGB" else image.convert("RGB")
        detections = self._inner.detect(image, threshold=threshold, max_num=max_faces)
        return parse_detections(detections)


def parse_detections(detections: Detections) -> list[Face]:
    bboxes = detections.bboxes
    keypoints = detections.keypoints
    assert bboxes.shape[0] == keypoints.shape[0]

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
