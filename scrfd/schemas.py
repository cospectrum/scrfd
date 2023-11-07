from pydantic import BaseModel


class Threshold(BaseModel):
    nms: float = 0.5
    probability: float = 0.4


class Point(BaseModel):
    x: float
    y: float


class Bbox(BaseModel):
    upper_left: Point
    lower_right: Point

    def height(self) -> float:
        return self.lower_right.y - self.upper_left.y

    def width(self) -> float:
        return self.lower_right.x - self.upper_left.x


class FaceKeypoints(BaseModel):
    left_eye: Point
    right_eye: Point
    nose: Point
    left_mouth: Point
    right_mouth: Point


class Face(BaseModel):
    bbox: Bbox
    probability: float
    keypoints: FaceKeypoints
