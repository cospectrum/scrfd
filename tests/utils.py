from scrfd import Face, Bbox, Point, FaceKeypoints


def round_face(face: Face) -> Face:
    probability = int(face.probability * 100) / 100
    return Face(
        probability=probability,
        bbox=round_bbox(face.bbox),
        keypoints=round_keypoints(face.keypoints),
    )


def round_bbox(bbox: Bbox) -> Bbox:
    return Bbox(
        upper_left=round_point(bbox.upper_left),
        lower_right=round_point(bbox.lower_right),
    )


def round_keypoints(kps: FaceKeypoints) -> FaceKeypoints:
    return FaceKeypoints(
        left_eye=round_point(kps.left_eye),
        right_eye=round_point(kps.right_eye),
        nose=round_point(kps.nose),
        left_mouth=round_point(kps.left_eye),
        right_mouth=round_point(kps.right_mouth),
    )


def round_point(point: Point) -> Point:
    return Point(x=int(point.x), y=int(point.y))
