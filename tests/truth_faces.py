from scrfd import Face, Bbox, FaceKeypoints, Point


TRUTH_FACES: dict[str, list[Face]] = {
    "newton.png": [
        Face(
            bbox=Bbox(upper_left=Point(x=334, y=89), lower_right=Point(x=438, y=232)),
            probability=0.81,
            keypoints=FaceKeypoints(
                left_eye=Point(x=371, y=143),
                right_eye=Point(x=419, y=145),
                nose=Point(x=399, y=177),
                left_mouth=Point(x=371.0, y=197.0),
                right_mouth=Point(x=409, y=199),
            ),
        ),
    ],
    "gauss.png": [
        Face(
            bbox=Bbox(
                upper_left=Point(x=376, y=203),
                lower_right=Point(x=654, y=568),
            ),
            probability=0.79,
            keypoints=FaceKeypoints(
                left_eye=Point(x=469, y=353),
                right_eye=Point(x=594, y=346),
                nose=Point(x=549, y=436.0),
                left_mouth=Point(x=487.0, y=489.0),
                right_mouth=Point(x=589, y=482.0),
            ),
        )
    ],
}
