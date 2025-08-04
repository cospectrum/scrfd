from PIL import ImageDraw
from PIL.Image import Image as PILImage

from .schemas import Face, Point

_Color = float | tuple[int, ...] | str


def draw_faces(
    image: PILImage,
    faces: list[Face],
    *,
    keypoint_radius: int = 4,
    keypoint_color: _Color = "red",
    box_width: int = 4,
    box_color: _Color = "red",
) -> PILImage:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    def to_tuple(p: Point) -> tuple[int, int]:
        return int(p.x), int(p.y)

    for face in faces:
        bbox = face.bbox
        p1 = to_tuple(bbox.upper_left)
        p2 = to_tuple(bbox.lower_right)
        draw.rectangle((p1, p2), outline=box_color, width=box_width)

        kps = face.keypoints
        keypoints = [
            kps.left_eye,
            kps.right_eye,
            kps.nose,
            kps.left_mouth,
            kps.right_mouth,
        ]
        for kp in keypoints:
            x, y = to_tuple(kp)
            r = keypoint_radius
            ellipse = [(x - r, y - r), (x + r, y + r)]
            draw.ellipse(ellipse, fill=keypoint_color)

    return image
