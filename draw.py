from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage

from scrfd import SCRFD, Face, Point


def main() -> None:
    model_path = "./models/scrfd.onnx"
    face_detector = SCRFD.from_path(model_path)

    image_path = "./images/solvay_conference_1927.jpg"
    image = Image.open(image_path).convert("RGB")

    faces = face_detector.detect(image)
    result = draw_faces(image, faces)
    result.save("draw_result.jpg")


def draw_faces(
    image: PILImage,
    faces: list[Face],
    radius: int = 3,
    box_width: int = 4,
) -> PILImage:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    def to_tuple(p: Point) -> tuple[int, int]:
        return int(p.x), int(p.y)

    for face in faces:
        bbox = face.bbox
        p1 = to_tuple(bbox.upper_left)
        p2 = to_tuple(bbox.lower_right)
        draw.rectangle((p1, p2), outline="red", width=box_width)

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
            ellipse = [(x - radius, y - radius), (x + radius, y + radius)]
            draw.ellipse(ellipse, fill="red")

    return image


if __name__ == "__main__":
    main()
