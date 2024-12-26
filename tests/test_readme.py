def test_readme() -> None:
    from scrfd import SCRFD, Threshold
    from PIL import Image

    face_detector = SCRFD.from_path("./models/scrfd.onnx")
    threshold = Threshold(probability=0.4)

    image = Image.open("./images/solvay_conference_1927.jpg").convert("RGB")
    faces = face_detector.detect(image, threshold=threshold)

    for face in faces:
        bbox = face.bbox
        kps = face.keypoints
        score = face.probability
        print(f"{bbox=}, {kps=}, {score=}")


def main() -> None:
    test_readme()


if __name__ == "__main__":
    main()
