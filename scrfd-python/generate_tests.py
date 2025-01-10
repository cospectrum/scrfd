from scrfd import SCRFD
from scrfd.common import draw_faces

from PIL import Image
from pathlib import Path

MODEL_PATH = Path("../models/scrfd.onnx")
IMAGES = Path("../images/")
SAVE_TO = Path("./generated")


def main() -> None:
    assert MODEL_PATH.exists()
    assert IMAGES.exists()
    SAVE_TO.mkdir(exist_ok=True)

    model = SCRFD.from_path(MODEL_PATH)

    for img_path in IMAGES.iterdir():
        img = Image.open(img_path).convert("RGB")
        faces = model.detect(img)
        img = draw_faces(img, faces)
        img.save(SAVE_TO / img_path.name)


if __name__ == "__main__":
    main()
