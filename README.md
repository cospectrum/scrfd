# SCRFD
[![github]](https://github.com/cospectrum/scrfd)
[![ci]](https://github.com/cospectrum/scrfd/actions)

[github]: https://img.shields.io/badge/github-cospectrum/scrfd-8da0cb?logo=github
[ci]: https://github.com/cospectrum/scrfd/workflows/ci/badge.svg

Efficient face detection using SCRFD.

```sh
pip install scrfd
```

```python
from scrfd import SCRFD, Threshold
from PIL import Image

face_detector = SCRFD.from_path("./models/scrfd.onnx")
threshold = Threshold(probability=0.4)

image = Image.open("./images/solvay_conference_1927.png").convert("RGB")
faces = face_detector.detect(image, threshold=threshold)

for face in faces:
    bbox = face.bbox
    kps = face.keypoints
    score = face.probability
    print(f"{bbox=}, {kps=}, {score=}")
```

<img align="middle" src="./readme.png" alt="face detection">
