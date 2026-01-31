# webdemo-js

SCRFD face detection demo in the browser. Uses webcam input and draws bounding boxes and keypoints (eyes, nose, mouth) on detected faces.

```sh
pnpm install
pnpm dev
```

Uses ONNX Runtime Web with automatic backend fallback (WebNN → WebGPU → WebGL → WASM → CPU).
