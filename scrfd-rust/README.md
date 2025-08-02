# scrfd-rust

## Getting Started

```rs
let face_detector = scrfd::Scrfd::from_bytes(MODEL_BYTES)?;

let img = image::ImageReader::new(io::Cursor::new(IMAGE_BYTES))
    .with_guessed_format()?
    .decode()?
    .into_rgb8();
let faces = face_detector.detect(&img)?;
```
