use scrfd::TractResult;
use std::io;

const MODEL_BYTES: &[u8] = include_bytes!("../../../models/scrfd.onnx");
const IMAGE_BYTES: &[u8] = include_bytes!("../../../images/solvay_conference_1927.png");

fn main() -> TractResult<()> {
    let face_detector = scrfd::Scrfd::from_bytes(MODEL_BYTES)?;

    let img = image::ImageReader::new(io::Cursor::new(IMAGE_BYTES))
        .with_guessed_format()?
        .decode()?
        .into_rgb8();
    let faces = face_detector.detect(&img)?;
    for face in faces {
        dbg!(face);
    }
    Ok(())
}
