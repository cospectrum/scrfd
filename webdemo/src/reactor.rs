use leptos::logging::{error, log};
use scrfd::image::{DynamicImage, Rgba, RgbaImage};

use futures::{sink::SinkExt, StreamExt};
use gloo_worker::reactor::{reactor, ReactorScope};

use crate::canvas::ImageBuf;

const MODEL_BYTES: &[u8] = include_bytes!("../../models/scrfd.onnx");

fn load_model() -> scrfd::Scrfd {
    let clock = web_time::Instant::now();
    let model = scrfd::Scrfd::from_bytes(MODEL_BYTES).expect("valid scrfd model");
    log!("parsed model in {:?}", clock.elapsed());
    model
}

#[reactor]
pub async fn ModelReactor(mut scope: ReactorScope<ImageBuf, ImageBuf>) {
    let model = load_model();
    log!("starting reactor loop");

    loop {
        if let Err(e) = handle_iteration(&mut scope, &model).await {
            error!("reactor failed: {:?}", e);
        };
    }
}

async fn handle_iteration(
    scope: &mut ReactorScope<ImageBuf, ImageBuf>,
    model: &scrfd::Scrfd,
) -> anyhow::Result<()> {
    let Some(input) = scope.next().await else {
        log!("no input for reactor");
        return Ok(());
    };
    let input = input.to_rgba_image()?;
    let output = process_image_with_model(model, input)?;
    let output = ImageBuf::new(output.width(), output.height(), output.into_vec());
    scope.send(output).await?;
    Ok(())
}

fn process_image_with_model(
    model: &scrfd::Scrfd,
    mut inout: RgbaImage,
) -> anyhow::Result<RgbaImage> {
    let img = DynamicImage::ImageRgba8(inout.clone()).into_rgb8();
    let threshold = scrfd::Threshold {
        score: 0.5,
        iou: 0.6,
    };
    let clock = web_time::Instant::now();
    let faces = model.detect_with_threshold(&img, threshold)?;
    log!("face detection took: {:?}", clock.elapsed());

    draw_faces(&mut inout, &faces);
    Ok(inout)
}

fn draw_faces(img: &mut RgbaImage, faces: &[scrfd::Face]) {
    for &face in faces {
        draw_face(img, face);
    }
}

fn draw_face(img: &mut RgbaImage, face: scrfd::Face) {
    let points = [
        face.keypoints.left_eye,
        face.keypoints.right_eye,
        face.keypoints.nose,
        face.keypoints.left_mouth,
        face.keypoints.right_mouth,
    ];
    for point in points {
        let radius = 4;
        let color = Rgba([255, 0, 0, 255]);
        let center = (point.x as i32, point.y as i32);
        imageproc::drawing::draw_filled_circle_mut(img, center, radius, color);
    }

    let color = Rgba([255, 0, 0, 255]);
    let (x, y) = (face.bbox.x as i32, face.bbox.y as i32);
    let (width, height) = (face.bbox.w as u32, face.bbox.h as u32);
    let rect = imageproc::rect::Rect::at(x, y).of_size(width, height);
    imageproc::drawing::draw_hollow_rect_mut(img, rect, color);
}
