use leptos::logging::{error, log};
use scrfd::image::{DynamicImage, Rgba, RgbaImage};

use futures::{sink::SinkExt, StreamExt};
use gloo_worker::reactor::{reactor, ReactorScope};

use crate::canvas::ImageBuf;

const MODEL_BYTES: &[u8] = include_bytes!("../../models/scrfd.onnx");

fn load_model() -> anyhow::Result<scrfd::Scrfd> {
    let clock = web_time::Instant::now();
    let model = scrfd::Scrfd::from_bytes(MODEL_BYTES)?;
    log!("parsed model in {:?}", clock.elapsed());
    Ok(model)
}

#[reactor]
pub async fn ModelReactor(mut scope: ReactorScope<ImageBuf, ImageBuf>) {
    log!("entered reactor");
    loop {
        if let Err(e) = start_reactor(&mut scope).await {
            error!("reactor failed: {:?}", e);
        }
    }
}

async fn start_reactor(scope: &mut ReactorScope<ImageBuf, ImageBuf>) -> anyhow::Result<()> {
    log!("started reactor");
    let model = load_model()?;
    loop {
        if let Err(e) = start_reactor_loop(scope, &model).await {
            error!("reactor loop failed: {:?}", e);
        }
    }
}

async fn start_reactor_loop(
    scope: &mut ReactorScope<ImageBuf, ImageBuf>,
    model: &scrfd::Scrfd,
) -> anyhow::Result<()> {
    log!("starting reactor loop");
    loop {
        let Some(input) = scope.next().await else {
            log!("no input for reactor");
            continue;
        };
        let input = input.to_rgba_image()?;
        let output = process_image_with_model(model, input)?;
        let output = ImageBuf::new(output.width(), output.height(), output.into_vec());
        scope.send(output).await?;
    }
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
    let x = face.bbox.upper_left.x as i32;
    let y = face.bbox.upper_left.y as i32;
    let (width, height) = (face.bbox.width() as u32, face.bbox.height() as u32);
    let rect = imageproc::rect::Rect::at(x, y).of_size(width, height);
    imageproc::drawing::draw_hollow_rect_mut(img, rect, color);
}
