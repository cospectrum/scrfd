use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::image;
use leptos::logging::log;

pub fn process_image_with_model(
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
