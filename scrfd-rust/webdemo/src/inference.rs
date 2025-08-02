use leptos::logging::log;
use scrfd::image::{DynamicImage, Rgba, RgbaImage};

pub fn process_image_with_model(
    model: &scrfd::Scrfd,
    mut inout: RgbaImage,
) -> anyhow::Result<RgbaImage> {
    let img = DynamicImage::ImageRgba8(inout.clone()).to_rgb8();
    let threshold = scrfd::Threshold {
        score: 0.5,
        iou: 0.6,
    };
    let faces = model.detect_with_threshold(&img, threshold)?;
    log!("detected {:?} faces", faces.len());
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
    let (x, y) = (face.bbox.x, face.bbox.y);
    let (width, height) = (face.bbox.w as u32, face.bbox.h as u32);
    let rect = imageproc::rect::Rect::at(x as i32, y as i32).of_size(width, height);
    imageproc::drawing::draw_hollow_rect_mut(img, rect, color);
}
