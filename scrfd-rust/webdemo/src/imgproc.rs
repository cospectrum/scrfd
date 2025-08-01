use anyhow::{anyhow, Result as AnyhowResult};
use leptos::{logging::*, prelude::*};
use scrfd::image::RgbaImage;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::ImageData;

type CanvasRef = NodeRef<leptos::html::Canvas>;
type VideoRef = NodeRef<leptos::html::Video>;

pub fn on_frame(canvas_ref: CanvasRef, video_ref: VideoRef) -> AnyhowResult<()> {
    log!("on_frame");

    let on_return = || {
        request_animation_frame(move || {
            if let Err(e) = on_frame(canvas_ref, video_ref) {
                error!("on_frame error: {:?}", e);
            }
        });
        Ok(())
    };
    let Some(canvas) = canvas_ref.get_untracked() else {
        log!("no canvas");
        return on_return();
    };
    let Some(video) = video_ref.get_untracked() else {
        log!("no video");
        return on_return();
    };
    canvas.set_width(video.video_width());
    canvas.set_height(video.video_height());

    let Some(ctx) = canvas
        .get_context("2d")
        .map_err(|e| anyhow!("get 2d ctx: {:?}", e))?
    else {
        log!("no 2d ctx");
        return on_return();
    };
    let ctx = ctx
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|e| anyhow!("2d ctx dyn cast: {:?}", e))?;

    ctx.draw_image_with_html_video_element_and_dw_and_dh(
        &video,
        0.,
        0.,
        canvas.width() as f64,
        canvas.height() as f64,
    )
    .map_err(|e| anyhow!("draw image: {:?}", e))?;
    let image_data = ctx
        .get_image_data(0., 0., canvas.width() as f64, canvas.height() as f64)
        .map_err(|e| anyhow!("get_image_data: {:?}", e))?;
    log!("got image data");

    let _img = rgba_from_image_data(image_data);
    // let img = DynamicImage::ImageRgba8(img).to_luma8();
    // let img = DynamicImage::ImageLuma8(img).to_rgba8();

    /*
    let image_data = try_image_data_from_rgba(&img)?;
    ctx.put_image_data(&image_data, 0., 0.)
        .map_err(|e| anyhow!("put_image_data: {:?}", e))?;
    */
    on_return()
}

fn try_image_data_from_rgba(img: &RgbaImage) -> AnyhowResult<ImageData> {
    let data = Clamped(img.as_ref());
    ImageData::new_with_u8_clamped_array_and_sh(data, img.width(), img.height())
        .map_err(|e| anyhow!("image data from clamped array: {:?}", e))
}

fn rgba_from_image_data(image_data: ImageData) -> RgbaImage {
    let pixels = image_data.data().0;
    RgbaImage::from_vec(image_data.width(), image_data.height(), pixels).expect("rgba image")
}
