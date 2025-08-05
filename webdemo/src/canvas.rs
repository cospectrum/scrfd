use anyhow::{anyhow, Result as AnyhowResult};
use leptos::prelude::*;
use scrfd::image::RgbaImage;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlVideoElement, ImageData};

type CanvasRef = NodeRef<leptos::html::Canvas>;
type VideoRef = NodeRef<leptos::html::Video>;

#[derive(Debug, Clone, Copy)]
pub struct RenderState {
    pub canvas: CanvasRef,
    pub temporary_canvas: CanvasRef,
    pub video: VideoRef,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageBuf {
    width: u32,
    height: u32,
    pixels_base64: String,
}

impl ImageBuf {
    pub fn new(width: u32, height: u32, pixels: impl AsRef<[u8]>) -> Self {
        use base64::prelude::*;
        let pixels_base64 = BASE64_STANDARD.encode(pixels);
        Self {
            width,
            height,
            pixels_base64,
        }
    }
    pub fn to_rgba_image(&self) -> AnyhowResult<RgbaImage> {
        use base64::prelude::*;
        let buf = BASE64_STANDARD.decode(&self.pixels_base64)?;
        RgbaImage::from_vec(self.width, self.height, buf).ok_or_else(|| anyhow!("invalid ImageBuf"))
    }
    pub fn to_image_data(&self) -> AnyhowResult<ImageData> {
        let img = self.to_rgba_image()?;
        let data = Clamped(img.as_ref());
        ImageData::new_with_u8_clamped_array_and_sh(data, img.width(), img.height())
            .map_err(|e| anyhow!("image data from clamped array: {:?}", e))
    }
}

pub fn write_frame(state: RenderState, frame: &ImageData) -> AnyhowResult<()> {
    let (ctx, _) = get_canvas_ctx_for_video(state.video, state.canvas)
        .map_err(|e| anyhow!("failed to get canvas ctx for write: {:?}", e))?;

    ctx.put_image_data(frame, 0., 0.)
        .map_err(|e| anyhow!("put_image_data: {:?}", e))?;

    Ok(())
}

pub fn read_frame(state: RenderState) -> AnyhowResult<ImageData> {
    let (ctx, video) = get_canvas_ctx_for_video(state.video, state.temporary_canvas)
        .map_err(|e| anyhow!("failed to get canvas ctx for read: {:?}", e))?;

    let video_width = video.video_width();
    let video_height = video.video_height();

    ctx.draw_image_with_html_video_element_and_dw_and_dh(
        &video,
        0.,
        0.,
        video_width as f64,
        video_height as f64,
    )
    .map_err(|e| anyhow!("draw image: {:?}", e))?;
    let frame = ctx
        .get_image_data(0., 0., video_width as f64, video_height as f64)
        .map_err(|e| anyhow!("get_image_data: {:?}", e))?;

    Ok(frame)
}

fn get_canvas_ctx_for_video(
    video: VideoRef,
    canvas: CanvasRef,
) -> AnyhowResult<(CanvasRenderingContext2d, HtmlVideoElement)> {
    let canvas = canvas
        .get_untracked()
        .ok_or_else(|| anyhow!("ref has no canvas"))?;

    let video = video
        .get_untracked()
        .ok_or_else(|| anyhow!("ref has no video"))?;

    let video_width = video.video_width();
    let video_height = video.video_height();
    canvas.set_width(video_width);
    canvas.set_height(video_height);

    let ctx = canvas
        .get_context("2d")
        .map_err(|e| anyhow!("failed to get canvas ctx: {:?}", e))?
        .ok_or_else(|| anyhow!("no canvas ctx"))?
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .map_err(|e| anyhow!("canvas ctx dyn cast failed: {:?}", e))?;

    Ok((ctx, video))
}
