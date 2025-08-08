mod app;
mod canvas;
mod effects;
mod image_processing;
mod video_scheduler;
mod worker;

use image_processing::process_image_with_model;
use video_scheduler::on_video_play;

pub use app::App;
pub use worker::Worker;
