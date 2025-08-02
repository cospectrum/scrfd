mod app;
mod inference;
mod onframe;

use inference::process_image_with_model;
use onframe::on_frame;

pub use app::App;
