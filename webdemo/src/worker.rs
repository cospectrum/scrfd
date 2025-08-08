use crate::canvas::ImageBuf;
use crate::process_image_with_model;

use futures::{sink::SinkExt, StreamExt};
use gloo_worker::reactor::{reactor, ReactorScope};
use leptos::logging::{error, log};

type Scope = ReactorScope<ImageBuf, ImageBuf>;

const MODEL_BYTES: &[u8] = include_bytes!("../../models/scrfd.onnx");

fn load_model() -> anyhow::Result<scrfd::Scrfd> {
    let clock = web_time::Instant::now();
    let model = scrfd::Scrfd::from_bytes(MODEL_BYTES)?;
    log!("parsed model in {:?}", clock.elapsed());
    Ok(model)
}

#[reactor]
pub async fn Worker(mut scope: Scope) {
    log!("entered worker reactor");
    if let Err(e) = start_worker(&mut scope).await {
        error!("worker failed: {:?}", e);
    }
    log!("worker finished");
}

async fn start_worker(scope: &mut Scope) -> anyhow::Result<()> {
    log!("started worker");
    let model = load_model()?;
    loop {
        match start_worker_loop(scope, &model).await {
            Err(e) => {
                error!("worker loop failed: {:?}", e);
                continue;
            }
            Ok(_) => {
                log!("worker loop finished");
                return Ok(());
            }
        }
    }
}

async fn start_worker_loop(scope: &mut Scope, model: &scrfd::Scrfd) -> anyhow::Result<()> {
    log!("starting worker loop");
    loop {
        let Some(input) = scope.next().await else {
            log!("no input for worker, returning");
            return Ok(());
        };
        let input = input.to_rgba_image()?;
        let output = process_image_with_model(model, input)?;
        let output = ImageBuf::new(output.width(), output.height(), output.into_vec());
        scope.send(output).await?;
    }
}
