use crate::canvas::ImageBuf;
use crate::process_image_with_model;

use anyhow::bail;
use futures::{sink::SinkExt, StreamExt};
use gloo_worker::reactor::{reactor, ReactorScope};
use leptos::logging::{error, log};

const MODEL_BYTES: &[u8] = include_bytes!("../../models/scrfd.onnx");

pub type WorkerId = u32;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum InputMsg {
    Init {
        worker_id: WorkerId,
        image: ImageBuf,
        frame_number: u32,
    },
    Image {
        frame_number: u32,
        image: ImageBuf,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputMsg {
    pub worker_id: WorkerId,
    pub image: ImageBuf,
    pub frame_number: u32,
}

type Scope = ReactorScope<InputMsg, OutputMsg>;

struct Ctx {
    model: scrfd::Scrfd,
    worker_id: Option<WorkerId>,
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
    let mut ctx = Ctx {
        model,
        worker_id: None,
    };
    loop {
        match start_worker_loop(scope, &mut ctx).await {
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

async fn start_worker_loop(scope: &mut Scope, ctx: &mut Ctx) -> anyhow::Result<()> {
    log!("starting worker loop");
    loop {
        let Some(msg) = scope.next().await else {
            log!("no input for worker, returning");
            return Ok(());
        };
        let (input, frame_number) = match msg {
            InputMsg::Init {
                worker_id,
                image,
                frame_number,
            } => {
                log!("worker got init msg, worker_id={}", worker_id);
                ctx.worker_id = Some(worker_id);
                (image, frame_number)
            }
            InputMsg::Image {
                frame_number,
                image,
            } => (image, frame_number),
        };
        let Some(worker_id) = ctx.worker_id else {
            bail!("ctx has no worker_id")
        };

        let output = {
            let input = input.to_rgba_image()?;
            let output = process_image_with_model(&ctx.model, input)?;
            ImageBuf::new(output.width(), output.height(), output.into_vec())
        };
        let output_msg = OutputMsg {
            worker_id,
            frame_number,
            image: output,
        };
        scope.send(output_msg).await?;
    }
}

fn load_model() -> anyhow::Result<scrfd::Scrfd> {
    let clock = web_time::Instant::now();
    let model = scrfd::Scrfd::from_bytes(MODEL_BYTES)?;
    log!("parsed model in {:?}", clock.elapsed());
    Ok(model)
}
