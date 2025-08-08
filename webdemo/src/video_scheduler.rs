use crate::canvas::{read_frame, write_frame, ImageBuf, RenderState};
use std::time::Duration;

use anyhow::{anyhow, Result as AnyhowResult};
use futures::{
    stream::{SelectAll, SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use gloo_worker::{reactor::ReactorBridge, Spawnable};
use leptos::{logging::*, prelude::*, task::spawn_local};
use wasm_bindgen::UnwrapThrowExt;

const INIT_STEP_SLEEP: f32 = 0.1;

type Reactor = ReactorBridge<crate::Worker>;
type Sink = SplitSink<Reactor, ImageBuf>;
type _Stream = SplitStream<Reactor>;
type Stream = SelectAll<_Stream>;

#[derive(Debug)]
struct Ctx {
    frame_number: usize,
    num_workers: i32,
    sinks: Vec<Sink>,
    stream: Stream,
}

pub fn on_video_play(state: RenderState) {
    let num_workers = get_num_workers();

    let (sinks, streams) = (0..num_workers)
        .map(|_| crate::Worker::spawner().spawn("./worker.js"))
        .map(|worker| worker.split())
        .collect::<(Vec<_>, Vec<_>)>();

    let stream = futures::stream::select_all(streams);

    let ctx = Ctx {
        frame_number: 0,
        num_workers,
        sinks,
        stream,
    };
    animation_loop(state, ctx);
}

fn animation_loop(state: RenderState, mut ctx: Ctx) {
    let sink_at = ctx.frame_number % ctx.num_workers as usize;

    spawn_local(async move {
        if ctx.frame_number < ctx.num_workers as usize {
            if let Err(e) = init_step(state, &mut ctx.sinks[sink_at]).await {
                error!("init step failed: {:?}", e);
            }
        } else if let Err(e) = request_step(state, &mut ctx.stream, &mut ctx.sinks[sink_at]).await {
            error!("request step failed: {:?}", e);
        }
        request_animation_frame(move || {
            ctx.frame_number += 1;
            animation_loop(state, ctx);
        });
    });
}

async fn init_step(state: RenderState, sink: &mut Sink) -> AnyhowResult<()> {
    log!("making init step");
    read_and_send_frame(state, sink).await?;
    gloo_timers::future::sleep(Duration::from_secs_f32(INIT_STEP_SLEEP)).await;

    Ok(())
}

async fn request_step(
    state: RenderState,
    stream: &mut Stream,
    sink: &mut Sink,
) -> AnyhowResult<()> {
    let out = stream
        .next()
        .await
        .ok_or_else(|| anyhow!("no output"))?
        .to_image_data()?;

    write_frame(state, &out)?;

    read_and_send_frame(state, sink).await?;
    Ok(())
}

async fn read_and_send_frame(state: RenderState, sink: &mut Sink) -> AnyhowResult<()> {
    let image_data = read_frame(state)?;
    let image_buf = ImageBuf::new(image_data.width(), image_data.height(), image_data.data().0);
    sink.send(image_buf).await?;
    Ok(())
}

fn get_num_workers() -> i32 {
    let num_cpus = get_num_cpus();
    assert!(num_cpus > 0);
    let num_workers = if num_cpus == 1 { 1 } else { num_cpus - 1 };
    assert!(num_workers > 0);
    num_workers
}

fn get_num_cpus() -> i32 {
    let navigator = web_sys::window()
        .expect_throw("failed to get window")
        .navigator();

    let num_cpus = navigator.hardware_concurrency() as i32;
    if num_cpus <= 0 {
        return 1;
    }
    num_cpus
}
