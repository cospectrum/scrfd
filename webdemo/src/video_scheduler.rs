use crate::{
    canvas::{read_frame, write_frame, ImageBuf, RenderState},
    worker,
};
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
type Sink = SplitSink<Reactor, worker::InputMsg>;
type _Stream = SplitStream<Reactor>;
type Stream = SelectAll<_Stream>;

#[derive(Debug)]
struct Ctx {
    current_frame_number: u32,
    last_shown_frame_number: Option<u32>,
    sinks: Vec<Sink>,
    stream: Stream,
}

impl Ctx {
    fn num_workers(&self) -> usize {
        self.sinks.len()
    }
}

pub fn on_video_play(state: RenderState) {
    let num_workers = get_num_workers();
    assert!(num_workers > 0);
    log!("num_workers={}", num_workers);

    let (sinks, streams) = (0..num_workers)
        .map(|_| crate::Worker::spawner().spawn("./worker.js"))
        .map(|worker| worker.split())
        .collect::<(Vec<_>, Vec<_>)>();

    let stream = futures::stream::select_all(streams);

    let ctx = Ctx {
        last_shown_frame_number: None,
        current_frame_number: 0,
        sinks,
        stream,
    };
    animation_loop(state, ctx);
}

fn animation_loop(state: RenderState, mut ctx: Ctx) {
    spawn_local(async move {
        if ctx.current_frame_number < ctx.num_workers() as u32 {
            if let Err(e) = init_step(state, &mut ctx).await {
                error!("init step failed: {:?}", e);
            }
        } else if let Err(e) = request_step(state, &mut ctx).await {
            error!("request step failed: {:?}", e);
        }
        request_animation_frame(move || {
            ctx.current_frame_number += 1;
            animation_loop(state, ctx);
        });
    });
}

async fn init_step(state: RenderState, ctx: &mut Ctx) -> AnyhowResult<()> {
    let worker_id = ctx.current_frame_number;
    log!("making init step, worker_id={}", worker_id);

    assert!(worker_id < ctx.sinks.len() as u32);
    let sink = &mut ctx.sinks[worker_id as usize];

    let image = read_image_buf(state)?;
    let msg = worker::InputMsg::Init {
        worker_id,
        image,
        frame_number: ctx.current_frame_number,
    };
    sink.send(msg).await?;

    gloo_timers::future::sleep(Duration::from_secs_f32(INIT_STEP_SLEEP)).await;

    Ok(())
}

async fn request_step(state: RenderState, ctx: &mut Ctx) -> AnyhowResult<()> {
    let output_msg = ctx
        .stream
        .next()
        .await
        .ok_or_else(|| anyhow!("no output"))?;

    let worker::OutputMsg {
        worker_id,
        image,
        frame_number,
    } = output_msg;

    let image_data = image.to_image_data()?;
    if ctx.last_shown_frame_number.is_none() || frame_number > ctx.last_shown_frame_number.unwrap()
    {
        ctx.last_shown_frame_number = Some(frame_number);
        write_frame(state, &image_data)?;
    } else {
        log!(
            "dropped frame_number={} from worker_id={}, last_shown_frame_number={}",
            frame_number,
            worker_id,
            ctx.last_shown_frame_number.unwrap(),
        )
    }

    let sink = &mut ctx.sinks[worker_id as usize];
    let next_image = read_image_buf(state)?;
    let msg = worker::InputMsg::Image {
        image: next_image,
        frame_number: ctx.current_frame_number + 1,
    };
    sink.send(msg).await?;

    Ok(())
}

fn read_image_buf(state: RenderState) -> AnyhowResult<ImageBuf> {
    let image_data = read_frame(state)?;
    let image_buf = ImageBuf::new(image_data.width(), image_data.height(), image_data.data().0);
    Ok(image_buf)
}

fn get_num_workers() -> i32 {
    let num_cpus = get_num_cpus();
    log!("num_cpus={}", num_cpus);
    assert!(num_cpus > 0);
    match num_cpus {
        1 => 1,
        2 => 1,
        3 => 2,
        n => n - 2,
    }
}

fn get_num_cpus() -> i32 {
    let navigator = web_sys::window()
        .expect_throw("failed to get window")
        .navigator();

    let num_cpus = navigator.hardware_concurrency() as i32;
    num_cpus.max(1)
}
