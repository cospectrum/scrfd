use crate::{
    canvas::{read_frame, write_frame, ImageBuf, RenderState},
    common,
};
use std::time::Duration;

use anyhow::{anyhow, Result as AnyhowResult};
use futures::StreamExt;
use gloo_worker::reactor::ReactorScoped;
use leptos::{logging::*, prelude::*, task::spawn_local};
use wasm_bindgen::UnwrapThrowExt;

const INIT_STEP_SLEEP: f32 = 0.05;

type Reactor<R> = gloo_worker::reactor::ReactorBridge<R>;

#[derive(Debug, Clone, Copy)]
struct Ctx {
    frame_number: usize,
    num_workers: i32,
}

pub fn on_video_play<R>(state: RenderState, reactor: Reactor<R>)
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    let num_cpus = get_num_cpus();
    let num_workers = common::get_num_workers(num_cpus);
    let ctx = Ctx {
        frame_number: 0,
        num_workers,
    };
    log!("animation ctx: {:?}", ctx);
    animation_loop(state, reactor, ctx);
}

fn animation_loop<R>(state: RenderState, mut reactor: Reactor<R>, mut ctx: Ctx)
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    spawn_local(async move {
        if ctx.frame_number < ctx.num_workers as usize {
            if let Err(e) = init_step(state, &mut reactor).await {
                error!("init step failed: {:?}", e);
            }
        } else if let Err(e) = request_step(state, &mut reactor).await {
            error!("request step failed: {:?}", e);
        }
        request_animation_frame(move || {
            ctx.frame_number += 1;
            animation_loop(state, reactor, ctx);
        });
    });
}

async fn init_step<R>(state: RenderState, reactor: &mut Reactor<R>) -> AnyhowResult<()>
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    log!("making init step");
    read_and_send_frame(state, reactor)?;
    gloo_timers::future::sleep(Duration::from_secs_f32(INIT_STEP_SLEEP)).await;

    Ok(())
}

async fn request_step<R>(state: RenderState, reactor: &mut Reactor<R>) -> AnyhowResult<()>
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    let out = reactor
        .next()
        .await
        .ok_or_else(|| anyhow!("no output"))?
        .to_image_data()?;

    write_frame(state, &out)?;

    read_and_send_frame(state, reactor)?;
    Ok(())
}

fn read_and_send_frame<R>(state: RenderState, reactor: &mut Reactor<R>) -> AnyhowResult<()>
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    let image_data = read_frame(state)?;
    let image_buf = ImageBuf::new(image_data.width(), image_data.height(), image_data.data().0);
    reactor.send_input(image_buf);
    Ok(())
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
