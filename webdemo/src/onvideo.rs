use anyhow::{anyhow, Result as AnyhowResult};
use futures::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use gloo_worker::reactor::{ReactorBridge, ReactorScoped};
use leptos::{logging::*, prelude::*, task::spawn_local};

use crate::canvas::{read_frame, write_frame, ImageBuf, RenderState};

pub fn on_video_play<R>(state: RenderState, reactor: gloo_worker::reactor::ReactorBridge<R>)
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    let (sink, stream) = reactor.split();
    start_writer(state, stream);
    start_reader(state, sink);
}

fn start_reader<R>(state: RenderState, mut sink: SplitSink<ReactorBridge<R>, ImageBuf>)
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    spawn_local(async move {
        if let Err(e) = reader_step(state, &mut sink).await {
            error!("reader step failed: {:?}", e);
        }
        request_animation_frame(move || start_reader(state, sink));
    });
}

async fn reader_step<R>(
    state: RenderState,
    sink: &mut SplitSink<ReactorBridge<R>, ImageBuf>,
) -> AnyhowResult<()>
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    let image_data = read_frame(state)?;
    let image_buf = ImageBuf::new(image_data.width(), image_data.height(), image_data.data().0);
    sink.send(image_buf).await?;
    Ok(())
}

fn start_writer<R>(state: RenderState, mut stream: SplitStream<ReactorBridge<R>>)
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    spawn_local(async move {
        if let Err(e) = write_step(state, &mut stream).await {
            error!("write step failed: {:?}", e);
        }
        request_animation_frame(move || start_writer(state, stream));
    });
}

async fn write_step<R>(
    state: RenderState,
    stream: &mut SplitStream<ReactorBridge<R>>,
) -> AnyhowResult<()>
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    let frame = stream
        .next()
        .await
        .ok_or_else(|| anyhow!("no stream output"))?
        .to_image_data()?;

    write_frame(state, &frame)
}
