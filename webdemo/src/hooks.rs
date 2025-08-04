use anyhow::{anyhow, Result as AnyhowResult};
use futures::StreamExt;
use gloo_worker::reactor::ReactorScoped;
use leptos::{logging::*, prelude::*, task::spawn_local};

use crate::canvas::{read_frame, write_frame, ImageBuf, RenderState};

pub fn on_video_play<R>(state: RenderState, mut reactor: gloo_worker::reactor::ReactorBridge<R>)
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    spawn_local(async move {
        if let Err(e) = handle_frame(state, &mut reactor).await {
            error!("handle_frame failed: {:?}", e);
        }
        request_animation_frame(move || on_video_play(state, reactor));
    });
}

async fn handle_frame<R>(
    state: RenderState,
    reactor: &mut gloo_worker::reactor::ReactorBridge<R>,
) -> AnyhowResult<()>
where
    R: gloo_worker::reactor::Reactor + 'static,
    <R as gloo_worker::reactor::Reactor>::Scope: ReactorScoped<Input = ImageBuf, Output = ImageBuf>,
{
    let image_data = read_frame(state)?;
    let image_buf = ImageBuf::new(image_data.width(), image_data.height(), image_data.data().0);
    reactor.send_input(image_buf);

    let out = reactor
        .next()
        .await
        .ok_or_else(|| anyhow!("no output"))?
        .to_image_data()?;

    write_frame(state, &out)
}
