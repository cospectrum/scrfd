use crate::{canvas::RenderState, on_video_play, reactor::ModelReactor};
use gloo_worker::Spawnable;
use leptos::{logging::*, prelude::*};
use wasm_bindgen::{prelude::Closure, JsCast};
use web_sys::Event;

pub fn setup_frame_listener(state: RenderState) {
    let Some(video) = state.video.get() else {
        return;
    };

    let frame_listener = Closure::<dyn Fn(Event)>::wrap(Box::new(move |_: Event| {
        log!("entered frame listener");
        let reactor = ModelReactor::spawner().spawn("./worker.js");
        on_video_play(state, reactor);
    }));

    let _ = video
        .add_event_listener_with_callback("play", frame_listener.as_ref().unchecked_ref())
        .inspect_err(|e| {
            error!("failed to attach frame listener: {:?}", e);
        });

    frame_listener.forget();
}
