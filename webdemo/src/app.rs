use crate::{canvas::RenderState, on_video_play, reactor::ModelReactor};
use gloo_worker::Spawnable;
use leptos::{logging::*, prelude::*};
use leptos_use::{use_user_media, UseUserMediaReturn};
use wasm_bindgen::{prelude::Closure, JsCast};
use web_sys::Event;

#[component]
pub fn App() -> impl IntoView {
    let video_ref = NodeRef::<leptos::html::Video>::new();
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let temporary_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let UseUserMediaReturn { stream, start, .. } = use_user_media();
    start();

    let frame_listener = Closure::<dyn Fn(Event)>::wrap(Box::new(move |_: Event| {
        log!("entered frame listener");
        let reactor = ModelReactor::spawner().spawn("./worker.js");
        let state = RenderState {
            video: video_ref,
            canvas: canvas_ref,
            temporary_canvas: temporary_canvas_ref,
        };
        on_video_play(state, reactor);
    }));

    Effect::new(move |_| {
        let Some(stream) = stream.get() else {
            return;
        };
        let stream = match stream {
            Ok(stream) => stream,
            Err(e) => {
                error!("failed to get media stream: {:?}", e);
                return;
            }
        };
        let Some(video) = video_ref.get() else {
            log!("video is not ready yeat");
            return;
        };
        video.set_src_object(Some(&stream))
    });

    Effect::new(move |_| {
        let Some(video) = video_ref.get() else {
            return;
        };
        let _ = video
            .add_event_listener_with_callback("play", frame_listener.as_ref().unchecked_ref())
            .inspect_err(|e| {
                error!("failed to attach frame listener: {:?}", e);
            });
    });

    view! {
      <div class="dashboard">
        <div class="panel">
          <video
            class="frame"
            node_ref=video_ref
            controls=false
            autoplay=true
            playsinline=true
            muted=true
            hidden=false
          ></video>
        </div>
        <div class="panel">
          <canvas class="frame" node_ref=canvas_ref></canvas>
        </div>
      </div>
      <canvas node_ref=temporary_canvas_ref style="display: none;"></canvas>
    }
}
