use crate::{canvas::RenderState, effects};
use leptos::{logging::*, prelude::*};
use leptos_use::{use_user_media, UseUserMediaReturn};

#[component]
pub fn App() -> impl IntoView {
    let video_ref = NodeRef::<leptos::html::Video>::new();
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let temporary_canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let UseUserMediaReturn { stream, start, .. } = use_user_media();
    start();

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
        let state = RenderState {
            video: video_ref,
            canvas: canvas_ref,
            temporary_canvas: temporary_canvas_ref,
        };
        effects::setup_frame_listener(state);
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
