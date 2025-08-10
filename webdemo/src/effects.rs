use crate::{canvas::RenderState, on_video_play};
use leptos::{logging::*, prelude::*};
use wasm_bindgen::{prelude::Closure, JsCast, JsValue};
use web_sys::{Event, MediaStream};

pub fn setup_video_stream(
    video_ref: NodeRef<leptos::html::Video>,
    stream: &Signal<Option<Result<MediaStream, JsValue>>, LocalStorage>,
) {
    let Some(stream) = stream.get() else {
        log!("setup_video_stream: stream is not ready yet");
        return;
    };
    let Some(video) = video_ref.get() else {
        log!("setup_video_stream: video is not ready yet");
        return;
    };

    let stream = match stream {
        Ok(stream) => stream,
        Err(e) => {
            error!("failed to get media stream: {:?}", e);
            return;
        }
    };
    video.set_src_object(Some(&stream))
}

pub fn setup_frame_listener(state: RenderState) {
    let Some(video) = state.video.get() else {
        log!("setup_frame_listener: video is not ready yet");
        return;
    };

    let frame_listener = Closure::<dyn Fn(Event)>::wrap(Box::new(move |_: Event| {
        log!("entered frame listener");
        on_video_play(state);
    }));

    let _ = video
        .add_event_listener_with_callback("play", frame_listener.as_ref().unchecked_ref())
        .inspect_err(|e| {
            error!("failed to attach frame listener: {:?}", e);
        });

    frame_listener.forget();
}
