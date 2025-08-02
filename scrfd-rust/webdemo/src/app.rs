use crate::{on_frame, process_image_with_model};
use leptos::{logging::*, prelude::*};
use leptos_use::{use_user_media, UseUserMediaReturn};
use wasm_bindgen::{prelude::Closure, JsCast};
use web_sys::Event;

const MODEL_BYTES: &[u8] = include_bytes!("../../../models/scrfd.onnx");

fn load_model() -> scrfd::Scrfd {
    let model = scrfd::Scrfd::from_bytes(MODEL_BYTES).expect("valid scrfd");
    log!("parsed model");
    model
}

#[component]
pub fn App() -> impl IntoView {
    let video_ref = NodeRef::<leptos::html::Video>::new();
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let UseUserMediaReturn { stream, start, .. } = use_user_media();
    start();

    let frame_listener = Closure::<dyn Fn(Event)>::wrap(Box::new(move |_: Event| {
        log!("entered frame listener");
        let model = load_model();
        on_frame(canvas_ref, video_ref, move |img| {
            process_image_with_model(&model, img)
        });
    }));

    Effect::new(move |_| {
        video_ref.get().map(|video| {
            match stream.get() {
                Some(Ok(stream)) => video.set_src_object(Some(&stream)),
                Some(Err(e)) => error!("failed to get media stream: {:?}", e),
                None => log!("no stream yet"),
            };

            let _ = video
                .add_event_listener_with_callback("play", frame_listener.as_ref().unchecked_ref())
                .inspect_err(|e| {
                    error!("failed to attach frame listener: {:?}", e);
                });
        })
    });

    view! {
        <canvas node_ref=canvas_ref></canvas>
        <video node_ref=video_ref controls=false autoplay=true muted=true hidden=false></video>
    }
}
