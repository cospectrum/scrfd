use crate::imgproc::on_frame;
use leptos::{logging::*, prelude::*};
use leptos_use::{use_user_media, UseUserMediaReturn};
use wasm_bindgen::{prelude::Closure, JsCast};
use web_sys::Event;

#[component]
pub fn App() -> impl IntoView {
    let video_ref = NodeRef::<leptos::html::Video>::new();
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let UseUserMediaReturn { stream, start, .. } = use_user_media();

    let frame_listener = Closure::<dyn Fn(Event)>::wrap(Box::new(move |_: Event| {
        log!("entered frame listener");
        on_frame(canvas_ref, video_ref).unwrap();
    }));

    start();

    Effect::new(move |_| {
        video_ref.get().map(|video| {
            match stream.get() {
                Some(Ok(stream)) => video.set_src_object(Some(&stream)),
                Some(Err(e)) => error!("Failed to get media stream: {:?}", e),
                None => log!("No stream yet"),
            };

            let _ = video
                .add_event_listener_with_callback("play", frame_listener.as_ref().unchecked_ref())
                .inspect_err(|e| {
                    error!("failed to attach frame listener: {:?}", e);
                });
        })
    });

    view! {
        <video node_ref=video_ref controls=false autoplay=true muted=true hidden=true></video>
        <canvas node_ref=canvas_ref></canvas>
    }
}
