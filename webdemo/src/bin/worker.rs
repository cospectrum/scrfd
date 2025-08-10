use gloo_worker::Registrable;

fn main() {
    console_error_panic_hook::set_once();

    webdemo::Worker::registrar().register();
}
