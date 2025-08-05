use gloo_worker::Registrable;
use webdemo::ModelReactor;

fn main() {
    console_error_panic_hook::set_once();

    ModelReactor::registrar().register();
}
