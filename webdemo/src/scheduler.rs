use crate::{canvas::ImageBuf, common};

use anyhow::{anyhow, bail, Result as AnyhowResult};
use futures::{
    sink::SinkExt,
    stream::{SplitSink, SplitStream},
    StreamExt,
};
use gloo_worker::{
    reactor::{reactor, ReactorBridge, ReactorScope},
    Spawnable,
};
use leptos::logging::{error, log};
use wasm_bindgen::JsCast;
use web_sys::{js_sys, WorkerGlobalScope};

type Scope = ReactorScope<ImageBuf, ImageBuf>;
type SchedulerStream = SplitStream<Scope>;
type SchedulerSink = SplitSink<Scope, ImageBuf>;

type WorkerReactor = ReactorBridge<crate::Worker>;
type WorkerSink = SplitSink<WorkerReactor, ImageBuf>;
type WorkerStream = SplitStream<WorkerReactor>;

#[reactor]
pub async fn Scheduler(scope: Scope) {
    log!("entered scheduler reactor");
    if let Err(e) = start_scheduler(scope).await {
        error!("scheduler failed: {:?}", e);
    }
    log!("scheduler finished");
}

/// Don't forget to replace it with the `start_scheduler` before git merge.
async fn __debug_start_scheduler(mut scope: Scope) -> AnyhowResult<()> {
    log!("starting scheduler loop");
    loop {
        let Some(input) = scope.next().await else {
            log!("no input for scheduler, returning");
            return Ok(());
        };
        scope.send(input).await?;
    }
}

async fn start_scheduler(scope: Scope) -> AnyhowResult<()> {
    log!("started scheduler");

    let num_cpus = get_num_cpus()?;
    assert!(num_cpus > 0);
    let num_workers = common::get_num_workers(num_cpus);
    assert!(num_workers > 0);
    log!("scheduler sees num_cpus: {:?}", num_cpus);

    let (worker_sinks, worker_streams) = (0..num_workers)
        .map(|_| crate::Worker::spawner().spawn("./worker.js"))
        .map(|worker| worker.split())
        .collect::<(Vec<_>, Vec<_>)>();

    log!("spawned workers");

    let (sink, stream) = scope.split();

    let (publihed, consumed) = futures::join!(
        start_publisher(stream, worker_sinks),
        start_consumer(sink, worker_streams)
    );
    match (publihed, consumed) {
        (Err(err), Ok(_)) => bail!("publisher failed: {:?}", err),
        (Ok(_), Err(err)) => bail!("consumer failed: {:?}", err),
        (Err(publish_err), Err(consume_err)) => {
            bail!(
                "publisher and consumer failed, errors respectfully: {:?}; {:?}",
                publish_err,
                consume_err
            );
        }
        _ => Ok(()),
    }
}

async fn start_consumer(
    mut sink: SchedulerSink,
    worker_streams: Vec<WorkerStream>,
) -> AnyhowResult<()> {
    log!("scheduler: start_consumer");

    let mut stream = futures::stream::select_all(worker_streams);
    loop {
        let Some(frame) = stream.next().await else {
            return Ok(());
        };
        sink.send(frame).await?;
    }
}

async fn start_publisher(
    mut stream: SchedulerStream,
    mut worker_sinks: Vec<WorkerSink>,
) -> AnyhowResult<()> {
    log!("scheduler: start_publisher");
    let mut frame_counter = 0usize;
    loop {
        let Some(input) = stream.next().await else {
            log!("no input for scheduler, returning");
            return Ok(());
        };
        let worker_num = frame_counter % worker_sinks.len();
        frame_counter += 1;

        worker_sinks[worker_num].send(input).await?;
    }
}

fn get_num_cpus() -> AnyhowResult<i32> {
    let navigator = js_sys::global()
        .dyn_into::<WorkerGlobalScope>()
        .map_err(|e| anyhow!("failed to get worker scope: {:?}", e))?
        .navigator();

    let num_cpus = navigator.hardware_concurrency() as i32;
    if num_cpus <= 0 {
        return Ok(1);
    }
    Ok(num_cpus)
}
