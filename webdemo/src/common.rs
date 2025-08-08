pub fn get_num_workers(num_cpus: i32) -> i32 {
    assert!(num_cpus > 0);
    let num_workers = if num_cpus == 1 { 1 } else { num_cpus - 1 };
    assert!(num_workers > 0);
    num_workers
}
