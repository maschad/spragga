use spragga::SprayList;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

/// Configuration for throughput testing
#[derive(Clone)]
struct TestConfig {
    num_threads: usize,
    duration_secs: u64,
    update_percentage: usize,
    initial_size: usize,
    total_ops: usize,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            duration_secs: 5,
            update_percentage: 100,
            initial_size: 1000,
            total_ops: 1_000_000,
        }
    }
}

/// Run throughput test and return operations per second
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::cast_precision_loss)]
fn run_throughput_test(config: TestConfig) -> f64 {
    let spray = Arc::new(SprayList::<i32, String>::new());
    spray.set_num_threads(config.num_threads);

    // Pre-populate with initial elements
    for i in 0..config.initial_size {
        #[allow(clippy::cast_possible_wrap)]
        #[allow(clippy::cast_possible_truncation)]
        spray.insert(&(i as i32), &format!("initial-{i}"));
    }

    let barrier = Arc::new(Barrier::new(config.num_threads));
    let total_ops = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    let start_time = Instant::now();

    // Spawn worker threads
    for thread_id in 0..config.num_threads {
        let spray_clone = spray.clone();
        let barrier_clone = barrier.clone();
        let total_ops_clone = total_ops.clone();
        let thread_config = config.clone();

        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start

            let mut rng = spragga::rng::MarsagliaXOR::new(
                u32::try_from(thread_id).unwrap().wrapping_mul(31) + 1,
            );
            let thread_start = Instant::now();
            let ops_per_thread = thread_config.total_ops / thread_config.num_threads;
            let mut local_ops = 0;

            while local_ops < ops_per_thread
                && thread_start.elapsed().as_secs() < thread_config.duration_secs
            {
                let is_update =
                    rng.range(100) < u32::try_from(thread_config.update_percentage).unwrap();

                if is_update {
                    // Update operations: insert or delete
                    if rng.range(2) == 0 {
                        // Insert operation
                        #[allow(clippy::cast_possible_wrap)]
                        let key = i32::try_from(thread_id * 1_000_000 + local_ops).unwrap();
                        spray_clone.insert(&key, &format!("thread{thread_id}-op{local_ops}"));
                    } else {
                        // Delete operation
                        let _ = spray_clone.delete_min();
                    }
                } else {
                    // Read operation
                    let _ = spray_clone.peek_min();
                }

                total_ops_clone.fetch_add(1, Ordering::Relaxed);
                local_ops += 1;
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start_time.elapsed();
    let ops_completed = total_ops.load(Ordering::Relaxed);
    ops_completed as f64 / duration.as_secs_f64()
}

fn print_usage() {
    println!("Usage: throughput_test [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --threads <N>         Number of threads (default: 4)");
    println!("  --duration <N>        Duration in seconds (default: 5)");
    println!("  --update-pct <N>      Update percentage 0-100 (default: 100)");
    println!("  --initial-size <N>    Initial size (default: 1000)");
    println!("  --total-ops <N>       Total operations (default: 1000000)");
    println!("  --csv                 Output CSV format");
    println!("  --scaling             Run scaling test (1,2,4,8,16 threads)");
    println!("  --help                Show this help");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = TestConfig::default();
    let mut csv_output = false;
    let mut scaling_test = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                if i + 1 < args.len() {
                    config.num_threads = args[i + 1].parse().unwrap_or(4);
                    i += 1;
                }
            }
            "--duration" => {
                if i + 1 < args.len() {
                    config.duration_secs = args[i + 1].parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--update-pct" => {
                if i + 1 < args.len() {
                    config.update_percentage = args[i + 1].parse().unwrap_or(100);
                    i += 1;
                }
            }
            "--initial-size" => {
                if i + 1 < args.len() {
                    config.initial_size = args[i + 1].parse().unwrap_or(1000);
                    i += 1;
                }
            }
            "--total-ops" => {
                if i + 1 < args.len() {
                    config.total_ops = args[i + 1].parse().unwrap_or(1_000_000);
                    i += 1;
                }
            }
            "--csv" => csv_output = true,
            "--scaling" => scaling_test = true,
            "--help" => {
                print_usage();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    if scaling_test {
        // Run scaling test similar to run_throughput.sh
        let thread_counts = vec![1, 2, 4, 8, 16];

        if csv_output {
            println!("#procs,spray_throughput");
            for &num_threads in &thread_counts {
                let mut test_config = config.clone();
                test_config.num_threads = num_threads;
                let throughput = run_throughput_test(test_config);
                println!("{throughput:.0},{num_threads}");
            }
        } else {
            println!("SprayList Scaling Test");
            println!("======================");
            println!();
            println!("Threads\tThroughput (ops/sec)");
            println!("-------\t-------------------");

            for &num_threads in &thread_counts {
                let mut test_config = config.clone();
                test_config.num_threads = num_threads;
                let throughput = run_throughput_test(test_config);
                println!("{num_threads}\t{throughput:.0}");
            }
        }
    } else {
        // Single test run
        let throughput = run_throughput_test(config.clone());

        if csv_output {
            println!("{throughput:.0}");
        } else {
            println!("SprayList Throughput Test");
            println!("=========================");
            println!();
            println!("Configuration:");
            println!("  Threads: {}", config.num_threads);
            println!("  Duration: {} seconds", config.duration_secs);
            println!("  Update percentage: {}%", config.update_percentage);
            println!("  Initial size: {}", config.initial_size);
            println!("  Target operations: {}", config.total_ops);
            println!();
            println!("Results:");
            println!("  Throughput: {throughput:.0} operations/second");
        }
    }
}
