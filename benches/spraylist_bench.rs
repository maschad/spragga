#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::semicolon_if_nothing_returned)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::semicolon_if_nothing_returned)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use spray::{SprayList, SprayParams};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

fn sequential_insert_delete(c: &mut Criterion) {
    c.bench_function("sequential insert/delete 1000", |b| {
        b.iter(|| {
            let spray = SprayList::new();

            // Insert 1000 elements
            for i in 0..1000 {
                spray.insert(&i, &i);
            }

            // Delete all elements
            while spray.delete_min().is_some() {}
        });
    });
}

/// Benchmark configuration similar to the C test
#[derive(Clone)]
struct BenchConfig {
    /// Number of threads
    num_threads: usize,
    /// Duration in seconds
    duration_secs: u64,
    /// Update percentage (0-100)
    update_percentage: usize,
    /// Initial size of the data structure
    initial_size: usize,
    /// Total operations target
    total_ops: usize,
}

impl Default for BenchConfig {
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

/// Performance metrics collection
#[derive(Default, Debug)]
struct PerfMetrics {
    total_ops: usize,
    insert_ops: usize,
    delete_ops: usize,
    peek_ops: usize,
    successful_ops: usize,
    #[allow(dead_code)]
    failed_ops: usize,
    duration_nanos: u64,
}

impl PerfMetrics {
    fn throughput(&self) -> f64 {
        self.total_ops as f64 / (self.duration_nanos as f64 / 1_000_000_000.0)
    }

    fn success_rate(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.successful_ops as f64 / self.total_ops as f64 * 100.0
        }
    }
}

/// Run throughput benchmark similar to the C implementation
#[allow(clippy::needless_pass_by_value)]
fn run_throughput_benchmark(config: BenchConfig) -> PerfMetrics {
    let spray = Arc::new(SprayList::<i32, String>::new());
    spray.set_num_threads(config.num_threads);

    // Pre-populate with initial elements
    for i in 0..config.initial_size {
        spray.insert(&i32::try_from(i).unwrap(), &format!("initial-{i}"));
    }

    let barrier = Arc::new(Barrier::new(config.num_threads));
    let total_ops = Arc::new(AtomicUsize::new(0));
    let insert_ops = Arc::new(AtomicUsize::new(0));
    let delete_ops = Arc::new(AtomicUsize::new(0));
    let peek_ops = Arc::new(AtomicUsize::new(0));
    let successful_ops = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    let start_time = Instant::now();

    // Spawn worker threads
    for thread_id in 0..config.num_threads {
        let spray_clone = spray.clone();
        let barrier_clone = barrier.clone();
        let total_ops_clone = total_ops.clone();
        let insert_ops_clone = insert_ops.clone();
        let delete_ops_clone = delete_ops.clone();
        let peek_ops_clone = peek_ops.clone();
        let successful_ops_clone = successful_ops.clone();
        let thread_config = config.clone();

        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start

            let mut rng = spray::rng::MarsagliaXOR::new((thread_id as u32).wrapping_mul(31) + 1);
            let thread_start = Instant::now();
            let ops_per_thread = thread_config.total_ops / thread_config.num_threads;
            let mut local_ops = 0;

            while local_ops < ops_per_thread
                && thread_start.elapsed().as_secs() < thread_config.duration_secs
            {
                let is_update = rng.range(100) < thread_config.update_percentage as u32;
                let success = if is_update {
                    // Update operations: insert or delete
                    if rng.range(2) == 0 {
                        // Insert operation
                        let key = (thread_id * 1_000_000 + local_ops) as i32;
                        let result =
                            spray_clone.insert(&key, &format!("thread{thread_id}-op{local_ops}"));
                        insert_ops_clone.fetch_add(1, Ordering::Relaxed);
                        result
                    } else {
                        // Delete operation
                        let result = spray_clone.delete_min().is_some();
                        delete_ops_clone.fetch_add(1, Ordering::Relaxed);
                        result
                    }
                } else {
                    // Read operation
                    let result = spray_clone.peek_min().is_some();
                    peek_ops_clone.fetch_add(1, Ordering::Relaxed);
                    result
                };

                if success {
                    successful_ops_clone.fetch_add(1, Ordering::Relaxed);
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

    PerfMetrics {
        total_ops: total_ops.load(Ordering::Relaxed),
        insert_ops: insert_ops.load(Ordering::Relaxed),
        delete_ops: delete_ops.load(Ordering::Relaxed),
        peek_ops: peek_ops.load(Ordering::Relaxed),
        successful_ops: successful_ops.load(Ordering::Relaxed),
        failed_ops: total_ops.load(Ordering::Relaxed) - successful_ops.load(Ordering::Relaxed),
        duration_nanos: u64::try_from(duration.as_nanos()).unwrap(),
    }
}

/// Benchmark throughput with different thread counts (similar to run_`throughput.sh`)
fn bench_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Thread counts similar to the reference script
    let thread_counts = vec![1, 2, 4, 8, 16];

    for &num_threads in &thread_counts {
        let config = BenchConfig {
            num_threads,
            duration_secs: 3,
            total_ops: 1_000_000,
            update_percentage: 100,
            initial_size: 1000,
        };

        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &config,
            |b, config| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::new(0, 0);
                    let mut total_throughput = 0.0;

                    for _ in 0..iters {
                        let start = Instant::now();
                        let metrics = run_throughput_benchmark(config.clone());
                        total_time += start.elapsed();
                        total_throughput += metrics.throughput();

                        // Print detailed metrics
                        eprintln!(
                            "Threads: {}, Ops: {}, Throughput: {:.0} ops/sec, Success: {:.1}%",
                            config.num_threads,
                            metrics.total_ops,
                            metrics.throughput(),
                            metrics.success_rate()
                        );
                    }

                    eprintln!(
                        "Average throughput for {} threads: {:.0} ops/sec",
                        config.num_threads,
                        total_throughput / iters as f64
                    );

                    total_time
                })
            },
        );
    }

    group.finish();
}

/// Benchmark with different update percentages
fn bench_mixed_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workloads");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let update_percentages = vec![0, 25, 50, 75, 100];
    let num_threads = 8;

    for &update_pct in &update_percentages {
        let config = BenchConfig {
            num_threads,
            duration_secs: 3,
            total_ops: 500_000,
            update_percentage: update_pct,
            initial_size: 5000,
        };

        group.bench_with_input(
            BenchmarkId::new("update_pct", update_pct),
            &config,
            |b, config| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::new(0, 0);

                    for _ in 0..iters {
                        let start = Instant::now();
                        let metrics = run_throughput_benchmark(config.clone());
                        total_time += start.elapsed();

                        eprintln!("Update %: {}, Throughput: {:.0} ops/sec, Insert: {}, Delete: {}, Peek: {}",
                                 config.update_percentage, metrics.throughput(),
                                 metrics.insert_ops, metrics.delete_ops, metrics.peek_ops);
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different `SprayList` configurations
fn bench_spray_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("spray_parameters");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    let configs = vec![
        ("default", SprayParams::default()),
        (
            "small_spray",
            SprayParams {
                spray_base: 16,
                spray_height: 10,
                ..SprayParams::default()
            },
        ),
        (
            "large_spray",
            SprayParams {
                spray_base: 64,
                spray_height: 30,
                ..SprayParams::default()
            },
        ),
        (
            "no_fallback",
            SprayParams {
                enable_fallback: false,
                ..SprayParams::default()
            },
        ),
    ];

    for (name, params) in configs {
        let params_clone = params.clone();
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::new(0, 0);

                for _ in 0..iters {
                    let spray =
                        Arc::new(SprayList::<i32, String>::with_params(params_clone.clone()));
                    spray.set_num_threads(8);

                    let start = Instant::now();

                    // Quick mixed workload test
                    let config = BenchConfig {
                        num_threads: 8,
                        duration_secs: 2,
                        total_ops: 100_000,
                        update_percentage: 100,
                        initial_size: 1000,
                    };

                    let metrics = run_throughput_benchmark(config);
                    total_time += start.elapsed();

                    eprintln!("{}: {:.0} ops/sec", name, metrics.throughput());
                }

                total_time
            });
        });
    }

    group.finish();
}

fn concurrent_operations(c: &mut Criterion) {
    let thread_counts = vec![1, 2, 4, 8];

    for num_threads in thread_counts {
        c.bench_function(&format!("concurrent {num_threads} threads"), |b| {
            b.iter(|| {
                let spray = Arc::new(SprayList::new());
                spray.set_num_threads(num_threads);

                let elements_per_thread = 1000 / num_threads;
                let mut handles = vec![];

                // Insert phase
                for t in 0..num_threads {
                    let spray_clone = spray.clone();
                    let handle = thread::spawn(move || {
                        for i in 0..elements_per_thread {
                            spray_clone.insert(&(t * 1000 + i), &i);
                        }
                    });
                    handles.push(handle);
                }

                for handle in handles {
                    handle.join().unwrap();
                }

                // Delete phase with timeout to prevent hanging
                let mut handles = vec![];
                for _ in 0..num_threads {
                    let spray_clone = spray.clone();
                    let handle = thread::spawn(move || {
                        let mut deleted = 0;
                        let mut attempts = 0;
                        let max_attempts = elements_per_thread * 10; // Allow more attempts

                        while deleted < elements_per_thread && attempts < max_attempts {
                            if spray_clone.delete_min().is_some() {
                                deleted += 1;
                            }
                            attempts += 1;
                        }
                    });
                    handles.push(handle);
                }

                for handle in handles {
                    handle.join().unwrap();
                }
            });
        });
    }
}

criterion_group!(
    benches,
    sequential_insert_delete,
    concurrent_operations,
    bench_throughput_scaling,
    bench_mixed_workloads,
    bench_spray_parameters
);
criterion_main!(benches);
