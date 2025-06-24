# SprayList

A Rust implementation of the SprayList data structure - a scalable concurrent priority queue with relaxed ordering semantics.

## Overview

SprayList is a concurrent data structure designed to address the scalability bottleneck in traditional priority queues. Based on the paper ["The SprayList: A Scalable Relaxed Priority Queue"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SprayList_full.pdf) by Alistarh et al., it provides high-performance concurrent access at the cost of relaxed ordering guarantees.

### Key Features

- **Scalable**: Avoids sequential bottlenecks through "spraying" technique
- **Lock-free**: Built on top of a non-blocking skip list
- **Relaxed ordering**: DeleteMin returns an element among the first O(p log³ p) elements
- **Thread-safe**: Supports concurrent insert and delete operations

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
spray = "0.1.0"
```

### Basic Example

```rust
use spray::SprayList;

fn main() {
    // Create a new SprayList
    let spray = SprayList::new();
    
    // Insert elements
    spray.insert(5, "five");
    spray.insert(3, "three");
    spray.insert(7, "seven");
    spray.insert(1, "one");
    
    // Delete minimum (may not be the exact minimum due to relaxed semantics)
    while let Some((key, value)) = spray.delete_min() {
        println!("Removed: {} -> {}", key, value);
    }
}
```

### Concurrent Example

```rust
use spray::SprayList;
use std::sync::Arc;
use std::thread;

fn main() {
    let spray = Arc::new(SprayList::new());
    spray.set_num_threads(4); // Optimize for 4 threads
    
    let mut handles = vec![];
    
    // Spawn threads to insert values
    for i in 0..4 {
        let spray_clone = spray.clone();
        let handle = thread::spawn(move || {
            for j in 0..100 {
                spray_clone.insert(i * 100 + j, format!("value-{}", j));
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Inserted {} elements", spray.len());
}
```

## API

### `SprayList::new() -> Self`
Creates a new empty SprayList with default parameters.

### `SprayList::with_params(params: SprayParams) -> Self`
Creates a new SprayList with custom spray parameters.

### `insert(&self, key: K, value: V) -> bool`
Inserts a key-value pair. Returns `true` if successful, `false` if key already exists.

### `delete_min(&self) -> Option<(K, V)>`
Removes and returns an element from near the minimum of the list. Due to the relaxed semantics, this may not be the exact minimum.

### `len(&self) -> usize`
Returns the number of elements in the list.

### `is_empty(&self) -> bool`
Returns true if the list is empty.

### `peek_min(&self) -> Option<K>`
Returns the minimum key without removing it.

### `set_num_threads(&self, num_threads: usize)`
Sets the expected number of threads for parameter tuning.

## Implementation Details

The SprayList is built on top of a lock-free skip list and uses a "spray" operation for the DeleteMin functionality:

1. **Random Walk**: Instead of always removing the minimum element, DeleteMin performs a random walk starting from the head of the list.

2. **Spray Width**: The maximum distance of the random walk is O(p log³ p) where p is the number of threads.

3. **Collision Avoidance**: Multiple threads performing DeleteMin are likely to delete different elements, reducing contention.

## Performance Characteristics

- **Insert**: O(log n) expected time
- **DeleteMin**: O(log³ p) expected time, independent of list size
- **Space**: O(n) expected space

The relaxed ordering allows the data structure to scale much better than traditional priority queues under high contention.

## Benchmarking

This implementation includes comprehensive benchmarks to measure SprayList performance and compare with the original research.

### Quick Start

Run all benchmarks:
```bash
./run_throughput.sh
```

Generate CSV data for analysis:
```bash
./run_throughput_csv.sh
```

### Benchmark Types

#### 1. Throughput Scaling Tests
Tests performance with different thread counts (1, 2, 4, 8, 16 threads):

```bash
cargo bench --bench spraylist_bench -- bench_throughput_scaling
```

**Sample Output:**
```
Threads: 1, Ops: 995234, Throughput: 331745 ops/sec, Success: 100.0%
Threads: 4, Ops: 987654, Throughput: 1234567 ops/sec, Success: 98.5%
Threads: 8, Ops: 976543, Throughput: 1987654 ops/sec, Success: 97.2%
```

#### 2. Mixed Workload Tests
Tests different ratios of insert/delete vs. read operations:

```bash
cargo bench --bench spraylist_bench -- bench_mixed_workloads
```

Tests update percentages: 0% (read-only), 25%, 50%, 75%, 100% (write-only)

#### 3. Parameter Comparison Tests
Compares different SprayList configurations:

```bash
cargo bench --bench spraylist_bench -- bench_spray_parameters
```

Tests configurations:
- **default**: Standard parameters (spray_base=32, spray_height=20)
- **small_spray**: Reduced spray width (spray_base=16, spray_height=10)
- **large_spray**: Increased spray width (spray_base=64, spray_height=30)
- **no_fallback**: Disabled exact deletemin fallback

### Custom Throughput Testing

Use the standalone throughput test binary for custom configurations:

```bash
# Build the test binary
cargo build --release --bin throughput_test

# Basic usage
./target/release/throughput_test --threads 8 --duration 10 --update-pct 100

# Scaling test with CSV output
./target/release/throughput_test --scaling --csv

# Show all options
./target/release/throughput_test --help
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--threads <N>` | Number of threads | 4 |
| `--duration <N>` | Test duration in seconds | 5 |
| `--update-pct <N>` | Update percentage (0-100) | 100 |
| `--initial-size <N>` | Initial data structure size | 1000 |
| `--total-ops <N>` | Target total operations | 1,000,000 |
| `--csv` | Output in CSV format | false |
| `--scaling` | Run thread scaling test | false |

### Understanding Results

#### Throughput Metrics
- **ops/sec**: Operations per second (higher is better)
- **Success Rate**: Percentage of successful operations
- **Insert/Delete/Peek counts**: Breakdown of operation types

#### Expected Performance Characteristics
- **Single Thread**: Baseline performance, limited by sequential execution
- **Low Thread Count (2-4)**: Near-linear scaling due to reduced contention
- **High Thread Count (8+)**: Sublinear scaling as contention increases, but still maintains good throughput due to spray mechanism

#### Interpreting Success Rates
- **100% Success**: All operations completed successfully
- **95-99% Success**: Normal for high-contention scenarios, indicates effective collision avoidance
- **<90% Success**: May indicate excessive contention or configuration issues

### Comparison with Reference Implementation

The benchmarks are designed to match the methodology from the [original SprayList research](https://github.com/jkopinsky/SprayList):

- Similar thread counts (1, 2, 4, 8, 16)
- Comparable operation mixes and durations
- CSV output format for data analysis
- Performance metrics aligned with the paper's evaluation

### Benchmark Results Analysis

To analyze benchmark results:

1. **Criterion HTML Reports**: Available in `target/criterion/*/report/index.html`
2. **CSV Data**: Use `--csv` flag for data processing with external tools
3. **Raw Data**: Located in `target/criterion/` directory

## Testing

The project includes comprehensive tests to verify correctness and thread safety:

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test test_concurrent_operations
cargo test test_thread_safety
cargo test test_spray_parameters
```

### Test Categories

- **Sequential Tests**: Basic functionality with single-threaded operations
- **Concurrent Tests**: Multi-threaded correctness validation
- **Parameter Tests**: Custom SprayList configuration validation
- **Thread Safety Tests**: High-contention stress testing with multiple threads
- **Fallback Tests**: Validation of exact deletemin fallback behavior

The test suite includes stress tests with up to 800 concurrent operations across 8 threads to ensure robust behavior under high contention.

## Limitations

- Elements returned by DeleteMin may not be in strict ascending order
- The ordering relaxation is bounded by O(p log³ p) where p is the number of threads
- Best suited for applications that can tolerate approximate priority ordering

## References

- [The SprayList Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SprayList_full.pdf)
- [Original C Implementation](https://github.com/jkopinsky/SprayList)

## License

This project is licensed under the MIT License.