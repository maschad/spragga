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

## Limitations

- Elements returned by DeleteMin may not be in strict ascending order
- The ordering relaxation is bounded by O(p log³ p) where p is the number of threads
- Best suited for applications that can tolerate approximate priority ordering

## References

- [The SprayList Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SprayList_full.pdf)
- [Original C Implementation](https://github.com/jkopinsky/SprayList)

## License

This project is licensed under the MIT License.