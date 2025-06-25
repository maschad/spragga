use spragga::SprayList;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn main() {
    println!("SprayList Example");

    // Create a new SprayList
    let spray = Arc::new(SprayList::new());
    spray.set_num_threads(4);

    // Sequential example
    println!("\n1. Sequential operations:");
    let seq_spray = SprayList::new();
    for i in &[10, 3, 15, 7, 1, 20, 5] {
        seq_spray.insert(i, &format!("value-{i}"));
    }

    println!("Inserted values: 10, 3, 15, 7, 1, 20, 5");
    println!("Size: {}", seq_spray.len());

    print!("Deleting elements (may not be in exact order due to relaxed semantics): ");
    while let Some((k, _v)) = seq_spray.delete_min() {
        print!("{k} ");
    }
    println!();

    // Concurrent example
    println!("\n2. Concurrent operations:");
    let start = Instant::now();
    let mut handles = vec![];

    // Spawn threads to insert values
    for i in 0..4 {
        let spray_clone = spray.clone();
        let handle = thread::spawn(move || {
            for j in 0..250 {
                spray_clone.insert(&(i * 1000 + j), &format!("thread-{i}-value-{j}"));
            }
        });
        handles.push(handle);
    }

    // Wait for insertions to complete
    for handle in handles {
        handle.join().unwrap();
    }

    println!("Inserted 1000 elements from 4 threads");
    println!("Size: {}", spray.len());

    // Spawn threads to delete values
    let mut handles = vec![];
    for _ in 0..4 {
        let spray_clone = spray.clone();
        let handle = thread::spawn(move || {
            let mut count = 0;
            for _ in 0..250 {
                if spray_clone.delete_min().is_some() {
                    count += 1;
                }
            }
            count
        });
        handles.push(handle);
    }

    // Wait for deletions to complete
    let mut total_deleted = 0;
    for handle in handles {
        total_deleted += handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    println!("Deleted {total_deleted} elements from 4 threads");
    println!("Final size: {}", spray.len());
    println!("Time elapsed: {elapsed:?}");

    println!("\nNote: SprayList provides relaxed ordering semantics for better scalability.");
    println!("Elements may not be deleted in strict ascending order.");
}
