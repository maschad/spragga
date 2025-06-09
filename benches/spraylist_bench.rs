use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spray::SprayList;
use std::sync::Arc;
use std::thread;

fn sequential_insert_delete(c: &mut Criterion) {
    c.bench_function("sequential insert/delete 1000", |b| {
        b.iter(|| {
            let spray = SprayList::new();
            
            // Insert 1000 elements
            for i in 0..1000 {
                spray.insert(black_box(i), i);
            }
            
            // Delete all elements
            while spray.delete_min().is_some() {}
        });
    });
}

fn concurrent_operations(c: &mut Criterion) {
    let thread_counts = vec![1, 2, 4, 8];
    
    for num_threads in thread_counts {
        c.bench_function(&format!("concurrent {} threads", num_threads), |b| {
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
                            spray_clone.insert(t * 1000 + i, i);
                        }
                    });
                    handles.push(handle);
                }
                
                for handle in handles {
                    handle.join().unwrap();
                }
                
                // Delete phase
                let mut handles = vec![];
                for _ in 0..num_threads {
                    let spray_clone = spray.clone();
                    let handle = thread::spawn(move || {
                        for _ in 0..elements_per_thread {
                            spray_clone.delete_min();
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

criterion_group!(benches, sequential_insert_delete, concurrent_operations);
criterion_main!(benches);