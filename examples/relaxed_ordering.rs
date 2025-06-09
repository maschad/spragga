use spray::SprayList;
use std::collections::HashMap;

fn main() {
    println!("SprayList Relaxed Ordering Demonstration\n");
    
    // Run multiple trials to show the relaxed nature
    let trials = 10;
    let elements = 50;
    
    println!("Running {} trials with {} elements each...\n", trials, elements);
    
    let mut position_stats: HashMap<i32, Vec<usize>> = HashMap::new();
    
    for trial in 0..trials {
        let spray = SprayList::new();
        
        // Insert elements 1 to 50
        for i in 1..=elements {
            spray.insert(i, format!("value-{}", i));
        }
        
        // Delete all elements and record their positions
        let mut position = 0;
        while let Some((key, _)) = spray.delete_min() {
            position_stats.entry(key).or_insert(Vec::new()).push(position);
            position += 1;
        }
    }
    
    // Analyze results
    println!("Key | Average Position | Min Position | Max Position");
    println!("----|-----------------|--------------|-------------");
    
    for key in 1..=10 { // Show first 10 elements
        if let Some(positions) = position_stats.get(&key) {
            let avg = positions.iter().sum::<usize>() as f64 / positions.len() as f64;
            let min = *positions.iter().min().unwrap();
            let max = *positions.iter().max().unwrap();
            
            println!("{:3} | {:15.2} | {:12} | {:12}", key, avg, min, max);
        }
    }
    
    println!("\nObservations:");
    println!("- Smaller keys tend to be deleted earlier, but not always");
    println!("- The 'spray' mechanism introduces controlled randomness");
    println!("- This relaxation allows for better scalability in concurrent scenarios");
    
    // Demonstrate with different thread counts
    println!("\nEffect of thread count on spray width:");
    let spray = SprayList::new();
    
    for threads in [1, 2, 4, 8, 16] {
        spray.set_num_threads(threads);
        
        // Insert some elements
        for i in 1..=100 {
            spray.insert(i, i);
        }
        
        // Delete a few and see what we get
        let mut first_few = Vec::new();
        for _ in 0..5 {
            if let Some((k, _)) = spray.delete_min() {
                first_few.push(k);
            }
        }
        
        println!("Threads: {:2} - First 5 deletions: {:?}", threads, first_few);
        
        // Clear for next iteration
        while spray.delete_min().is_some() {}
    }
}