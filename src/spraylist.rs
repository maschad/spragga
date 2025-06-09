//! SprayList: A scalable relaxed priority queue

use crate::skiplist::{Node, SkipList};
use crossbeam_epoch::{self as epoch, Guard, Shared};
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Parameters for the spray operation
pub struct SprayParams {
    /// Base parameter for spray width (D in the paper)
    pub spray_base: usize,
    /// Height parameter (H in the paper)
    pub spray_height: usize,
    /// Number of attempts before giving up
    pub max_attempts: usize,
}

impl Default for SprayParams {
    fn default() -> Self {
        Self {
            spray_base: 32,     // D = 32 from the paper
            spray_height: 20,   // H = 20 from the paper
            max_attempts: 100,  // Reasonable retry limit
        }
    }
}

/// A scalable relaxed priority queue
pub struct SprayList<K, V> {
    /// The underlying skiplist
    skiplist: Arc<SkipList<K, V>>,
    /// Parameters for the spray operation
    params: SprayParams,
    /// Approximate number of threads (for scaling)
    num_threads: AtomicUsize,
}

impl<K: Ord + Default + Clone + Send + Sync + 'static, V: Default + Clone + Send + Sync + 'static> SprayList<K, V> {
    /// Create a new SprayList
    pub fn new() -> Self {
        Self::with_params(SprayParams::default())
    }

    /// Create a new SprayList with custom parameters
    pub fn with_params(params: SprayParams) -> Self {
        Self {
            skiplist: Arc::new(SkipList::new()),
            params,
            num_threads: AtomicUsize::new(1),
        }
    }

    /// Set the number of threads (for parameter tuning)
    pub fn set_num_threads(&self, num_threads: usize) {
        self.num_threads.store(num_threads, AtomicOrdering::Relaxed);
    }

    /// Insert a key-value pair
    pub fn insert(&self, key: K, value: V) -> bool {
        let guard = &epoch::pin();
        self.skiplist.insert(key, value, guard)
    }

    /// Delete and return an element from near the minimum
    /// This is the key operation that uses the spray mechanism
    pub fn delete_min(&self) -> Option<(K, V)> {
        let guard = &epoch::pin();
        let mut rng = rand::thread_rng();
        
        let p = self.num_threads.load(AtomicOrdering::Relaxed).max(1);
        let spray_width = self.calculate_spray_width(p);

        for _ in 0..self.params.max_attempts {
            // Start from the head of the skiplist
            let mut current = self.skiplist.head();
            if current.is_null() {
                return None;
            }

            // Skip the sentinel node
            current = unsafe { current.deref() }.next[0].load(AtomicOrdering::SeqCst, guard);
            if current.is_null() {
                return None;
            }

            // Perform random walk with spray width
            let steps = self.random_walk_steps(spray_width, &mut rng);
            
            for _ in 0..steps {
                let next = unsafe { current.deref() }.next[0].load(AtomicOrdering::SeqCst, guard);
                
                // Check if we've reached the end or a marked node
                if next.is_null() || next.tag() & 1 != 0 {
                    break;
                }
                
                current = next;
            }

            // Try to delete the current node
            if !current.is_null() {
                let node = unsafe { current.deref() };
                
                // Try to mark and remove this node
                if node.mark_tower() {
                    // Successfully marked, now try to unlink
                    self.unlink_node(current, guard);
                    
                    // Return the key-value pair
                    // Note: In a real implementation, we'd need better memory management
                    let key = unsafe { std::ptr::read(&node.key) };
                    let value = unsafe { std::ptr::read(&node.value) };
                    
                    self.skiplist.decrement_size();
                    return Some((key, value));
                }
            }
        }

        // If we couldn't delete using spray, fall back to exact minimum
        self.delete_exact_min(guard)
    }

    /// Calculate the spray width based on number of threads
    fn calculate_spray_width(&self, p: usize) -> usize {
        // From the paper: spray width is O(p log^3 p)
        let log_p = (p as f64).log2().max(1.0) as usize;
        p * log_p.pow(3)
    }

    /// Generate the number of steps for random walk
    fn random_walk_steps(&self, spray_width: usize, rng: &mut impl Rng) -> usize {
        // Use geometric distribution with parameter based on spray width
        let mut steps = 0;
        let prob = 1.0 / (self.params.spray_base as f64);
        
        while steps < spray_width && rng.gen::<f64>() > prob {
            steps += 1;
        }
        
        steps.min(spray_width)
    }

    /// Unlink a marked node from all levels
    fn unlink_node(&self, node_ptr: Shared<Node<K, V>>, guard: &Guard) {
        let node = unsafe { node_ptr.deref() };
        
        for level in (0..=node.level).rev() {
            // Find predecessors at this level
            let mut pred = self.skiplist.head();
            let mut curr = unsafe { pred.deref() }.next[level].load(AtomicOrdering::SeqCst, guard);
            
            // Search for the node at this level
            while !curr.is_null() && curr != node_ptr {
                let curr_node = unsafe { curr.deref() };
                
                // Skip marked nodes
                if curr_node.next[level].load(AtomicOrdering::SeqCst, guard).tag() & 1 != 0 {
                    let next = curr_node.next[level].load(AtomicOrdering::SeqCst, guard).with_tag(0);
                    let _ = unsafe { pred.deref() }.next[level].compare_exchange(
                        curr,
                        next,
                        AtomicOrdering::SeqCst,
                        AtomicOrdering::SeqCst,
                        guard,
                    );
                    curr = next;
                    continue;
                }
                
                pred = curr;
                curr = curr_node.next[level].load(AtomicOrdering::SeqCst, guard);
            }
            
            // If we found the node, try to unlink it
            if curr == node_ptr {
                let next = node.next[level].load(AtomicOrdering::SeqCst, guard).with_tag(0);
                let _ = unsafe { pred.deref() }.next[level].compare_exchange(
                    curr,
                    next,
                    AtomicOrdering::SeqCst,
                    AtomicOrdering::SeqCst,
                    guard,
                );
            }
        }
    }

    /// Delete the exact minimum (fallback for when spray fails)
    fn delete_exact_min(&self, guard: &Guard) -> Option<(K, V)> {
        loop {
            let head = self.skiplist.head();
            if head.is_null() {
                return None;
            }

            let first = unsafe { head.deref() }.next[0].load(AtomicOrdering::SeqCst, guard);
            if first.is_null() {
                return None;
            }

            let node = unsafe { first.deref() };
            
            // Try to mark and remove this node
            if node.mark_tower() {
                // Successfully marked, now unlink
                self.unlink_node(first, guard);
                
                // Return the key-value pair
                let key = unsafe { std::ptr::read(&node.key) };
                let value = unsafe { std::ptr::read(&node.value) };
                
                self.skiplist.decrement_size();
                return Some((key, value));
            }
        }
    }

    /// Get the current size of the SprayList
    pub fn len(&self) -> usize {
        self.skiplist.len()
    }

    /// Check if the SprayList is empty
    pub fn is_empty(&self) -> bool {
        self.skiplist.is_empty()
    }

    /// Find the minimum key without removing it
    pub fn peek_min(&self) -> Option<K> {
        let guard = &epoch::pin();
        let head = self.skiplist.head();
        
        if head.is_null() {
            return None;
        }

        let first = unsafe { head.deref() }.next[0].load(AtomicOrdering::SeqCst, guard);
        if first.is_null() {
            return None;
        }

        Some(unsafe { first.deref() }.key.clone())
    }
}

impl<K: Ord + Default + Clone + Send + Sync + 'static, V: Default + Clone + Send + Sync + 'static> Default for SprayList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: SprayList is thread-safe
unsafe impl<K: Send + Sync, V: Send + Sync> Send for SprayList<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for SprayList<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn test_sequential_operations() {
        let spray = SprayList::new();
        
        // Insert some values
        assert!(spray.insert(5, "five"));
        assert!(spray.insert(3, "three"));
        assert!(spray.insert(7, "seven"));
        assert!(spray.insert(1, "one"));
        assert!(spray.insert(9, "nine"));
        
        assert_eq!(spray.len(), 5);
        
        // Delete min should return values near the minimum
        let mut deleted = Vec::new();
        while let Some((k, v)) = spray.delete_min() {
            deleted.push(k);
        }
        
        assert_eq!(deleted.len(), 5);
        assert!(spray.is_empty());
        
        // The first few elements should be small
        assert!(deleted[0] <= 5); // Relaxed ordering
    }

    #[test]
    fn test_concurrent_operations() {
        let spray = Arc::new(SprayList::new());
        spray.set_num_threads(4);
        
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];
        
        // Spawn threads to insert values
        for i in 0..4 {
            let spray_clone = spray.clone();
            let barrier_clone = barrier.clone();
            
            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                for j in 0..25 {
                    let key = i * 100 + j;
                    spray_clone.insert(key, format!("value-{}", key));
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all insertions to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(spray.len(), 100);
        
        // Now delete all elements
        let mut deleted = BTreeSet::new();
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];
        
        for _ in 0..4 {
            let spray_clone = spray.clone();
            let barrier_clone = barrier.clone();
            
            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                let mut local_deleted = Vec::new();
                for _ in 0..25 {
                    if let Some((k, _)) = spray_clone.delete_min() {
                        local_deleted.push(k);
                    }
                }
                local_deleted
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            let local = handle.join().unwrap();
            for k in local {
                deleted.insert(k);
            }
        }
        
        assert_eq!(deleted.len(), 100);
        assert!(spray.is_empty());
    }
}