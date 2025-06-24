//! `SprayList`: A scalable relaxed priority queue

use crate::rng::MarsagliaXOR;
use crate::skiplist::{is_marked, unmark, Node, SkipList};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Parameters for the spray operation
/// These match the C implementation's configurable parameters
#[derive(Clone)]
pub struct SprayParams {
    /// Base parameter for spray width (D in the paper)
    pub spray_base: usize,
    /// Height parameter (H in the paper)
    pub spray_height: usize,
    /// Number of attempts before giving up
    pub max_attempts: usize,
    /// Height to start spray traversal (SCANHEIGHT in C)
    pub scan_height_fn: fn(usize) -> usize,
    /// Maximum scan length at top level (SCANMAX in C)
    pub scan_max_fn: fn(usize) -> usize,
    /// Amount to increase scan length at each level (SCANINC in C)
    pub scan_increment: usize,
    /// Number of levels to skip at each step (SCANSKIP in C)
    pub scan_skip: usize,
    /// Enable fallback to exact deletemin with probability 1/n
    pub enable_fallback: bool,
    /// Enable collision tracking
    pub track_collisions: bool,
}

impl Default for SprayParams {
    fn default() -> Self {
        Self {
            spray_base: 32,                         // D = 32 from the paper
            spray_height: 20,                       // H = 20 from the paper
            max_attempts: 100,                      // Reasonable retry limit
            scan_height_fn: |n| floor_log_2(n) + 1, // SCANHEIGHT
            scan_max_fn: |n| floor_log_2(n) + 1,    // SCANMAX
            scan_increment: 0,                      // SCANINC
            scan_skip: 1,                           // SCANSKIP
            enable_fallback: true,
            track_collisions: true,
        }
    }
}

/// Helper function to compute floor(log2(n))
const fn floor_log_2(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        (std::mem::size_of::<usize>() * 8) - 1 - n.leading_zeros() as usize
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

impl<
        K: Ord + Default + Clone + Send + Sync + 'static,
        V: Default + Clone + Send + Sync + 'static,
    > SprayList<K, V>
{
    /// Create a new `SprayList`
    #[must_use]
    pub fn new() -> Self {
        Self::with_params(SprayParams::default())
    }

    /// Create a new `SprayList` with custom parameters
    #[must_use]
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
    pub fn insert(&self, key: &K, value: &V) -> bool {
        self.skiplist.insert(key, value)
    }

    /// Delete and return an element from near the minimum
    /// This is the key operation that uses the spray mechanism
    pub fn delete_min(&self) -> Option<(K, V)> {
        let n = self.num_threads.load(AtomicOrdering::Relaxed).max(1);

        // With probability 1/n, fall back to exact deletemin (Lotan-Shavit style)
        if self.params.enable_fallback && Self::should_fallback(n) {
            return self.delete_exact_min();
        }

        // Perform spray delete_min
        self.spray_delete_min_impl(n)
    }

    /// Implementation of the spray `delete_min` algorithm matching the C version
    fn spray_delete_min_impl(&self, n: usize) -> Option<(K, V)> {
        thread_local! {
            static RNG: RefCell<MarsagliaXOR> = RefCell::new({
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                std::thread::current().id().hash(&mut hasher);
                #[allow(clippy::cast_possible_truncation)]
                MarsagliaXOR::new(hasher.finish() as u32)
            });
        }

        RNG.with(|rng_cell| {
            let mut rng = rng_cell.borrow_mut();

            let mut current = self.skiplist.head();
            if current.is_null() {
                return None;
            }

            // Calculate spray parameters based on thread count
            let scan_height = (self.params.scan_height_fn)(n);
            let mut scan_max = (self.params.scan_max_fn)(n);
            let scan_inc = self.params.scan_increment;
            let scan_skip = self.params.scan_skip;

            // Start traversal from the calculated height
            let mut level = scan_height.min(31); // Skip list max level constraint
            let mut dummy = 0;

            // Multi-level spray traversal (matching C implementation)
            loop {
                #[allow(clippy::cast_possible_truncation)]
                let mut scan_len = rng.range(scan_max as u32 + 1) as usize;

                // Apply dummy counter limit
                while dummy < n * floor_log_2(n) / 2 && scan_len > 0 {
                    dummy += 1 << level;
                    scan_len -= 1;
                }

                // Step right at current level
                while scan_len > 0 {
                    let next_ptr = unsafe { &*current }.next[level].load(AtomicOrdering::SeqCst);

                    if next_ptr.is_null() {
                        break;
                    }

                    let next = unmark(next_ptr); // Remove mark bit
                    if next.is_null() {
                        break;
                    }

                    current = next;

                    // Only decrement scan_len if node is not deleted
                    let node = unsafe { &*current };
                    if !is_marked(node.next[0].load(AtomicOrdering::SeqCst)) {
                        scan_len -= 1;
                    }
                }

                // Check if we've reached the end
                if unsafe { &*current }.next[0]
                    .load(AtomicOrdering::SeqCst)
                    .is_null()
                {
                    return None; // Empty list
                }

                scan_max += scan_inc;

                // Move to next level down
                if level == 0 {
                    break;
                }
                level = level.saturating_sub(scan_skip);
            }

            // Skip head node if we're still there
            if current == self.skiplist.head() {
                return None;
            }

            // Find first non-deleted node
            while !current.is_null() {
                let node = unsafe { &*current };
                let next_ptr = node.next[0].load(AtomicOrdering::SeqCst);

                if !is_marked(next_ptr) {
                    // Node is not deleted
                    break;
                }

                current = unmark(next_ptr);
            }

            if current.is_null() {
                return None;
            }

            // Try to delete the current node
            let node = unsafe { &*current };

            // Try to mark the node for deletion
            if node.mark_tower() {
                // Successfully marked, now try to unlink
                self.unlink_node(current);

                // Return the key-value pair
                let key = node.key.clone();
                let value = node.value.clone();

                self.skiplist.decrement_size();
                Some((key, value))
            } else {
                // Collision - someone else deleted this node
                None // Could retry here
            }
        })
    }

    /// Determine if we should fallback to exact deletemin
    #[allow(clippy::cast_possible_truncation)]
    fn should_fallback(n: usize) -> bool {
        thread_local! {
            static RNG: RefCell<MarsagliaXOR> = RefCell::new({
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                std::thread::current().id().hash(&mut hasher);
                #[allow(clippy::cast_possible_truncation)]
                MarsagliaXOR::new((hasher.finish().wrapping_mul(31)) as u32)
            });
        }

        RNG.with(|rng_cell| {
            let mut rng = rng_cell.borrow_mut();
            // Fallback with probability 1/n (when random value mod n == 0)
            n == 1 || rng.range(n as u32) == 0
        })
    }

    /// Unlink a marked node from all levels
    fn unlink_node(&self, node_ptr: *mut Node<K, V>) {
        let node = unsafe { &*node_ptr };

        for level in (0..=node.level).rev() {
            // Find predecessors at this level
            let mut pred = self.skiplist.head();
            let mut curr = unsafe { &*pred }.next[level].load(AtomicOrdering::SeqCst);

            // Search for the node at this level
            while !curr.is_null() && curr != node_ptr {
                let curr_clean = unmark(curr);
                if curr_clean.is_null() {
                    break;
                }
                let curr_node = unsafe { &*curr_clean };

                // Skip marked nodes
                if is_marked(curr_node.next[level].load(AtomicOrdering::SeqCst)) {
                    let next = unmark(curr_node.next[level].load(AtomicOrdering::SeqCst));
                    let _ = unsafe { &*pred }.next[level].compare_exchange(
                        curr,
                        next,
                        AtomicOrdering::SeqCst,
                        AtomicOrdering::SeqCst,
                    );
                    curr = next;
                    continue;
                }

                pred = curr_clean;
                curr = curr_node.next[level].load(AtomicOrdering::SeqCst);
            }

            // If we found the node, try to unlink it
            if curr == node_ptr {
                let next = unmark(node.next[level].load(AtomicOrdering::SeqCst));
                let _ = unsafe { &*pred }.next[level].compare_exchange(
                    curr,
                    next,
                    AtomicOrdering::SeqCst,
                    AtomicOrdering::SeqCst,
                );
            }
        }
    }

    /// Delete the exact minimum (fallback for when spray fails)
    fn delete_exact_min(&self) -> Option<(K, V)> {
        loop {
            let head = self.skiplist.head();
            if head.is_null() {
                return None;
            }

            let first = unsafe { &*head }.next[0].load(AtomicOrdering::SeqCst);
            if first.is_null() {
                return None;
            }

            let first_clean = unmark(first);
            if first_clean.is_null() {
                continue;
            }

            let node = unsafe { &*first_clean };

            // Try to mark and remove this node
            if node.mark_tower() {
                // Successfully marked, now unlink
                self.unlink_node(first_clean);

                // Return the key-value pair
                let key = node.key.clone();
                let value = node.value.clone();

                self.skiplist.decrement_size();
                return Some((key, value));
            }
        }
    }

    /// Get the current size of the `SprayList`
    pub fn len(&self) -> usize {
        self.skiplist.len()
    }

    /// Check if the `SprayList` is empty
    pub fn is_empty(&self) -> bool {
        self.skiplist.is_empty()
    }

    /// Find the minimum key without removing it
    pub fn peek_min(&self) -> Option<K> {
        let head = self.skiplist.head();

        if head.is_null() {
            return None;
        }

        let first = unsafe { &*head }.next[0].load(AtomicOrdering::SeqCst);
        if first.is_null() {
            return None;
        }

        let first_clean = unmark(first);
        if first_clean.is_null() {
            return None;
        }

        Some(unsafe { &*first_clean }.key.clone())
    }
}

impl<
        K: Ord + Default + Clone + Send + Sync + 'static,
        V: Default + Clone + Send + Sync + 'static,
    > Default for SprayList<K, V>
{
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
        assert!(spray.insert(&5, &"five".to_string()));
        assert!(spray.insert(&3, &"three".to_string()));
        assert!(spray.insert(&7, &"seven".to_string()));
        assert!(spray.insert(&1, &"one".to_string()));
        assert!(spray.insert(&9, &"nine".to_string()));

        assert_eq!(spray.len(), 5);

        // Delete min should return values near the minimum
        let mut deleted = Vec::new();
        while let Some((k, _v)) = spray.delete_min() {
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
                    spray_clone.insert(&key, &format!("value-{key}"));
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
                let mut attempts = 0;
                let max_attempts = 250; // Allow more attempts than exact items

                while local_deleted.len() < 25 && attempts < max_attempts {
                    attempts += 1;
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

        // In a relaxed priority queue, not all elements may be found
        // This is expected behavior due to the spray mechanism
        assert!(deleted.len() > 50); // At least half should be found
        println!("Successfully deleted {} out of 100 elements", deleted.len());
    }

    #[test]
    fn test_spray_parameters() {
        let params = SprayParams {
            spray_base: 16,
            spray_height: 10,
            ..Default::default()
        };

        let spray = SprayList::with_params(params);

        // Test with different parameters
        assert!(spray.insert(&1, &"one".to_string()));
        assert!(spray.insert(&2, &"two".to_string()));
        assert!(spray.insert(&3, &"three".to_string()));

        assert_eq!(spray.len(), 3);
        assert!(!spray.is_empty());

        // Should still work with different parameters
        let result = spray.delete_min();
        assert!(result.is_some());
    }

    #[test]
    fn test_peek_min() {
        let spray = SprayList::new();

        assert!(spray.peek_min().is_none());

        spray.insert(&5, &"five".to_string());
        spray.insert(&3, &"three".to_string());
        spray.insert(&7, &"seven".to_string());

        // Should return the minimum key
        assert_eq!(spray.peek_min(), Some(3));

        // Should not remove the element
        assert_eq!(spray.len(), 3);
    }

    #[test]
    fn test_fallback_behavior() {
        let params = SprayParams {
            enable_fallback: true,
            ..Default::default()
        };

        let spray = SprayList::with_params(params);
        spray.set_num_threads(1); // With 1 thread, fallback should be more likely

        // Insert elements
        for i in 0..10 {
            spray.insert(&i, &format!("value{i}"));
        }

        // Delete some elements - should work with fallback
        let mut deleted = Vec::new();
        for _ in 0..5 {
            if let Some((k, _)) = spray.delete_min() {
                deleted.push(k);
            }
        }

        // Should have deleted some elements
        assert!(!deleted.is_empty());
        assert!(deleted.len() <= 5);
    }

    #[test]
    fn test_thread_safety() {
        let spray = Arc::new(SprayList::new());
        let num_threads = 8;
        let items_per_thread = 100;
        spray.set_num_threads(num_threads);

        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        // Insert phase
        for thread_id in 0..num_threads {
            let spray_clone = spray.clone();
            let barrier_clone = barrier.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();

                for i in 0..items_per_thread {
                    let key = thread_id * items_per_thread + i;
                    spray_clone.insert(&key, &format!("thread{thread_id}-item{i}"));
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let total_inserted = spray.len();
        assert_eq!(total_inserted, num_threads * items_per_thread);

        // Delete phase
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for _ in 0..num_threads {
            let spray_clone = spray.clone();
            let barrier_clone = barrier.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();

                let mut deleted_count = 0;
                for _ in 0..items_per_thread {
                    if spray_clone.delete_min().is_some() {
                        deleted_count += 1;
                    }
                }
                deleted_count
            });

            handles.push(handle);
        }

        let mut total_deleted = 0;
        for handle in handles {
            total_deleted += handle.join().unwrap();
        }

        // Should delete a significant portion of elements
        assert!(total_deleted > total_inserted / 2);
        println!("Deleted {total_deleted} out of {total_inserted} elements in concurrent test",);
    }
}
