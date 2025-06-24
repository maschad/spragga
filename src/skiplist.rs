//! Lock-free skiplist implementation using std library atomics
//! Based on Fraser's algorithm with pointer marking for node deletion

use std::cmp::Ordering;
use std::ptr::{self};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering as AtomicOrdering};

const MAX_LEVEL: usize = 32;

/// Type alias for the result of find operation
type FindResult<K, V> = (Vec<*mut Node<K, V>>, Vec<*mut Node<K, V>>);

/// A node in the skiplist
#[derive(Debug)]
pub struct Node<K, V> {
    pub key: K,
    pub value: V,
    pub level: usize,
    pub next: Vec<AtomicPtr<Node<K, V>>>,
}

impl<K, V> Node<K, V> {
    /// Create a new node with the given level
    pub fn new(key: K, value: V, level: usize) -> Self {
        let mut next = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            next.push(AtomicPtr::new(ptr::null_mut()));
        }
        Self {
            key,
            value,
            level,
            next,
        }
    }

    /// Mark a pointer for deletion by setting the least significant bit
    pub fn mark_tower(&self) -> bool {
        for level in (0..=self.level).rev() {
            let mut next = self.next[level].load(AtomicOrdering::SeqCst);
            loop {
                // Check if already marked (LSB = 1)
                if (next as usize) & 1 != 0 {
                    if level == 0 {
                        return false; // Already marked
                    }
                    break;
                }
                // Try to mark by setting LSB
                let marked = ((next as usize) | 1) as *mut Self;
                match self.next[level].compare_exchange(
                    next,
                    marked,
                    AtomicOrdering::SeqCst,
                    AtomicOrdering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(current) => next = current,
                }
            }
        }
        true
    }
}

/// Helper functions for pointer marking
#[inline]
pub fn is_marked<K, V>(ptr: *mut Node<K, V>) -> bool {
    (ptr as usize) & 1 != 0
}

#[inline]
pub fn unmark<K, V>(ptr: *mut Node<K, V>) -> *mut Node<K, V> {
    ((ptr as usize) & !1) as *mut Node<K, V>
}

#[inline]
pub fn mark<K, V>(ptr: *mut Node<K, V>) -> *mut Node<K, V> {
    ((ptr as usize) | 1) as *mut Node<K, V>
}

/// A lock-free skiplist
pub struct SkipList<K, V> {
    head: AtomicPtr<Node<K, V>>,
    size: AtomicUsize,
}

impl<K: Ord + Default + Clone, V: Default + Clone> SkipList<K, V> {
    /// Create a new empty skiplist
    #[must_use]
    pub fn new() -> Self {
        let head = Box::into_raw(Box::new(Node::new(
            K::default(),
            V::default(),
            MAX_LEVEL - 1,
        )));
        Self {
            head: AtomicPtr::new(head),
            size: AtomicUsize::new(0),
        }
    }

    /// Get the current size of the skiplist
    pub fn len(&self) -> usize {
        self.size.load(AtomicOrdering::Relaxed)
    }

    /// Check if the skiplist is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generate a random level for a new node using our own RNG
    fn random_level() -> usize {
        use crate::rng::MarsagliaXOR;
        thread_local! {
            static RNG: std::cell::RefCell<MarsagliaXOR> = std::cell::RefCell::new({
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                std::thread::current().id().hash(&mut hasher);
                #[allow(clippy::cast_possible_truncation)]
                MarsagliaXOR::new((hasher.finish() as u32).wrapping_mul(17))
            });
        }

        RNG.with(|rng_cell| {
            let mut rng = rng_cell.borrow_mut();
            let mut level = 0;
            while level < MAX_LEVEL - 1 && (rng.next() & 1) == 1 {
                level += 1;
            }
            level
        })
    }

    /// Find the position where a key should be inserted (Fraser's search)
    pub fn find(&self, key: &K) -> FindResult<K, V> {
        let mut preds = vec![ptr::null_mut(); MAX_LEVEL];
        let mut succs = vec![ptr::null_mut(); MAX_LEVEL];

        'retry: loop {
            let mut pred = self.head.load(AtomicOrdering::SeqCst);

            for level in (0..MAX_LEVEL).rev() {
                let mut curr = unsafe { &*pred }.next[level].load(AtomicOrdering::SeqCst);

                loop {
                    if curr.is_null() {
                        break;
                    }

                    // Remove mark bit for comparison
                    let curr_clean = unmark(curr);
                    if curr_clean.is_null() {
                        break;
                    }

                    let curr_node = unsafe { &*curr_clean };
                    let next = curr_node.next[level].load(AtomicOrdering::SeqCst);

                    // Skip marked nodes
                    if is_marked(next) {
                        let next_clean = unmark(next);
                        if unsafe { &*pred }.next[level]
                            .compare_exchange(
                                curr,
                                next_clean,
                                AtomicOrdering::SeqCst,
                                AtomicOrdering::SeqCst,
                            )
                            .is_err()
                        {
                            continue 'retry;
                        }
                        curr = next_clean;
                        continue;
                    }

                    match curr_node.key.cmp(key) {
                        Ordering::Less => {
                            pred = curr_clean;
                            curr = next;
                        }
                        _ => break,
                    }
                }

                preds[level] = pred;
                succs[level] = unmark(curr);
            }

            return (preds, succs);
        }
    }

    /// Insert a key-value pair into the skiplist
    pub fn insert(&self, key: &K, value: &V) -> bool {
        let level = Self::random_level();

        loop {
            let (preds, succs) = self.find(key);

            // Check if key already exists
            if !succs[0].is_null() && unsafe { &*succs[0] }.key == *key {
                return false;
            }

            // Create new node
            let new_node = Box::into_raw(Box::new(Node::new(key.clone(), value.clone(), level)));
            let new_node_ref = unsafe { &*new_node };

            // Set next pointers
            for (i, succ) in succs.iter().enumerate().take(level + 1) {
                new_node_ref.next[i].store(*succ, AtomicOrdering::SeqCst);
            }

            // Try to link at level 0 first
            match unsafe { &*preds[0] }.next[0].compare_exchange(
                succs[0],
                new_node,
                AtomicOrdering::SeqCst,
                AtomicOrdering::SeqCst,
            ) {
                Ok(_) => {
                    // Successfully linked at level 0, now link at higher levels
                    for i in 1..=level {
                        loop {
                            let pred = &unsafe { &*preds[i] }.next[i];
                            let curr_next = new_node_ref.next[i].load(AtomicOrdering::SeqCst);

                            // If our node got marked, give up on higher levels
                            if is_marked(curr_next) {
                                break;
                            }

                            if pred
                                .compare_exchange(
                                    succs[i],
                                    new_node,
                                    AtomicOrdering::SeqCst,
                                    AtomicOrdering::SeqCst,
                                )
                                .is_ok()
                            {
                                break;
                            }

                            // Retry find for this level
                            let (new_preds, new_succs) = self.find(&new_node_ref.key);
                            if i < new_preds.len() && i < new_succs.len() {
                                // Update for this specific level and retry
                                new_node_ref.next[i].store(new_succs[i], AtomicOrdering::SeqCst);
                                if new_preds[i] != preds[i] {
                                    break; // Topology changed, stop trying higher levels
                                }
                            } else {
                                break;
                            }
                        }
                    }

                    self.size.fetch_add(1, AtomicOrdering::Relaxed);
                    return true;
                }
                Err(_) => {
                    // Clean up and retry
                    unsafe { drop(Box::from_raw(new_node)) };
                }
            }
        }
    }

    /// Remove a key from the skiplist
    pub fn remove(&self, key: &K) -> Option<V> {
        loop {
            let (preds, succs) = self.find(key);

            if succs[0].is_null() || unsafe { &*succs[0] }.key != *key {
                return None;
            }

            let node = unsafe { &*succs[0] };

            // Mark the node for deletion
            if !node.mark_tower() {
                continue;
            }

            // Try to unlink at all levels
            for level in (0..=node.level).rev() {
                let next = node.next[level].load(AtomicOrdering::SeqCst);
                let next_clean = unmark(next);

                let _ = unsafe { &*preds[level] }.next[level].compare_exchange(
                    succs[level],
                    next_clean,
                    AtomicOrdering::SeqCst,
                    AtomicOrdering::SeqCst,
                );
            }

            self.size.fetch_sub(1, AtomicOrdering::Relaxed);

            // We can't safely return the value due to potential concurrent access
            // In a real implementation, we'd need hazard pointers or epoch-based reclamation
            return Some(node.value.clone());
        }
    }

    /// Get a reference to the head node (for `SprayList` operations)
    pub fn head(&self) -> *mut Node<K, V> {
        self.head.load(AtomicOrdering::SeqCst)
    }

    /// Decrement the size counter (for `SprayList` operations)
    pub fn decrement_size(&self) {
        self.size.fetch_sub(1, AtomicOrdering::Relaxed);
    }
}

impl<K: Ord + Default + Clone, V: Default + Clone> Default for SkipList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// Memory management note: In a production implementation, we would need
// proper hazard pointers or epoch-based reclamation to safely reclaim memory.
// For this educational implementation, we're accepting potential memory leaks
// to focus on the SprayList algorithm itself.

// Safety: SkipList is thread-safe through atomic operations
unsafe impl<K: Send + Sync, V: Send + Sync> Send for SkipList<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for SkipList<K, V> {}
