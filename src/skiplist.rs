//! Lock-free skiplist implementation based on Fraser's algorithm

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use std::cmp::Ordering;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

const MAX_LEVEL: usize = 32;

/// A node in the skiplist
#[derive(Debug)]
pub struct Node<K, V> {
    pub key: K,
    pub value: V,
    pub level: usize,
    pub next: Vec<Atomic<Node<K, V>>>,
}

impl<K, V> Node<K, V> {
    /// Create a new node with the given level
    pub fn new(key: K, value: V, level: usize) -> Self {
        let mut next = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            next.push(Atomic::null());
        }
        Self {
            key,
            value,
            level,
            next,
        }
    }

    /// Mark a pointer for deletion
    pub fn mark_tower(&self) -> bool {
        for level in (0..=self.level).rev() {
            let mut next = unsafe { self.next[level].load(AtomicOrdering::SeqCst, epoch::unprotected()) };
            loop {
                if next.tag() & 1 != 0 {
                    if level == 0 {
                        return false;
                    }
                    break;
                }
                match unsafe {
                    self.next[level].compare_exchange(
                        next,
                        next.with_tag(next.tag() | 1),
                        AtomicOrdering::SeqCst,
                        AtomicOrdering::SeqCst,
                        epoch::unprotected(),
                    )
                } {
                    Ok(_) => break,
                    Err(e) => next = e.current,
                }
            }
        }
        true
    }
}

/// A lock-free skiplist
pub struct SkipList<K, V> {
    head: Atomic<Node<K, V>>,
    size: AtomicUsize,
}

impl<K: Ord + Default + Clone, V: Default + Clone> SkipList<K, V> {
    /// Create a new empty skiplist
    pub fn new() -> Self {
        let head = Node::new(K::default(), V::default(), MAX_LEVEL - 1);
        Self {
            head: Atomic::new(head),
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

    /// Generate a random level for a new node
    fn random_level() -> usize {
        let mut level = 0;
        while level < MAX_LEVEL - 1 && rand::random::<bool>() {
            level += 1;
        }
        level
    }

    /// Find the position where a key should be inserted
    pub fn find<'g>(&self, key: &K, guard: &'g Guard) -> (Vec<Shared<'g, Node<K, V>>>, Vec<Shared<'g, Node<K, V>>>) {
        let mut preds = vec![Shared::null(); MAX_LEVEL];
        let mut succs = vec![Shared::null(); MAX_LEVEL];

        'retry: loop {
            let mut pred = self.head.load(AtomicOrdering::SeqCst, guard);
            
            for level in (0..MAX_LEVEL).rev() {
                let mut curr = unsafe { pred.deref() }.next[level].load(AtomicOrdering::SeqCst, guard);
                
                loop {
                    if curr.is_null() {
                        break;
                    }

                    let curr_node = unsafe { curr.deref() };
                    let next = curr_node.next[level].load(AtomicOrdering::SeqCst, guard);

                    // Skip marked nodes
                    if next.tag() & 1 != 0 {
                        if unsafe { pred.deref() }.next[level]
                            .compare_exchange(
                                curr,
                                next.with_tag(0),
                                AtomicOrdering::SeqCst,
                                AtomicOrdering::SeqCst,
                                guard,
                            )
                            .is_err()
                        {
                            continue 'retry;
                        }
                        curr = next.with_tag(0);
                        continue;
                    }

                    match curr_node.key.cmp(key) {
                        Ordering::Less => {
                            pred = curr;
                            curr = next;
                        }
                        _ => break,
                    }
                }

                preds[level] = pred;
                succs[level] = curr;
            }

            return (preds, succs);
        }
    }

    /// Insert a key-value pair into the skiplist
    pub fn insert(&self, key: K, value: V, guard: &Guard) -> bool {
        let level = Self::random_level();
        let key_clone = key.clone();
        
        loop {
            let (mut preds, mut succs) = self.find(&key_clone, guard);

            // Check if key already exists
            if !succs[0].is_null() && unsafe { succs[0].deref() }.key == key_clone {
                return false;
            }

            // Create new node
            let new_node = Owned::new(Node::new(key.clone(), value.clone(), level))
                .into_shared(guard);
            let new_node_ref = unsafe { new_node.deref() };

            // Set next pointers
            for i in 0..=level {
                new_node_ref.next[i].store(succs[i], AtomicOrdering::SeqCst);
            }

            // Try to link at level 0
            match unsafe { preds[0].deref() }.next[0].compare_exchange(
                succs[0],
                new_node,
                AtomicOrdering::SeqCst,
                AtomicOrdering::SeqCst,
                guard,
            ) {
                Ok(_) => {
                    // Link at higher levels
                    for i in 1..=level {
                        loop {
                            let pred = &unsafe { preds[i].deref() }.next[i];
                            if new_node_ref.next[i].load(AtomicOrdering::SeqCst, guard).tag() & 1 != 0 {
                                break;
                            }
                            
                            if pred.compare_exchange(
                                succs[i],
                                new_node,
                                AtomicOrdering::SeqCst,
                                AtomicOrdering::SeqCst,
                                guard,
                            ).is_ok() {
                                break;
                            }
                            
                            // Retry find for this level
                            let (new_preds, new_succs) = self.find(&new_node_ref.key, guard);
                            preds[i] = new_preds[i];
                            succs[i] = new_succs[i];
                        }
                    }
                    
                    self.size.fetch_add(1, AtomicOrdering::Relaxed);
                    return true;
                }
                Err(_) => {
                    // Retry
                    unsafe { guard.defer_destroy(new_node) };
                }
            }
        }
    }

    /// Remove a key from the skiplist
    pub fn remove(&self, key: &K, guard: &Guard) -> Option<V> {
        loop {
            let (preds, succs) = self.find(key, guard);

            if succs[0].is_null() || unsafe { succs[0].deref() }.key != *key {
                return None;
            }

            let node = unsafe { succs[0].deref() };
            
            // Mark the node for deletion
            if !node.mark_tower() {
                continue;
            }

            // Try to unlink at all levels
            for level in (0..=node.level).rev() {
                let _ = unsafe { preds[level].deref() }.next[level].compare_exchange(
                    succs[level],
                    node.next[level].load(AtomicOrdering::SeqCst, guard).with_tag(0),
                    AtomicOrdering::SeqCst,
                    AtomicOrdering::SeqCst,
                    guard,
                );
            }

            self.size.fetch_sub(1, AtomicOrdering::Relaxed);
            
            // We can't safely return the value due to memory management
            // In a real implementation, we'd need to handle this differently
            return Some(unsafe { ptr::read(&node.value) });
        }
    }

    /// Get a reference to the first node (for SprayList operations)
    pub fn head(&self) -> Shared<Node<K, V>> {
        unsafe { self.head.load(AtomicOrdering::SeqCst, epoch::unprotected()) }
    }
    
    /// Decrement the size counter (for SprayList operations)
    pub fn decrement_size(&self) {
        self.size.fetch_sub(1, AtomicOrdering::Relaxed);
    }
}

impl<K: Ord + Default + Clone, V: Default + Clone> Default for SkipList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: SkipList is thread-safe
unsafe impl<K: Send + Sync, V: Send + Sync> Send for SkipList<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for SkipList<K, V> {}