//! SprayList: A Scalable Relaxed Priority Queue
//!
//! This crate provides an implementation of the SprayList data structure,
//! a scalable concurrent priority queue with relaxed ordering semantics.

pub mod skiplist;
pub mod spraylist;

pub use spraylist::SprayList;