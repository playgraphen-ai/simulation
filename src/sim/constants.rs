//! Centralized constants for the simulation to prevent desynchronization between Rust and shaders.

pub const MAX_PEOPLE: u32 = 524288;
pub const MAX_BUILDINGS: u32 = 524288;
pub const MAX_SEGMENTS: u32 = 65536;

pub const MAX_PATH_REQUESTS: u32 = 131072;
pub const MAX_PATH_LEN: u32 = 512;

/// Default minimum size for GPU buffers like occupancy or building stats.
pub const MIN_GPU_BUFFER_CAPACITY: u32 = 65536;
