//! Road segments as an f32 texture, plus adjacency info used by pathfinding.
//!
//! Each segment = 2 RGBA32F texels (8 floats).
//! Texel 0: [end_a.x, end_a.y, end_b.x, end_b.y]   // tile coordinates
//! Texel 1: [speed_mean, conn_a_count_and_ids_packed, conn_b_count_and_ids_packed, length]
//!
//! conn_X packing: we pack up to 3 neighbour segment ids by encoding count in
//! the high bits of a single f32: `count * 2^24 + id0 * 2^16 + id1 * 2^8 + id2`.
//! (Neighbour ids fit in 24 bits which is plenty for up to ~16M segments, but
//! we stay safe as long as total segments < 256; a separate SSBO could replace
//! this later when the map scales up.)
//!
//! The CPU mirror also keeps a plain `Vec<RoadSegment>` so Rust-side systems
//! (UI, pathfinding warmup, counters) can iterate without touching GPU memory.

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

pub const ROAD_CAPACITY: u32 = 8192;
pub const TEXELS_PER_SEGMENT: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RoadRow {
    // Texel 0: Endpoints
    pub ax: f32,
    pub ay: f32,
    pub bx: f32,
    pub by: f32,
    // Texel 1: Stats & Meta
    pub speed_mean: f32,
    pub length: f32,
    pub count_a: f32,
    pub count_b: f32,
    // Texel 2: Connections A
    pub conn_a0: f32,
    pub conn_a1: f32,
    pub conn_a2: f32,
    pub _pad_a: f32,
    // Texel 3: Connections B
    pub conn_b0: f32,
    pub conn_b1: f32,
    pub conn_b2: f32,
    pub _pad_b: f32,
}

#[derive(Clone, Debug, Default)]
pub struct RoadSegment {
    pub a: (u32, u32),
    pub b: (u32, u32),
    /// Up to 3 neighbour segment ids at end A.
    pub conn_a: Vec<u32>,
    /// Up to 3 neighbour segment ids at end B.
    pub conn_b: Vec<u32>,
    pub speed_mean: f32,
    pub length: f32,
}

#[derive(Resource)]
pub struct RoadData {
    pub segments: Vec<RoadSegment>,
    pub rows: Vec<RoadRow>,
    pub tex_width: u32,
    pub tex_height: u32,
    pub dirty: bool,
    /// Usage counter per segment (incremented whenever a person crosses it).
    /// Used by the itinerary cache to detect hot segments.
    pub usage: Vec<u32>,
}

impl Default for RoadData {
    fn default() -> Self {
        let total_texels = ROAD_CAPACITY * TEXELS_PER_SEGMENT;
        let side = (total_texels as f32).sqrt().ceil() as u32;
        // Align width to TEXELS_PER_SEGMENT to ensure rows don't split segments
        let width = ((side + TEXELS_PER_SEGMENT - 1) / TEXELS_PER_SEGMENT) * TEXELS_PER_SEGMENT;
        let height = (total_texels + width - 1) / width;
        Self {
            segments: Vec::new(),
            rows: vec![RoadRow::default(); ROAD_CAPACITY as usize],
            tex_width: width,
            tex_height: height,
            dirty: true,
            usage: Vec::new(),
        }
    }
}

impl RoadData {
    /// Add a segment between two adjacent tiles. Returns the segment id.
    /// Auto-wires neighbour connections up to 3 per end.
    pub fn push_segment(&mut self, a: (u32, u32), b: (u32, u32)) -> Option<u32> {
        if self.segments.len() as u32 >= ROAD_CAPACITY {
            return None;
        }
        let length = {
            let dx = a.0 as f32 - b.0 as f32;
            let dy = a.1 as f32 - b.1 as f32;
            (dx * dx + dy * dy).sqrt().max(1.0)
        };
        let id = self.segments.len() as u32;
        // Auto-connect with any existing segment sharing an endpoint.
        let mut conn_a = Vec::new();
        let mut conn_b = Vec::new();
        let mut modified_others = Vec::new();
        for (other_id, other) in self.segments.iter_mut().enumerate() {
            let oid = other_id as u32;
            let mut modified = false;
            if other.a == a || other.b == a {
                if conn_a.len() < 3 { conn_a.push(oid); }
                // Mirror: the other segment also gains us as neighbour.
                if other.a == a && other.conn_a.len() < 3 { other.conn_a.push(id); modified = true; }
                if other.b == a && other.conn_b.len() < 3 { other.conn_b.push(id); modified = true; }
            }
            if other.a == b || other.b == b {
                if conn_b.len() < 3 { conn_b.push(oid); }
                if other.a == b && other.conn_a.len() < 3 { other.conn_a.push(id); modified = true; }
                if other.b == b && other.conn_b.len() < 3 { other.conn_b.push(id); modified = true; }
            }
            if modified {
                modified_others.push(oid);
            }
        }
        self.segments.push(RoadSegment {
            a, b, conn_a, conn_b,
            speed_mean: 1.0,
            length,
        });
        self.usage.push(0);
        self.dirty = true;
        self.refresh_row(id);
        for oid in modified_others {
            self.refresh_row(oid);
        }
        Some(id)
    }

    pub fn refresh_row(&mut self, id: u32) {
        let seg = &self.segments[id as usize];
        
        self.rows[id as usize] = RoadRow {
            ax: seg.a.0 as f32,
            ay: seg.a.1 as f32,
            bx: seg.b.0 as f32,
            by: seg.b.1 as f32,
            
            speed_mean: seg.speed_mean,
            length: seg.length,
            count_a: seg.conn_a.len() as f32,
            count_b: seg.conn_b.len() as f32,

            conn_a0: seg.conn_a.get(0).copied().unwrap_or(0) as f32,
            conn_a1: seg.conn_a.get(1).copied().unwrap_or(0) as f32,
            conn_a2: seg.conn_a.get(2).copied().unwrap_or(0) as f32,
            _pad_a: 0.0,

            conn_b0: seg.conn_b.get(0).copied().unwrap_or(0) as f32,
            conn_b1: seg.conn_b.get(1).copied().unwrap_or(0) as f32,
            conn_b2: seg.conn_b.get(2).copied().unwrap_or(0) as f32,
            _pad_b: 0.0,
        };
    }
}

