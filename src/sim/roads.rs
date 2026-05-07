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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum RoadType {
    #[default]
    Normal,
    Highway,
}

pub const ROAD_CAPACITY: u32 = 8192;
pub const TEXELS_PER_SEGMENT: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RoadRow {
    // Texel 0: Endpoints of the super-segment
    pub ax: f32,
    pub ay: f32,
    pub bx: f32,
    pub by: f32,
    // Texel 1: Stats & Meta
    pub speed_mean: f32,
    pub links_offset: f32,
    pub links_count: f32,
    pub length: f32,
    // Texel 2: Connections A
    pub conn_a0: f32,
    pub conn_a1: f32,
    pub conn_a2: f32,
    pub count_a: f32,
    // Texel 3: Connections B
    pub conn_b0: f32,
    pub conn_b1: f32,
    pub conn_b2: f32,
    pub count_b: f32,
    // Texel 4: Extra metadata
    pub road_type: f32, // 0 for Normal, 1 for Highway
    pub unused1: f32,
    pub unused2: f32,
    pub unused3: f32,
}

/// A single link between two adjacent tiles.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RoadLink {
    pub a: (u32, u32),
    pub b: (u32, u32),
    pub road_type: RoadType,
}

/// A super-segment consisting of multiple links between two junctions/ends.
#[derive(Clone, Debug, Default)]
pub struct RoadSegment {
    pub a: (u32, u32),
    pub b: (u32, u32),
    /// The ordered list of tiles making up this segment, from A to B.
    pub points: Vec<(u32, u32)>,
    /// Up to 3 neighbour segment ids at end A.
    pub conn_a: Vec<u32>,
    /// Up to 3 neighbour segment ids at end B.
    pub conn_b: Vec<u32>,
    pub speed_mean: f32,
    pub length: f32,
    pub links_offset: u32,
    pub road_type: RoadType,
}

#[derive(Resource)]
pub struct RoadData {
    /// The super-segments used for pathfinding.
    pub segments: Vec<RoadSegment>,
    /// All points (tiles) of all segments, packed for the GPU.
    pub all_points: Vec<(u32, u32)>,
    pub rows: Vec<RoadRow>,
    pub tex_width: u32,
    pub tex_height: u32,
    pub dirty: bool,
    /// Usage counter per segment (incremented whenever a person crosses it).
    pub usage: Vec<u32>,
    
    /// Internal representation of all placed road links before grouping.
    pub links: Vec<RoadLink>,
}

impl Default for RoadData {
    fn default() -> Self {
        let total_texels = ROAD_CAPACITY * TEXELS_PER_SEGMENT;
        let side = (total_texels as f32).sqrt().ceil() as u32;
        let width = ((side + TEXELS_PER_SEGMENT - 1) / TEXELS_PER_SEGMENT) * TEXELS_PER_SEGMENT;
        let height = (total_texels + width - 1) / width;
        Self {
            segments: Vec::new(),
            all_points: Vec::new(),
            rows: vec![RoadRow::default(); ROAD_CAPACITY as usize],
            tex_width: width,
            tex_height: height,
            dirty: true,
            usage: Vec::new(),
            links: Vec::new(),
        }
    }
}

impl RoadData {
    /// Add a raw link between two tiles. Does NOT update the GPU until rebuild_topology is called.
    pub fn push_link(&mut self, a: (u32, u32), b: (u32, u32), road_type: RoadType) {
        if a == b {
            // Self-links are only to ensure isolated tiles exist in the system.
        }
        if self.links.contains(&RoadLink { a, b, road_type }) || self.links.contains(&RoadLink { a: b, b: a, road_type }) {
            return;
        }
        self.links.push(RoadLink { a, b, road_type });
        self.dirty = true;
    }

    /// Rebuilds the super-segment topology from the raw links.
    /// Returns a map from link to segment_id for updating the grid.
    pub fn rebuild_topology(&mut self) -> Vec<((u32, u32), u32)> {
        use std::collections::{HashMap, HashSet};
        
        let mut adj: HashMap<(u32, u32), Vec<(u32, u32, RoadType)>> = HashMap::new();
        for link in &self.links {
            adj.entry(link.a).or_default().push((link.b.0, link.b.1, link.road_type));
            adj.entry(link.b).or_default().push((link.a.0, link.a.1, link.road_type));
        }

        // A junction is any point with != 2 neighbours OR where neighbour types differ.
        let junctions: HashSet<(u32, u32)> = adj.iter()
            .filter(|(_, neighbors)| {
                if neighbors.len() != 2 { return true; }
                neighbors[0].2 != neighbors[1].2
            })
            .map(|(&pos, _)| pos)
            .collect();

        let mut visited_links = HashSet::new();
        let mut new_segments = Vec::new();
        let mut tile_to_seg = Vec::new();

        // Start from each junction and follow paths
        for &start_junction in &junctions {
            if let Some(neighbors) = adj.get(&start_junction) {
                for &(nx, ny, rtype) in neighbors {
                    let neighbor = (nx, ny);
                    let link = if start_junction < neighbor { (start_junction, neighbor, rtype) } else { (neighbor, start_junction, rtype) };
                    if visited_links.contains(&link) { continue; }
                    
                    // Follow the path
                    let mut path = vec![start_junction, neighbor];
                    visited_links.insert(link);
                    
                    let mut current = neighbor;
                    let mut prev = start_junction;
                    
                    while !junctions.contains(&current) {
                        let nexts = &adj[&current];
                        let next_info = if (nexts[0].0, nexts[0].1) == prev { nexts[1] } else { nexts[0] };
                        let next = (next_info.0, next_info.1);
                        let next_link = if current < next { (current, next, rtype) } else { (next, current, rtype) };
                        
                        path.push(next);
                        visited_links.insert(next_link);
                        prev = current;
                        current = next;
                    }
                    
                    let length = (path.len() as f32 - 1.0).max(1.0);
                    let seg_id = new_segments.len() as u32;
                    
                    for &pos in &path {
                        tile_to_seg.push((pos, seg_id));
                    }

                    new_segments.push(RoadSegment {
                        a: start_junction,
                        b: current,
                        points: path,
                        conn_a: Vec::new(),
                        conn_b: Vec::new(),
                        speed_mean: if rtype == RoadType::Highway { 2.0 } else { 1.0 },
                        length,
                        links_offset: 0,
                        road_type: rtype,
                    });
                }
            }
        }

        // Handle isolated loops (no junctions)
        for link_obj in &self.links {
            let link = if link_obj.a < link_obj.b { (link_obj.a, link_obj.b, link_obj.road_type) } else { (link_obj.b, link_obj.a, link_obj.road_type) };
            if visited_links.contains(&link) { continue; }

            // This must be part of a loop. Pick an arbitrary start.
            let mut path = vec![link_obj.a, link_obj.b];
            visited_links.insert(link);
            let mut current = link_obj.b;
            let mut prev = link_obj.a;
            let rtype = link_obj.road_type;
            
            while current != link_obj.a {
                let nexts = &adj[&current];
                let next_info = if (nexts[0].0, nexts[0].1) == prev { nexts[1] } else { nexts[0] };
                let next = (next_info.0, next_info.1);
                let next_link = if current < next { (current, next, rtype) } else { (next, current, rtype) };
                path.push(next);
                visited_links.insert(next_link);
                prev = current;
                current = next;
            }

            let length = (path.len() as f32 - 1.0).max(1.0);
            let seg_id = new_segments.len() as u32;
            for &pos in &path {
                tile_to_seg.push((pos, seg_id));
            }
            new_segments.push(RoadSegment {
                a: link_obj.a,
                b: link_obj.a,
                points: path,
                conn_a: Vec::new(),
                conn_b: Vec::new(),
                speed_mean: if rtype == RoadType::Highway { 2.0 } else { 1.0 },
                length,
                links_offset: 0,
                road_type: rtype,
            });
        }

        // Wire connections between super-segments
        let mut pos_to_segs: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (id, seg) in new_segments.iter().enumerate() {
            pos_to_segs.entry(seg.a).or_default().push(id as u32);
            pos_to_segs.entry(seg.b).or_default().push(id as u32);
        }

        for id in 0..new_segments.len() {
            let seg_a = new_segments[id].a;
            let seg_b = new_segments[id].b;
            
            let mut conn_a = Vec::new();
            if let Some(others) = pos_to_segs.get(&seg_a) {
                for &oid in others {
                    if oid != id as u32 && conn_a.len() < 3 {
                        conn_a.push(oid);
                    }
                }
            }
            
            let mut conn_b = Vec::new();
            if let Some(others) = pos_to_segs.get(&seg_b) {
                for &oid in others {
                    if oid != id as u32 && conn_b.len() < 3 {
                        conn_b.push(oid);
                    }
                }
            }
            
            new_segments[id].conn_a = conn_a;
            new_segments[id].conn_b = conn_b;
        }

        // Flatten points for GPU
        self.all_points.clear();
        for seg in new_segments.iter_mut() {
            seg.links_offset = self.all_points.len() as u32;
            for &p in &seg.points {
                self.all_points.push(p);
            }
        }

        self.segments = new_segments;
        self.usage.resize(self.segments.len(), 0);
        self.rows.fill(RoadRow::default());
        for i in 0..self.segments.len() {
            self.refresh_row(i as u32);
        }
        self.dirty = true;
        
        tile_to_seg
    }

    pub fn get_tile_t(&self, seg_id: u32, tile: (u32, u32)) -> f32 {
        if let Some(seg) = self.segments.get(seg_id as usize) {
            if let Some(pos) = seg.points.iter().position(|&p| p == tile) {
                if seg.points.len() > 1 {
                    return pos as f32 / (seg.points.len() - 1) as f32;
                }
            }
        }
        0.0
    }

    pub fn refresh_row(&mut self, id: u32) {
        let seg = &self.segments[id as usize];
        
        self.rows[id as usize] = RoadRow {
            ax: seg.a.0 as f32,
            ay: seg.a.1 as f32,
            bx: seg.b.0 as f32,
            by: seg.b.1 as f32,
            
            speed_mean: seg.speed_mean,
            links_offset: seg.links_offset as f32,
            links_count: seg.points.len() as f32,
            length: seg.length,

            conn_a0: seg.conn_a.get(0).copied().unwrap_or(0) as f32,
            conn_a1: seg.conn_a.get(1).copied().unwrap_or(0) as f32,
            conn_a2: seg.conn_a.get(2).copied().unwrap_or(0) as f32,
            count_a: seg.conn_a.len() as f32,

            conn_b0: seg.conn_b.get(0).copied().unwrap_or(0) as f32,
            conn_b1: seg.conn_b.get(1).copied().unwrap_or(0) as f32,
            conn_b2: seg.conn_b.get(2).copied().unwrap_or(0) as f32,
            count_b: seg.conn_b.len() as f32,

            road_type: match seg.road_type {
                RoadType::Normal => 0.0,
                RoadType::Highway => 1.0,
            },
            ..default()
        };
    }
}

