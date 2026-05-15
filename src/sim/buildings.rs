//! Building data as an f32 texture.
//!
//! Each building = 3 RGBA32F texels (12 floats).
//! Texel 0: [x, y, btype, level]
//! Texel 1: [income_or_rent, occupants, capacity, road_segment_id]
//! Texel 2: [growth, age_seconds, road_t, assigned]
//!
//! btype: 0=Residential, 1=Office, 2=Shop.
//! level: 0..4 (five visual stages from undeveloped to dense).

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::sim::grid::{CityGrid, Tile, ZoneType};

use crate::sim::constants::MAX_BUILDINGS;
pub const BUILDING_CAPACITY: u32 = MAX_BUILDINGS;
pub const TEXELS_PER_BUILDING: u32 = 3;
pub const MAX_LEVEL: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BuildingRow {
    pub x: f32,
    pub y: f32,
    pub btype: f32,
    pub level: f32,
    pub income: f32,
    pub occupants: f32,
    pub capacity: f32,
    pub road_seg: f32,
    pub growth: f32,
    pub age_seconds: f32,
    pub road_t: f32,
    pub assigned: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Building {
    pub tile: (u32, u32),
    pub btype: ZoneType,
    pub level: u32,
    pub occupants: u32,
    pub assigned: u32,
    pub capacity: u32,
    pub income: f32,
    pub road_seg: u32,
    pub road_t: f32,
    /// Accumulator controlling upgrade / abandon cycles.
    pub growth: f32,
    /// Seconds since this building was constructed (grace period for abandon).
    pub age_seconds: f32,
}

#[derive(Resource)]
pub struct BuildingData {
    pub items: Vec<Building>,
    pub rows: Vec<BuildingRow>,
    pub tex_width: u32,
    pub tex_height: u32,
    pub dirty: bool,
}

impl Default for BuildingData {
    fn default() -> Self {
        let total = BUILDING_CAPACITY * TEXELS_PER_BUILDING;
        let side = (total as f32).sqrt().ceil() as u32;
        let width = ((side + TEXELS_PER_BUILDING - 1) / TEXELS_PER_BUILDING) * TEXELS_PER_BUILDING;
        let height = (total + width - 1) / width;
        Self {
            items: Vec::new(),
            rows: vec![BuildingRow::default(); BUILDING_CAPACITY as usize],
            tex_width: width,
            tex_height: height,
            dirty: true,
        }
    }
}

impl BuildingData {
    pub fn push(&mut self, b: Building) -> Option<u32> {
        if self.items.len() as u32 >= BUILDING_CAPACITY {
            return None;
        }
        let id = self.items.len() as u32;
        self.items.push(b);
        self.refresh_row(id);
        self.dirty = true;
        Some(id)
    }

    pub fn refresh_row(&mut self, id: u32) {
        let b = &self.items[id as usize];
        let size = match b.btype {
            ZoneType::Residential => 3.0,
            _ => 4.0,
        };
        let offset = size / 2.0;

        self.rows[id as usize] = BuildingRow {
            x: b.tile.0 as f32 + offset,
            y: b.tile.1 as f32 + offset,
            btype: match b.btype {
                ZoneType::Residential => 0.0,
                ZoneType::Office => 1.0,
                ZoneType::Shop => 2.0,
            },
            level: b.level as f32,
            income: b.income,
            occupants: b.occupants as f32,
            capacity: b.capacity as f32,
            road_seg: b.road_seg as f32,
            growth: b.growth,
            age_seconds: b.age_seconds,
            road_t: b.road_t,
            assigned: b.assigned as f32,
        };
    }
}

/// Capacity by level: each upgrade roughly doubles capacity.
fn capacity_for(btype: ZoneType, level: u32) -> u32 {
    let base = match btype {
        ZoneType::Residential => 72, // 9x for 3x3
        ZoneType::Office => 192,  // 16x for 4x4
        ZoneType::Shop => 256,   // 16x for 4x4
    };
    base * (1 << level)
}

/// Income/rent by level: higher levels yield more.
fn income_for(btype: ZoneType, level: u32) -> f32 {
    let base = match btype {
        ZoneType::Residential => 18.0,  // 9x for 3x3
        ZoneType::Office => 80.0,   // 16x for 4x4
        ZoneType::Shop => 48.0,     // 16x for 4x4
    };
    base * (level as f32 + 1.0)
}

/// Public helper used by the road/zone construction systems to materialize a
/// building on a zoned area adjacent to a given road segment.
pub fn spawn_building(
    data: &mut BuildingData,
    grid: &mut CityGrid,
    tile: (u32, u32),
    btype: ZoneType,
    road_seg: u32,
    road_t: f32,
) -> Option<u32> {
    let size = match btype {
        ZoneType::Residential => 3,
        _ => 4,
    };

    // Check if area is clear and within bounds
    for dy in 0..size {
        for dx in 0..size {
            let tx = tile.0 + dx;
            let ty = tile.1 + dy;
            if tx >= grid.width || ty >= grid.height { return None; }
            match grid.get(tx, ty) {
                Some(Tile::Empty) | Some(Tile::Zone(_)) => {},
                _ => return None, // Already something here
            }
        }
    }

    let level = 0;
    let id = data.push(Building {
        tile,
        btype,
        level,
        occupants: 0,
        assigned: 0,
        capacity: capacity_for(btype, level),
        income: income_for(btype, level),
        road_seg,
        road_t,
        growth: 0.0,
        age_seconds: 0.0,
    })?;

    for dy in 0..size {
        for dx in 0..size {
            grid.set(tile.0 + dx, tile.1 + dy, Tile::Building(id));
        }
    }
    Some(id)
}
