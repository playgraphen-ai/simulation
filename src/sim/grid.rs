//! 2D tile grid: zone assignment + cell state.
//!
//! Each tile is one of: Empty, Zoned(ZoneType), Road, Building(id).
//! Zones are painter input — the actual building only materializes once a
//! zoned tile is adjacent to a road and the building-spawner runs.

use bevy::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoneType {
    Residential,
    Office,
    Shop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Biome {
    Water,
    Plains,
    Forest,
    Desert,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tile {
    Empty,
    Water,
    Zone(ZoneType),
    Road(u32),      // index into RoadData.segments
    Building(u32),  // index into BuildingData.items
}

#[derive(Resource, Default)]
pub struct CityGrid {
    pub width: u32,
    pub height: u32,
    pub tiles: Vec<Tile>,
    pub elevations: Vec<f32>,
    pub biomes: Vec<Biome>,
}

impl CityGrid {
    pub fn reset(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
        let count = (w * h) as usize;
        self.tiles = vec![Tile::Empty; count];
        self.elevations = vec![0.0; count];
        self.biomes = vec![Biome::Plains; count];
    }

    #[inline]
    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    pub fn get(&self, x: u32, y: u32) -> Option<Tile> {
        if x < self.width && y < self.height {
            Some(self.tiles[self.idx(x, y)])
        } else {
            None
        }
    }

    pub fn set(&mut self, x: u32, y: u32, t: Tile) {
        if x < self.width && y < self.height {
            let i = self.idx(x, y);
            self.tiles[i] = t;
        }
    }

    /// Returns the four orthogonal neighbours that are within bounds.
    pub fn neighbours4(&self, x: u32, y: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
        let w = self.width;
        let h = self.height;
        [(-1i32, 0), (1, 0), (0, -1), (0, 1)]
            .into_iter()
            .filter_map(move |(dx, dy)| {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
                    Some((nx as u32, ny as u32))
                } else {
                    None
                }
            })
    }
}
