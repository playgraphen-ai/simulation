//! Starter scenario: builds a test city at startup so we can iterate without
//! click-painting every run. Layout:
//!
//!   - one long horizontal trunk road across the middle of the map
//!   - three perpendicular side streets ("Residential", "Office", "Shop")
//!     branching north off the trunk
//!   - every tile adjacent to a side street is zoned with that street's type,
//!     and the road tool auto-materialises buildings on both sides
//!   - N people spawned straight away
//!
//! Runs once at Startup after the grid is reset.

use bevy::prelude::*;

use super::{
    buildings::{spawn_building, BuildingData},
    grid::{CityGrid, Tile, ZoneType},
    roads::{RoadData, RoadType},
    SpawnPeopleRequest,
};
use crate::ui::inspector::{Selection, SelectedObj};

/// Tile coordinates of the trunk and side-streets so other systems can
/// reference them if needed.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ScenarioLayout {
    pub entry_seg: u32,
}

pub fn build_starter_scenario(
    mut grid: ResMut<CityGrid>,
    mut roads: ResMut<RoadData>,
    mut buildings: ResMut<BuildingData>,
    mut commands: Commands,
    mut spawn_ev: MessageWriter<SpawnPeopleRequest>,
    settings: Res<crate::ui::menu::MenuSettings>,
) {
    let block_size = 8;
    let grid_size_x = settings.grid_x;
    let grid_size_y = settings.grid_y;
    
    // Define 64 distant origin points for the 64 cities, spaced 100 units apart
    let mut offsets = Vec::new();
    let spacing = 100;
    let base_x = 100;
    let base_y = 100;
    for cy in 0..8 {
        for cx in 0..8 {
            offsets.push((base_x + cx * spacing, base_y + cy * spacing));
        }
    }

    let mut first_building_id = None;
    let mid_gy = grid_size_y / 2;

    for &(start_x, start_y) in &offsets {
    // 1. Build the grid of roads
    // Horizontal roads
    for gy in 0..=grid_size_y {
        let y = start_y + gy * block_size;
        let mut prev = None;
        
        // For the middle horizontal road, extend it slightly to the left as an entry point
        let row_start_x = if gy == mid_gy { start_x.saturating_sub(10) } else { start_x };
        let row_end_x = start_x + grid_size_x * block_size;

        for x in row_start_x..=row_end_x {
            if grid.get(x, y) == Some(Tile::Water) {
                prev = None; // break the road
                continue;
            }
            let here = (x, y);
            if let Some(other) = prev {
                roads.push_link(here, other, RoadType::Normal);
            } else {
                roads.push_link(here, here, RoadType::Normal);
            }
            prev = Some(here);
        }
    }
    // Vertical roads
    for gx in 0..=grid_size_x {
        let x = start_x + gx * block_size;
        let mut prev = None;
        for y in start_y..=(start_y + grid_size_y * block_size) {
            if grid.get(x, y) == Some(Tile::Water) {
                prev = None;
                continue;
            }
            let here = (x, y);
            if let Some(other) = prev {
                roads.push_link(here, other, RoadType::Normal);
            } else {
                roads.push_link(here, here, RoadType::Normal);
            }
            prev = Some(here);
        }
    }
}

let city_w = grid_size_x * block_size;
let city_h = grid_size_y * block_size;
let mid_ox = (grid_size_x / 2) * block_size;
let mid_oy = mid_gy * block_size;

let gap_w = spacing - city_w;
let gap_h = spacing - city_h;

let x_min = offsets[0].0.saturating_sub(20);
let x_max = offsets[63].0 + city_w + 20;
let y_min = offsets[0].1.saturating_sub(20);
let y_max = offsets[63].1 + city_h + 20;

// Continuous highways
for i in 0..7 {
    // Vertical
    let hx = offsets[0].0 + i * spacing + city_w + gap_w / 2;
    for y in y_min..=y_max {
        roads.push_link((hx, y), (hx, y.saturating_sub(1).max(y_min)), RoadType::Highway);
    }
    
    // Horizontal
    let hy = offsets[0].1 + i * spacing + city_h + gap_h / 2;
    for x in x_min..=x_max {
        roads.push_link((x, hy), (x.saturating_sub(1).max(x_min), hy), RoadType::Highway);
    }
}

// Connect the 64 cities to the highways
for cy in 0..8 {
    for cx in 0..8 {
        let idx = (cy * 8 + cx) as usize;
        let current = offsets[idx];
        
        // Connect Right to vertical highway
        if cx < 7 {
            let hx = current.0 + city_w + gap_w / 2;
            let x_start = current.0 + city_w;
            let y = current.1 + mid_oy;
            for x in x_start..=hx {
                roads.push_link((x, y), (x.saturating_sub(1).max(x_start), y), RoadType::Highway);
            }
        }
        
        // Connect Left to vertical highway
        if cx > 0 {
            let hx = current.0 - gap_w / 2;
            let x_start = hx;
            let x_end = current.0;
            let y = current.1 + mid_oy;
            for x in x_start..=x_end {
                roads.push_link((x, y), (x.saturating_sub(1).max(x_start), y), RoadType::Highway);
            }
        }

        // Connect Down to horizontal highway
        if cy < 7 {
            let hy = current.1 + city_h + gap_h / 2;
            let y_start = current.1 + city_h;
            let x = current.0 + mid_ox;
            for y in y_start..=hy {
                roads.push_link((x, y), (x, y.saturating_sub(1).max(y_start)), RoadType::Highway);
            }
        }
        
        // Connect Up to horizontal highway
        if cy > 0 {
            let hy = current.1 - gap_h / 2;
            let y_start = hy;
            let y_end = current.1;
            let x = current.0 + mid_ox;
            for y in y_start..=y_end {
                roads.push_link((x, y), (x, y.saturating_sub(1).max(y_start)), RoadType::Highway);
            }
        }
    }
}

let tile_to_seg = roads.rebuild_topology();
    let mut entry_seg = 0;
    
    let entry_x = offsets[0].0.saturating_sub(10);
    let entry_y = offsets[0].1 + mid_gy * block_size;

    for ((tx, ty), seg_id) in tile_to_seg {
        let rtype = roads.segments[seg_id as usize].road_type;
        if rtype == RoadType::Highway {
            for dx in -1..=2 {
                for dy in -1..=2 {
                    let nx = tx as i32 + dx;
                    let ny = ty as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as u32) < grid.width && (ny as u32) < grid.height {
                        grid.set(nx as u32, ny as u32, Tile::Road(seg_id));
                    }
                }
            }
        } else {
            for dx in 0..=1 {
                for dy in 0..=1 {
                    let nx = tx as u32 + dx;
                    let ny = ty as u32 + dy;
                    if nx < grid.width && ny < grid.height {
                        grid.set(nx, ny, Tile::Road(seg_id));
                    }
                }
            }
        }
        if tx == entry_x && ty == entry_y {
            entry_seg = seg_id;
        }
    }

    commands.insert_resource(ScenarioLayout { entry_seg });

    // 2. Zone the blocks for all cities
    for &(start_x, start_y) in &offsets {
        for gy in 0..grid_size_y {
            for gx in 0..grid_size_x {
                // Pseudo-random distribution based on coordinates to get 40% res, 30% office, 30% shop
                let pseudo_rand = (gx * 7 + gy * 13 + start_x + start_y) % 10;
                let zone = match pseudo_rand {
                    0..=3 => ZoneType::Residential, // 40%
                    4..=6 => ZoneType::Office,      // 30%
                    _ => ZoneType::Shop,            // 30%
                };

                let bx = start_x + gx * block_size;
                let by = start_y + gy * block_size;

                let b_size = match zone {
                    ZoneType::Residential => 3,
                    _ => 4,
                };
                
                // Start a bit inside the block to avoid the road's footprint (0..1 for normal roads)
                // Normal roads occupy +0, +1. Highways occupy -1, +0, +1, +2.
                // Since cities use Normal roads internally, we can start at offset 2 and go up to block_size - b_size.
                let mut placed_any = false;
                for dy in 2..=(block_size.saturating_sub(b_size)) {
                    for dx in 2..=(block_size.saturating_sub(b_size)) {
                        let tx = bx + dx;
                        let ty = by + dy;
                        
                        let mut nearest_road = None;
                        // Find if there's an adjacent road
                        'outer: for oy in 0..b_size {
                            for ox in 0..b_size {
                                if tx + ox >= grid.width || ty + oy >= grid.height { continue; }
                                for (nx, ny) in grid.neighbours4(tx + ox, ty + oy) {
                                    if let Some(Tile::Road(sid)) = grid.get(nx, ny) {
                                        nearest_road = Some((sid, (nx, ny)));
                                        break 'outer;
                                    }
                                }
                            }
                        }

                        if let Some((seg_id, road_tile)) = nearest_road {
                            let road_t = roads.get_tile_t(seg_id, road_tile);
                            // Set zoning before trying to spawn so it overrides empty tiles
                            for oy in 0..b_size {
                                for ox in 0..b_size {
                                    if tx + ox < grid.width && ty + oy < grid.height {
                                        if grid.get(tx + ox, ty + oy) == Some(Tile::Empty) {
                                            grid.set(tx + ox, ty + oy, Tile::Zone(zone));
                                        }
                                    }
                                }
                            }
                            if let Some(bid) = spawn_building(&mut buildings, &mut grid, (tx, ty), zone, seg_id, road_t) {
                                if first_building_id.is_none() {
                                    first_building_id = Some(bid);
                                }
                                placed_any = true;
                                // Skip forward to not overlap with this building
                                break; 
                            }
                        }
                    }
                    if placed_any && b_size == 4 {
                        break; // Only place one 4x4 building per block to avoid crowding
                    }
                }
            }
        }
    }

    if let Some(bid) = first_building_id {
        commands.insert_resource(Selection {
            obj: Some(SelectedObj::Building(bid)),
        });
    }

    grid.set_changed();

    spawn_ev.write(SpawnPeopleRequest { count: settings.population });

    info!(
        "64 cities built, each {}x{} blocks, connected. {} people spawned.",
        grid_size_x, grid_size_y, settings.population
    );
}
